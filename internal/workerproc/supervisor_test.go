package workerproc

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestPersistentSupervisorRestartsCrashedWorkerForHealth(t *testing.T) {
	worker := writeWorkerScript(t, `#!/bin/sh
read request || exit 0
printf '%s\n' '{"requestID":"request-1","ok":true,"result":{"status":"ok"}}'
while read request; do
  printf '%s\n' '{"requestID":"request-2","ok":true,"result":{"shutdown":true}}'
  exit 0
done
`)
	supervisor, err := StartPersistentSupervisor(worker)
	if err != nil {
		t.Fatal(err)
	}
	defer terminateSupervisor(t, supervisor)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	initial, err := supervisor.Call(ctx, PersistentRequest{Command: "health"})
	if err != nil {
		t.Fatalf("initial health: %v", err)
	}
	if initial.WorkerObservationSequence != 1 {
		t.Fatalf("initial observation sequence = %d", initial.WorkerObservationSequence)
	}
	if err := supervisor.Kill(); err != nil {
		t.Fatalf("kill: %v", err)
	}
	restarted, err := supervisor.Call(ctx, PersistentRequest{Command: "health"})
	if err != nil {
		t.Fatalf("health after restart: %v", err)
	}
	if restarted.WorkerObservationSequence != 2 {
		t.Fatalf("restarted observation sequence = %d", restarted.WorkerObservationSequence)
	}
	if supervisor.RestartCount() != 1 {
		t.Fatalf("restart count = %d", supervisor.RestartCount())
	}
}

func TestPersistentSupervisorRestartsAfterInferenceTimeout(t *testing.T) {
	worker := writeWorkerScript(t, `#!/bin/sh
read request || exit 0
printf '%s\n' '{"requestID":"request-1","ok":true,"result":{"status":"ok"}}'
read request || exit 0
sleep 5
`)
	supervisor, err := StartPersistentSupervisor(worker)
	if err != nil {
		t.Fatal(err)
	}
	defer terminateSupervisor(t, supervisor)
	healthContext, cancelHealth := context.WithTimeout(context.Background(), 2*time.Second)
	if _, err := supervisor.Call(healthContext, PersistentRequest{Command: "health"}); err != nil {
		t.Fatalf("initial health: %v", err)
	}
	cancelHealth()
	inferenceContext, cancelInference := context.WithTimeout(context.Background(), 50*time.Millisecond)
	_, err = supervisor.Call(inferenceContext, PersistentRequest{
		Command: "decode",
		Forward: &PersistentForwardRequest{
			ShardID: "shard", SequenceID: "sequence", InputKind: "tokens",
			Input: WireTensor{Shape: []int{1, 1}, DType: "int32", Data: []byte{1, 0, 0, 0}},
		},
	})
	cancelInference()
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("decode error = %v", err)
	}
	recoveryContext, cancelRecovery := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancelRecovery()
	if _, err := supervisor.Call(recoveryContext, PersistentRequest{Command: "health"}); err != nil {
		t.Fatalf("recovery health: %v", err)
	}
	if supervisor.RestartCount() != 1 {
		t.Fatalf("restart count = %d", supervisor.RestartCount())
	}
}

func TestPersistentSupervisorDoesNotRestartForInvalidRequest(t *testing.T) {
	worker := writeWorkerScript(t, `#!/bin/sh
while read request; do
  printf '%s\n' '{"requestID":"request-1","ok":true,"result":{"shutdown":true}}'
  exit 0
done
`)
	supervisor, err := StartPersistentSupervisor(worker)
	if err != nil {
		t.Fatal(err)
	}
	defer terminateSupervisor(t, supervisor)
	if _, err := supervisor.Call(context.Background(), PersistentRequest{}); err == nil {
		t.Fatal("empty command succeeded")
	}
	if _, err := supervisor.Call(context.Background(), PersistentRequest{Command: "decode"}); !errors.Is(err, ErrInferenceDeadlineRequired) {
		t.Fatalf("decode error = %v", err)
	}
	if supervisor.RestartCount() != 0 {
		t.Fatalf("restart count = %d", supervisor.RestartCount())
	}
}

func TestPersistentSupervisorDoesNotRestartWhenQueuedCallExpires(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "first-call-started")
	worker := writeWorkerScript(t, fmt.Sprintf(`#!/bin/sh
read request || exit 0
: > %q
sleep 0.2
printf '%%s\n' '{"requestID":"request-1","ok":true,"result":{"status":"ok"}}'
while read request; do
  printf '%%s\n' '{"requestID":"request-2","ok":true,"result":{"status":"ok"}}'
done
`, marker))
	supervisor, err := StartPersistentSupervisor(worker)
	if err != nil {
		t.Fatal(err)
	}
	defer terminateSupervisor(t, supervisor)

	firstResult := make(chan error, 1)
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_, callErr := supervisor.Call(ctx, PersistentRequest{Command: "health"})
		firstResult <- callErr
	}()
	waitForFile(t, marker)

	queuedContext, cancelQueued := context.WithTimeout(context.Background(), 40*time.Millisecond)
	started := time.Now()
	_, err = supervisor.Call(queuedContext, PersistentRequest{
		Command: "decode",
		Forward: &PersistentForwardRequest{
			ShardID: "shard", SequenceID: "sequence", InputKind: "tokens",
			Input: WireTensor{Shape: []int{1, 1}, DType: "int32", Data: []byte{1, 0, 0, 0}},
		},
	})
	cancelQueued()
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("queued decode error = %v", err)
	}
	if elapsed := time.Since(started); elapsed > 150*time.Millisecond {
		t.Fatalf("queued decode observed deadline after %v", elapsed)
	}
	if err := <-firstResult; err != nil {
		t.Fatalf("first health: %v", err)
	}
	if supervisor.RestartCount() != 0 {
		t.Fatalf("queued call restarted healthy worker %d times", supervisor.RestartCount())
	}

	healthContext, cancelHealth := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancelHealth()
	if _, err := supervisor.Call(healthContext, PersistentRequest{Command: "health"}); err != nil {
		t.Fatalf("health after queued deadline: %v", err)
	}
}

func waitForFile(t *testing.T, path string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		if _, err := os.Stat(path); err == nil {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", path)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func terminateSupervisor(t *testing.T, supervisor *PersistentSupervisor) {
	t.Helper()
	if err := supervisor.Kill(); err != nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = supervisor.Wait(ctx)
}

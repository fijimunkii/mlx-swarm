package workerproc

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type recordingWriteCloser struct {
	writes [][]byte
}

func (writer *recordingWriteCloser) Write(payload []byte) (int, error) {
	writer.writes = append(writer.writes, append([]byte(nil), payload...))
	return len(payload), nil
}

func (*recordingWriteCloser) Close() error { return nil }

func TestPersistentWriteLoopDiscardsCanceledRequestBeforeWrite(t *testing.T) {
	stdin := &recordingWriteCloser{}
	client := &PersistentClient{
		stdin: stdin, done: make(chan struct{}), writes: make(chan persistentWrite),
		pending: map[string]chan PersistentResponse{
			"canceled": make(chan PersistentResponse, 1),
		},
	}
	go client.writeLoop()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result := make(chan error, 1)
	client.writes <- persistentWrite{
		ctx: ctx, requestID: "canceled", payload: []byte("late open\n"), result: result,
	}
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatalf("write result = %v, want context canceled", err)
	}
	client.mu.Lock()
	_, pending := client.pending["canceled"]
	client.mu.Unlock()
	if pending {
		t.Fatal("canceled unwritten request remained pending")
	}
	if len(stdin.writes) != 0 {
		t.Fatalf("canceled request reached worker: %q", stdin.writes)
	}
	close(client.done)
}

func TestPersistentClientReportsUnexpectedEOF(t *testing.T) {
	worker := writeWorkerScript(t, "#!/bin/sh\nexit 0\n")
	client, err := StartPersistent(worker)
	if err != nil {
		t.Fatalf("StartPersistent: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := client.Wait(ctx); err == nil || !strings.Contains(err.Error(), "unexpected EOF") {
		t.Fatalf("Wait error = %v, want unexpected EOF", err)
	}
}

func TestPersistentClientKillsWorkerThatClosesStdoutWithoutExiting(t *testing.T) {
	worker := writeWorkerScript(t, `#!/bin/sh
exec 1>&-
while read ignored; do :; done
`)
	client, err := StartPersistent(worker)
	if err != nil {
		t.Fatalf("StartPersistent: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	_, err = client.Call(ctx, PersistentRequest{Command: "health"})
	if err == nil || !strings.Contains(err.Error(), "closed stdout without exiting") {
		t.Fatalf("Call error = %v, want bounded stdout EOF error", err)
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Call reached its deadline instead of killing worker: %v", err)
	}
}

func TestPersistentClientAcceptsAcknowledgedShutdown(t *testing.T) {
	worker := writeWorkerScript(t, `#!/bin/sh
read request
printf '%s\n' '{"requestID":"request-1","ok":true,"result":{"shutdown":true}}'
`)
	client, err := StartPersistent(worker)
	if err != nil {
		t.Fatalf("StartPersistent: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := client.Shutdown(ctx); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
	if _, err := client.Call(ctx, PersistentRequest{Command: "state"}); !errors.Is(err, errPersistentWorkerStopped) {
		t.Fatalf("Call after shutdown error = %v, want stopped worker", err)
	}
}

func TestPersistentClientReservesCanceledRequestID(t *testing.T) {
	worker := writeWorkerScript(t, `#!/bin/sh
read first
sleep 0.1
printf '%s\n' '{"requestID":"shared","ok":true,"result":{"status":"ok"}}'
while read request; do
  printf '%s\n' '{"requestID":"shared","ok":true,"result":{"status":"ok"}}'
done
`)
	client, err := StartPersistent(worker)
	if err != nil {
		t.Fatalf("StartPersistent: %v", err)
	}
	defer func() {
		_ = client.Kill()
		waitCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = client.Wait(waitCtx)
	}()

	canceledCtx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	_, err = client.Call(canceledCtx, PersistentRequest{RequestID: "shared", Command: "health"})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("canceled Call error = %v, want deadline exceeded", err)
	}
	if _, err := client.Call(context.Background(), PersistentRequest{RequestID: "shared", Command: "health"}); err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("reused pending ID error = %v, want duplicate", err)
	}

	deadline := time.Now().Add(2 * time.Second)
	for {
		response, err := client.Call(context.Background(), PersistentRequest{RequestID: "shared", Command: "health"})
		if err == nil {
			if response.Result == nil || response.Result.Status != "ok" {
				t.Fatalf("released ID response = %#v", response)
			}
			break
		}
		if !strings.Contains(err.Error(), "duplicate") {
			t.Fatalf("released ID error = %v", err)
		}
		if time.Now().After(deadline) {
			t.Fatal("late response did not release canceled request ID")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestPersistentClientGeneratedIDSkipsCallerReservedID(t *testing.T) {
	worker := writeWorkerScript(t, `#!/bin/sh
read first
sleep 0.1
printf '%s\n' '{"requestID":"request-1","ok":true,"result":{"status":"ok"}}'
read second
printf '%s\n' '{"requestID":"request-2","ok":true,"result":{"status":"ok"}}'
`)
	client, err := StartPersistent(worker)
	if err != nil {
		t.Fatalf("StartPersistent: %v", err)
	}
	defer func() {
		_ = client.Kill()
		waitCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = client.Wait(waitCtx)
	}()

	canceledCtx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	_, err = client.Call(canceledCtx, PersistentRequest{RequestID: "request-1", Command: "health"})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("reserved Call error = %v, want deadline exceeded", err)
	}
	callCtx, cancelCall := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancelCall()
	response, err := client.Call(callCtx, PersistentRequest{Command: "health"})
	if err != nil {
		t.Fatalf("generated-ID Call: %v", err)
	}
	if response.RequestID != "request-2" {
		t.Fatalf("generated request ID = %q, want request-2", response.RequestID)
	}
}

func TestPersistentClientKillsInferenceWorkerAfterDeadline(t *testing.T) {
	worker := writeWorkerScript(t, `#!/bin/sh
read request
printf '%s\n' '{"requestID":"ready","ok":true,"result":{"status":"ok"}}'
while :; do :; done
`)
	client, err := StartPersistent(worker)
	if err != nil {
		t.Fatalf("StartPersistent: %v", err)
	}
	defer func() {
		_ = client.Kill()
		waitCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = client.Wait(waitCtx)
	}()

	readyCtx, cancelReady := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancelReady()
	if _, err := client.Call(readyCtx, PersistentRequest{RequestID: "ready", Command: "health"}); err != nil {
		t.Fatalf("ready Call: %v", err)
	}

	largeRequest := PersistentRequest{
		RequestID: "blocked-write",
		Command:   "forward",
		Forward: &PersistentForwardRequest{
			ShardID:    "shard",
			SequenceID: "sequence",
			InputKind:  "hidden",
			Input: WireTensor{
				Shape: []int{1 << 20},
				DType: "uint8",
				Data:  bytes.Repeat([]byte{1}, 1<<20),
			},
		},
	}
	writeCtx, cancelWrite := context.WithTimeout(context.Background(), 100*time.Millisecond)
	started := time.Now()
	_, err = client.Call(writeCtx, largeRequest)
	cancelWrite()
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("blocked write error = %v, want deadline exceeded", err)
	}
	if elapsed := time.Since(started); elapsed > 500*time.Millisecond {
		t.Fatalf("blocked write cancellation took %v", elapsed)
	}

	waitCtx, cancelWait := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancelWait()
	if waitErr := client.Wait(waitCtx); waitErr == nil || errors.Is(waitErr, context.DeadlineExceeded) {
		t.Fatalf("timed-out inference worker wait error = %v", waitErr)
	}
}

func TestPersistentClientRejectsUnmatchedResponseID(t *testing.T) {
	worker := writeWorkerScript(t, `#!/bin/sh
read request
printf '%s\n' '{"requestID":"unknown","ok":true,"result":{"status":"ok"}}'
while read ignored; do :; done
`)
	client, err := StartPersistent(worker)
	if err != nil {
		t.Fatalf("StartPersistent: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_, err = client.Call(ctx, PersistentRequest{Command: "health"})
	if err == nil || !strings.Contains(err.Error(), "unmatched request ID") {
		t.Fatalf("Call error = %v, want unmatched request ID", err)
	}
	if errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Call reached deadline instead of bounded protocol error: %v", err)
	}
}

func TestPersistentClientKillsWorkerBeforeReportingMalformedResponse(t *testing.T) {
	worker := writeWorkerScript(t, `#!/bin/sh
read request
printf '%s\n' 'not-json'
while read ignored; do :; done
`)
	client, err := StartPersistent(worker)
	if err != nil {
		t.Fatalf("StartPersistent: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_, err = client.Call(ctx, PersistentRequest{Command: "health"})
	if err == nil || !strings.Contains(err.Error(), "decode persistent worker response") {
		t.Fatalf("Call error = %v, want decode error", err)
	}
	if errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Call reached deadline instead of bounded read error: %v", err)
	}
}

func TestCappedTailBufferBoundsAndMarksStderr(t *testing.T) {
	var buffer cappedTailBuffer
	if _, err := buffer.Write([]byte("discarded-prefix")); err != nil {
		t.Fatalf("Write prefix: %v", err)
	}
	payload := bytes.Repeat([]byte("x"), maxPersistentStderrBytes+32)
	copy(payload[len(payload)-4:], "tail")
	if _, err := buffer.Write(payload); err != nil {
		t.Fatalf("Write payload: %v", err)
	}
	got := buffer.String()
	if !strings.HasPrefix(got, "[stderr truncated;") {
		t.Fatalf("stderr marker missing: %q", got[:min(len(got), 80)])
	}
	if !strings.HasSuffix(got, "tail") {
		t.Fatalf("stderr tail missing")
	}
	if len(got) > maxPersistentStderrBytes+128 {
		t.Fatalf("stderr length = %d, want capped tail plus marker", len(got))
	}
}

func writeWorkerScript(t *testing.T, contents string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "fake-worker")
	if err := os.WriteFile(path, []byte(contents), 0o700); err != nil {
		t.Fatalf("write fake worker: %v", err)
	}
	return path
}

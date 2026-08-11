package main

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

type recordingPersistentCaller struct {
	request     workerproc.PersistentRequest
	deadline    time.Time
	hasDeadline bool
}

type recordingPersistentLifecycle struct {
	shutdownErr error
	killed      bool
	waited      bool
}

func (w *recordingPersistentLifecycle) Shutdown(context.Context) error {
	return w.shutdownErr
}

func (w *recordingPersistentLifecycle) Kill() error {
	w.killed = true
	return nil
}

func (w *recordingPersistentLifecycle) Wait(context.Context) error {
	w.waited = true
	return errors.New("worker killed")
}

func (c *recordingPersistentCaller) Call(
	ctx context.Context,
	request workerproc.PersistentRequest,
) (workerproc.PersistentResponse, error) {
	c.request = request
	c.deadline, c.hasDeadline = ctx.Deadline()
	return workerproc.PersistentResponse{RequestID: "daemon-request-1", OK: true}, nil
}

func TestForwardPersistentRequestRequiresAndPropagatesInferenceDeadline(t *testing.T) {
	worker := &recordingPersistentCaller{}
	if _, err := forwardPersistentRequest(
		context.Background(), worker,
		workerproc.PersistentRequest{Command: "decode"},
	); !errors.Is(err, workerproc.ErrInferenceDeadlineRequired) {
		t.Fatalf("missing deadline error = %v", err)
	}
	wireDeadline := time.Now().Add(time.Second).UnixMilli()
	_, err := forwardPersistentRequest(
		context.Background(), worker,
		workerproc.PersistentRequest{
			Command: "decode", DeadlineUnixMillis: wireDeadline,
			Forward: &workerproc.PersistentForwardRequest{},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !worker.hasDeadline || worker.deadline.UnixMilli() != wireDeadline ||
		worker.request.DeadlineUnixMillis != wireDeadline {
		t.Fatalf("deadline was not propagated: caller=%v request=%+v", worker.deadline, worker.request)
	}
}

func TestForwardPersistentRequestRemapsCallerID(t *testing.T) {
	worker := &recordingPersistentCaller{}
	response, err := forwardPersistentRequest(
		context.Background(),
		worker,
		workerproc.PersistentRequest{RequestID: "client-request-1", Command: "health"},
	)
	if err != nil {
		t.Fatalf("forwardPersistentRequest: %v", err)
	}
	if worker.request.RequestID != "" {
		t.Fatalf("worker request ID = %q, want daemon-generated ID", worker.request.RequestID)
	}
	if response.RequestID != "client-request-1" {
		t.Fatalf("response request ID = %q, want caller ID", response.RequestID)
	}
}

func TestForwardPersistentRequestPreservesGeneratedIDWhenCallerOmitsOne(t *testing.T) {
	worker := &recordingPersistentCaller{}
	response, err := forwardPersistentRequest(
		context.Background(),
		worker,
		workerproc.PersistentRequest{Command: "health"},
	)
	if err != nil {
		t.Fatalf("forwardPersistentRequest: %v", err)
	}
	if response.RequestID != "daemon-request-1" {
		t.Fatalf("response request ID = %q, want daemon ID", response.RequestID)
	}
}

func TestShutdownPersistentWorkerKillsAndReapsAfterGracefulFailure(t *testing.T) {
	worker := &recordingPersistentLifecycle{shutdownErr: errors.New("transport failed")}
	shutdownPersistentWorker(worker)
	if !worker.killed {
		t.Fatal("worker was not killed after graceful shutdown failed")
	}
	if !worker.waited {
		t.Fatal("worker was not reaped after kill fallback")
	}
}

func TestShutdownPersistentWorkerDoesNotKillAfterAcknowledgement(t *testing.T) {
	worker := &recordingPersistentLifecycle{}
	shutdownPersistentWorker(worker)
	if worker.killed || worker.waited {
		t.Fatalf("acknowledged shutdown used fallback: killed=%t waited=%t", worker.killed, worker.waited)
	}
}

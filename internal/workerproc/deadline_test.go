package workerproc

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestRequestContextRequiresInferenceDeadline(t *testing.T) {
	_, _, _, err := RequestContext(context.Background(), PersistentRequest{Command: "decode"})
	if !errors.Is(err, ErrInferenceDeadlineRequired) {
		t.Fatalf("RequestContext error = %v", err)
	}
}

func TestRequestContextPropagatesEarliestDeadline(t *testing.T) {
	parent, cancelParent := context.WithTimeout(context.Background(), time.Second)
	defer cancelParent()
	wireDeadline := time.Now().Add(100 * time.Millisecond).UnixMilli()
	ctx, cancel, request, err := RequestContext(parent, PersistentRequest{
		Command: "prefill", DeadlineUnixMillis: wireDeadline,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer cancel()
	deadline, ok := ctx.Deadline()
	if !ok || deadline.UnixMilli() != wireDeadline || request.DeadlineUnixMillis != wireDeadline {
		t.Fatalf("deadline propagation: context=%v request=%d", deadline, request.DeadlineUnixMillis)
	}
}

func TestRequestContextAllowsUndeadlinedControlRequest(t *testing.T) {
	ctx, cancel, request, err := RequestContext(context.Background(), PersistentRequest{Command: "health"})
	if err != nil {
		t.Fatal(err)
	}
	defer cancel()
	if _, ok := ctx.Deadline(); ok || request.DeadlineUnixMillis != 0 {
		t.Fatalf("unexpected control deadline: %+v", request)
	}
}

func TestRequestContextPreservesCallerDeadlineAndRoundsUpWirePrecision(t *testing.T) {
	parentDeadline := time.Now().Add(time.Second).Truncate(time.Millisecond).Add(750 * time.Microsecond)
	parent, cancelParent := context.WithDeadline(context.Background(), parentDeadline)
	defer cancelParent()
	ctx, cancel, request, err := RequestContext(parent, PersistentRequest{Command: "forward"})
	if err != nil {
		t.Fatal(err)
	}
	defer cancel()
	deadline, ok := ctx.Deadline()
	wantWire := parentDeadline.UnixMilli() + 1
	if !ok || !deadline.Equal(parentDeadline) || request.DeadlineUnixMillis != wantWire {
		t.Fatalf(
			"deadline = %v, request=%d, want local=%v wire=%d",
			deadline, request.DeadlineUnixMillis, parentDeadline, wantWire,
		)
	}
}

func TestWireDeadlineUnixMillisDoesNotShortenOneMillisecondTimeout(t *testing.T) {
	started := time.Now().Truncate(time.Millisecond).Add(999 * time.Microsecond)
	deadline := started.Add(time.Millisecond)
	wire := time.UnixMilli(WireDeadlineUnixMillis(deadline))
	if wire.Before(deadline) || wire.Sub(started) < time.Millisecond {
		t.Fatalf("wire deadline %v shortened timeout from %v to %v", wire, started, deadline)
	}
}

type delayedDeadlineContext struct {
	context.Context
	deadline time.Time
}

func (ctx delayedDeadlineContext) Deadline() (time.Time, bool) {
	return ctx.deadline, true
}

func TestContextCompletionErrorObservesDeadlineBeforeTimerCallback(t *testing.T) {
	ctx := delayedDeadlineContext{
		Context: context.Background(), deadline: time.Now().Add(-time.Millisecond),
	}
	if err := contextCompletionError(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("context completion error = %v", err)
	}
}

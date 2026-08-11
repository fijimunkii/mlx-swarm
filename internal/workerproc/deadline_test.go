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

func TestRequestContextAlignsCallerDeadlineToWirePrecision(t *testing.T) {
	parentDeadline := time.Now().Add(time.Second).Truncate(time.Millisecond).Add(750 * time.Microsecond)
	parent, cancelParent := context.WithDeadline(context.Background(), parentDeadline)
	defer cancelParent()
	ctx, cancel, request, err := RequestContext(parent, PersistentRequest{Command: "forward"})
	if err != nil {
		t.Fatal(err)
	}
	defer cancel()
	deadline, ok := ctx.Deadline()
	want := time.UnixMilli(parentDeadline.UnixMilli())
	if !ok || !deadline.Equal(want) || request.DeadlineUnixMillis != want.UnixMilli() {
		t.Fatalf("deadline = %v, request=%d, want %v", deadline, request.DeadlineUnixMillis, want)
	}
}

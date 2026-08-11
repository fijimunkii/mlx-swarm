package workerproc

import (
	"context"
	"errors"
	"fmt"
	"time"
)

var ErrInferenceDeadlineRequired = errors.New("inference request requires a deadline")

// WireDeadlineUnixMillis rounds an exact local deadline up to the wire
// protocol's millisecond precision so serialization never shortens it.
func WireDeadlineUnixMillis(deadline time.Time) int64 {
	millis := deadline.UnixMilli()
	if deadline.After(time.UnixMilli(millis)) {
		millis++
	}
	return millis
}

// contextCompletionError observes an elapsed wall-clock deadline even when
// the context timer callback has not yet been scheduled.
func contextCompletionError(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if deadline, ok := ctx.Deadline(); ok && !deadline.After(time.Now()) {
		return context.DeadlineExceeded
	}
	return nil
}

// RequestContext applies the earliest caller or wire deadline and writes it
// back to inference requests before they cross a process or network boundary.
func RequestContext(
	parent context.Context,
	request PersistentRequest,
) (context.Context, context.CancelFunc, PersistentRequest, error) {
	if err := parent.Err(); err != nil {
		return nil, nil, request, err
	}
	deadline, hasContextDeadline := parent.Deadline()
	deadlineFromContext := hasContextDeadline
	if request.DeadlineUnixMillis > 0 {
		wireDeadline := time.UnixMilli(request.DeadlineUnixMillis)
		if !hasContextDeadline || wireDeadline.Before(deadline) {
			deadline = wireDeadline
			deadlineFromContext = false
		}
	}
	if isInferenceCommand(request.Command) && !hasContextDeadline && request.DeadlineUnixMillis <= 0 {
		return nil, nil, request, fmt.Errorf("%s: %w", request.Command, ErrInferenceDeadlineRequired)
	}
	if deadline.IsZero() {
		return parent, func() {}, request, nil
	}
	if !deadline.After(time.Now()) {
		return nil, nil, request, context.DeadlineExceeded
	}
	if deadlineFromContext {
		request.DeadlineUnixMillis = WireDeadlineUnixMillis(deadline)
	} else {
		request.DeadlineUnixMillis = deadline.UnixMilli()
	}
	ctx, cancel := context.WithDeadline(parent, deadline)
	return ctx, cancel, request, nil
}

func isInferenceCommand(command string) bool {
	switch command {
	case "forward", "prefill", "decode":
		return true
	default:
		return false
	}
}

package workerproc

import (
	"context"
	"errors"
	"fmt"
	"time"
)

var ErrInferenceDeadlineRequired = errors.New("inference request requires a deadline")

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
	if request.DeadlineUnixMillis > 0 {
		wireDeadline := time.UnixMilli(request.DeadlineUnixMillis)
		if !hasContextDeadline || wireDeadline.Before(deadline) {
			deadline = wireDeadline
		}
	}
	if isInferenceCommand(request.Command) && !hasContextDeadline && request.DeadlineUnixMillis <= 0 {
		return nil, nil, request, fmt.Errorf("%s: %w", request.Command, ErrInferenceDeadlineRequired)
	}
	if deadline.IsZero() {
		return parent, func() {}, request, nil
	}
	// The wire protocol has millisecond precision. Make the local timer expire
	// at that same instant so neither side can accept a result that the other
	// already considers late.
	deadline = time.UnixMilli(deadline.UnixMilli())
	if !deadline.After(time.Now()) {
		return nil, nil, request, context.DeadlineExceeded
	}
	request.DeadlineUnixMillis = deadline.UnixMilli()
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

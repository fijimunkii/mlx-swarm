package workerproc

import (
	"context"
	"errors"
)

// State returns the current persistent worker snapshot for either a direct
// process or a remote swarmd target.
func State(ctx context.Context, caller PersistentCaller) (*PersistentWorkerState, error) {
	if caller == nil {
		return nil, errors.New("persistent worker caller is required")
	}
	response, err := caller.Call(ctx, PersistentRequest{Command: "state"})
	if err != nil {
		return nil, err
	}
	if response.Result == nil || response.Result.State == nil {
		return nil, errors.New("state returned no worker snapshot")
	}
	return response.Result.State, nil
}

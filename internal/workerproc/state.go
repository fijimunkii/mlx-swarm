package workerproc

import (
	"context"
	"errors"
)

// StateObservation pairs worker state with the daemon-local call sequence at
// which it was sampled.
type StateObservation struct {
	State               *PersistentWorkerState
	ObservationSequence uint64
}

// State returns the current persistent worker snapshot for either a direct
// process or a remote swarmd target.
func State(ctx context.Context, caller PersistentCaller) (*PersistentWorkerState, error) {
	observation, err := ObserveState(ctx, caller)
	if err != nil {
		return nil, err
	}
	return observation.State, nil
}

// ObserveState returns a state snapshot and its daemon-local call sequence.
// A zero sequence is accepted for legacy and test callers, but cannot prove
// post-mutation ordering to the mesh scheduler.
func ObserveState(ctx context.Context, caller PersistentCaller) (StateObservation, error) {
	if caller == nil {
		return StateObservation{}, errors.New("persistent worker caller is required")
	}
	response, err := caller.Call(ctx, PersistentRequest{Command: "state"})
	if err != nil {
		return StateObservation{}, err
	}
	if response.Result == nil || response.Result.State == nil {
		return StateObservation{}, errors.New("state returned no worker snapshot")
	}
	return StateObservation{
		State:               response.Result.State,
		ObservationSequence: response.WorkerObservationSequence,
	}, nil
}

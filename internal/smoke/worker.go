// Package smoke provides shared request and assertion helpers for smoke proofs.
package smoke

import (
	"context"
	"errors"
	"fmt"

	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

// Call invokes one worker command and includes the command in transport errors.
func Call(
	ctx context.Context,
	caller workerproc.PersistentCaller,
	request workerproc.PersistentRequest,
) (workerproc.PersistentResponse, error) {
	response, err := caller.Call(ctx, request)
	if err != nil {
		return workerproc.PersistentResponse{}, fmt.Errorf("%s: %w", request.Command, err)
	}
	return response, nil
}

// LoadShard loads a shard and requires its resulting snapshot.
func LoadShard(
	ctx context.Context,
	caller workerproc.PersistentCaller,
	request workerproc.PersistentLoadShardRequest,
) (*workerproc.PersistentShardSnapshot, error) {
	response, err := Call(ctx, caller, workerproc.PersistentRequest{
		Command: "loadShard", LoadShard: &request,
	})
	if err != nil {
		return nil, err
	}
	if response.Result == nil || response.Result.Shard == nil {
		return nil, errors.New("loadShard returned no shard snapshot")
	}
	return response.Result.Shard, nil
}

// State returns the required state snapshot from a worker.
func State(
	ctx context.Context,
	caller workerproc.PersistentCaller,
) (*workerproc.PersistentWorkerState, error) {
	response, err := Call(ctx, caller, workerproc.PersistentRequest{Command: "state"})
	if err != nil {
		return nil, err
	}
	if response.Result == nil || response.Result.State == nil {
		return nil, errors.New("state returned no worker snapshot")
	}
	return response.Result.State, nil
}

// RequireNoSequenceState verifies that every caller released retained sequence
// and KV-cache memory.
func RequireNoSequenceState(
	ctx context.Context,
	callers ...workerproc.PersistentCaller,
) error {
	for index, caller := range callers {
		state, err := State(ctx, caller)
		if err != nil {
			return err
		}
		if state.KVCacheBytes != 0 || state.RetainedBytes != 0 {
			return fmt.Errorf(
				"caller %d retained sequence state: kv=%d retained=%d",
				index, state.KVCacheBytes, state.RetainedBytes,
			)
		}
	}
	return nil
}

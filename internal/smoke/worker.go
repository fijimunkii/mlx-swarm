// Package smoke provides shared orchestration for executable smoke proofs.
package smoke

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

// Worker owns either a directly supervised worker or a remote swarmd caller.
type Worker struct {
	Caller workerproc.PersistentCaller
	direct *workerproc.PersistentClient
	closed bool
}

// OpenWorker opens a remote endpoint when endpoint is nonempty and otherwise
// starts a directly supervised worker process.
func OpenWorker(workerPath string, endpoint string) (*Worker, error) {
	if endpoint != "" {
		client, err := workerproc.NewHTTPPersistentClient(endpoint, nil)
		if err != nil {
			return nil, err
		}
		return &Worker{Caller: client}, nil
	}
	client, err := workerproc.StartPersistent(workerPath)
	if err != nil {
		return nil, err
	}
	return &Worker{Caller: client, direct: client}, nil
}

// IsDirect reports whether this harness owns a local worker process.
func (worker *Worker) IsDirect() bool {
	return worker != nil && worker.direct != nil
}

// Shutdown gracefully stops a directly supervised worker. It is a no-op for
// remote workers whose lifecycle belongs to swarmd.
func (worker *Worker) Shutdown(ctx context.Context) error {
	if worker == nil || worker.direct == nil || worker.closed {
		return nil
	}
	if err := worker.direct.Shutdown(ctx); err != nil {
		return err
	}
	worker.closed = true
	return nil
}

// Cleanup releases a directly supervised worker, falling back to a kill when
// it did not complete a graceful shutdown.
func (worker *Worker) Cleanup() {
	if worker == nil || worker.direct == nil || worker.closed {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	if err := worker.direct.Shutdown(ctx); err == nil {
		worker.closed = true
		cancel()
		return
	}
	cancel()
	_ = worker.direct.Kill()
	reapCtx, reapCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer reapCancel()
	_ = worker.direct.Wait(reapCtx)
	worker.closed = true
}

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

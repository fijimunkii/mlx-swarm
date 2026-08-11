package smoke

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

// SequenceTarget identifies one shard participating in a sequence lifecycle.
type SequenceTarget struct {
	Name    string
	Caller  workerproc.PersistentCaller
	ShardID string
	OwnerID string
}

type openedSequence struct {
	target     SequenceTarget
	sequenceID string
}

// SequenceSet tracks a matrix of sequences opened across worker shards.
type SequenceSet struct {
	opened []openedSequence
}

// Cleanup makes a bounded best effort to close every tracked sequence.
func (set *SequenceSet) Cleanup() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = set.Close(ctx)
}

// OpenSequences opens each ID on every target and rolls back prior opens if a
// later open fails.
func OpenSequences(
	ctx context.Context,
	targets []SequenceTarget,
	sequenceIDs ...string,
) (*SequenceSet, error) {
	if len(targets) == 0 {
		return nil, errors.New("sequence targets are empty")
	}
	for index, target := range targets {
		if target.Caller == nil {
			return nil, fmt.Errorf("sequence target %d has no caller", index)
		}
		if target.ShardID == "" {
			return nil, fmt.Errorf("sequence target %d has no shard ID", index)
		}
	}
	if len(sequenceIDs) == 0 {
		return nil, errors.New("sequence IDs are empty")
	}
	seen := make(map[string]struct{}, len(sequenceIDs))
	for _, sequenceID := range sequenceIDs {
		if sequenceID == "" {
			return nil, errors.New("sequence ID is empty")
		}
		if _, exists := seen[sequenceID]; exists {
			return nil, fmt.Errorf("duplicate sequence ID %q", sequenceID)
		}
		seen[sequenceID] = struct{}{}
	}
	ownerBytes := make([]byte, 16)
	if _, err := rand.Read(ownerBytes); err != nil {
		return nil, fmt.Errorf("generate sequence owner: %w", err)
	}
	ownerID := "smoke-owner-" + hex.EncodeToString(ownerBytes)
	for index := range targets {
		if targets[index].OwnerID == "" {
			targets[index].OwnerID = ownerID
		}
	}
	set := &SequenceSet{}
	for _, sequenceID := range sequenceIDs {
		for _, target := range targets {
			err := SequenceCommand(ctx, target, "openSequence", sequenceID)
			var responseErr *workerproc.WorkerResponseError
			// A transport failure is ambiguous: the worker may have applied the
			// open after the caller stopped waiting. The private owner makes it
			// safe to track and close that attempt. A worker rejection is
			// definitive and may refer to another owner's existing sequence.
			if err == nil || !errors.As(err, &responseErr) {
				set.opened = append(set.opened, openedSequence{target: target, sequenceID: sequenceID})
			}
			if err != nil {
				rollbackCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				rollbackErr := set.Close(rollbackCtx)
				cancel()
				if rollbackErr != nil {
					return set, errors.Join(err, fmt.Errorf("rollback opened sequences: %w", rollbackErr))
				}
				return nil, err
			}
		}
	}
	return set, nil
}

// CloseSequence closes one ID on all targets where the set opened it.
func (set *SequenceSet) CloseSequence(ctx context.Context, sequenceID string) error {
	if set == nil {
		return nil
	}
	var closeErr error
	retained := make([]openedSequence, 0, len(set.opened))
	for index := len(set.opened) - 1; index >= 0; index-- {
		opened := set.opened[index]
		if opened.sequenceID != sequenceID {
			retained = append(retained, opened)
			continue
		}
		if err := closeSequence(ctx, opened); err != nil {
			closeErr = errors.Join(closeErr, err)
			retained = append(retained, opened)
		}
	}
	for left, right := 0, len(retained)-1; left < right; left, right = left+1, right-1 {
		retained[left], retained[right] = retained[right], retained[left]
	}
	set.opened = retained
	return closeErr
}

// Close closes all sequences that remain open in reverse order.
func (set *SequenceSet) Close(ctx context.Context) error {
	if set == nil {
		return nil
	}
	var closeErr error
	retained := make([]openedSequence, 0, len(set.opened))
	for index := len(set.opened) - 1; index >= 0; index-- {
		opened := set.opened[index]
		if err := closeSequence(ctx, opened); err != nil {
			closeErr = errors.Join(closeErr, err)
			retained = append(retained, opened)
		}
	}
	for left, right := 0, len(retained)-1; left < right; left, right = left+1, right-1 {
		retained[left], retained[right] = retained[right], retained[left]
	}
	set.opened = retained
	return closeErr
}

func closeSequence(ctx context.Context, opened openedSequence) error {
	err := SequenceCommand(ctx, opened.target, "closeSequence", opened.sequenceID)
	var responseErr *workerproc.WorkerResponseError
	if errors.As(err, &responseErr) && strings.Contains(responseErr.Message, "is not open on shard") {
		return nil
	}
	return err
}

// SequenceCommand sends an open or close command to one target.
func SequenceCommand(
	ctx context.Context,
	target SequenceTarget,
	command string,
	sequenceID string,
) error {
	_, err := Call(ctx, target.Caller, workerproc.PersistentRequest{
		Command: command,
		Sequence: &workerproc.PersistentSequenceRequest{
			ShardID: target.ShardID, SequenceID: sequenceID, OwnerID: target.OwnerID,
		},
	})
	if err == nil || target.Name == "" {
		return err
	}
	return fmt.Errorf("%s: %w", target.Name, err)
}

// InferenceRequest constructs one persistent forward request.
func InferenceRequest(
	command string,
	shardID string,
	sequenceID string,
	position uint64,
	inputKind string,
	input workerproc.WireTensor,
) workerproc.PersistentRequest {
	return workerproc.PersistentRequest{
		Command: command,
		Forward: &workerproc.PersistentForwardRequest{
			ShardID: shardID, SequenceID: sequenceID, Position: position,
			InputKind: inputKind, Input: input,
		},
	}
}

// Infer executes a cache-aware inference command and validates its metadata.
func Infer(
	ctx context.Context,
	caller workerproc.PersistentCaller,
	command string,
	shardID string,
	sequenceID string,
	position uint64,
	inputKind string,
	input workerproc.WireTensor,
) (*workerproc.PersistentForwardResult, error) {
	response, err := Call(ctx, caller, InferenceRequest(
		command, shardID, sequenceID, position, inputKind, input,
	))
	if err != nil {
		return nil, err
	}
	if response.Result == nil || response.Result.Forward == nil {
		return nil, fmt.Errorf("%s returned no tensor", command)
	}
	result := response.Result.Forward
	if result.Operation != command || result.Position != position || result.KVCacheBytes == 0 {
		return nil, fmt.Errorf(
			"%s returned inconsistent metadata: operation=%s position=%d kv=%d",
			command, result.Operation, result.Position, result.KVCacheBytes,
		)
	}
	return result, nil
}

// ExpectWorkerError reports whether a request failed with the expected worker
// response text.
func ExpectWorkerError(
	ctx context.Context,
	caller workerproc.PersistentCaller,
	request workerproc.PersistentRequest,
	want string,
) bool {
	_, err := caller.Call(ctx, request)
	var responseError *workerproc.WorkerResponseError
	return errors.As(err, &responseError) && strings.Contains(responseError.Message, want)
}

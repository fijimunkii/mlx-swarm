package workerproc

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"time"
)

const sequenceCleanupTimeout = 10 * time.Second

// SequenceTarget identifies one shard participating in a sequence lifecycle.
type SequenceTarget struct {
	Name    string
	Caller  PersistentCaller
	ShardID string
}

type openedSequence struct {
	target     SequenceTarget
	sequenceID string
	ownerID    string
}

// SequenceSet tracks a matrix of sequences opened across worker shards.
type SequenceSet struct {
	opened []openedSequence
}

// OpenSequences opens each ID on every target with one private owner and rolls
// back prior or ambiguous opens if a later open fails. A non-nil set returned
// with an error still owns sequences whose rollback failed and must be cleaned.
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
	ownerID := "sequence-owner-" + hex.EncodeToString(ownerBytes)
	set := &SequenceSet{}
	for _, sequenceID := range sequenceIDs {
		for _, target := range targets {
			err := sequenceCommand(ctx, target, "openSequence", sequenceID, ownerID)
			var responseErr *WorkerResponseError
			// Transport failures are ambiguous: the worker may apply the open
			// after the caller stops waiting. The private owner makes the
			// attempt safe to close. A worker rejection is definitive and may
			// refer to another owner's existing sequence.
			if err == nil || !errors.As(err, &responseErr) {
				set.opened = append(set.opened, openedSequence{
					target: target, sequenceID: sequenceID, ownerID: ownerID,
				})
			}
			if err != nil {
				rollbackErr := set.Cleanup()
				if rollbackErr != nil {
					return set, errors.Join(err, fmt.Errorf("rollback opened sequences: %w", rollbackErr))
				}
				return nil, err
			}
		}
	}
	return set, nil
}

// CloseSequence closes one ID on every target where the set opened it.
func (set *SequenceSet) CloseSequence(ctx context.Context, sequenceID string) error {
	return set.closeUsing(
		func(opened openedSequence) bool { return opened.sequenceID == sequenceID },
		func(opened openedSequence) error { return closeSequence(ctx, opened) },
	)
}

// Close closes all sequences that remain open in reverse order.
func (set *SequenceSet) Close(ctx context.Context) error {
	return set.closeUsing(
		func(openedSequence) bool { return true },
		func(opened openedSequence) error { return closeSequence(ctx, opened) },
	)
}

// Cleanup makes a bounded close attempt for every tracked sequence and returns
// all cleanup failures. Failed closes remain tracked for a later retry.
func (set *SequenceSet) Cleanup() error {
	return set.closeUsing(
		func(openedSequence) bool { return true },
		func(opened openedSequence) error {
			ctx, cancel := context.WithTimeout(context.Background(), sequenceCleanupTimeout)
			defer cancel()
			return closeSequence(ctx, opened)
		},
	)
}

func (set *SequenceSet) closeUsing(
	shouldClose func(openedSequence) bool,
	closeOne func(openedSequence) error,
) error {
	if set == nil {
		return nil
	}
	var closeErr error
	retained := make([]openedSequence, 0, len(set.opened))
	for index := len(set.opened) - 1; index >= 0; index-- {
		opened := set.opened[index]
		if !shouldClose(opened) {
			retained = append(retained, opened)
			continue
		}
		if err := closeOne(opened); err != nil {
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
	err := sequenceCommand(
		ctx, opened.target, "closeSequence", opened.sequenceID, opened.ownerID,
	)
	var responseErr *WorkerResponseError
	if errors.As(err, &responseErr) {
		// A restarted worker has already released both the shard and every
		// sequence it owned, so cleanup is complete even though the close can
		// no longer find its original target.
		if strings.Contains(responseErr.Message, "is not open on shard") ||
			strings.Contains(responseErr.Message, "is not loaded") {
			return nil
		}
	}
	return err
}

func sequenceCommand(
	ctx context.Context,
	target SequenceTarget,
	command string,
	sequenceID string,
	ownerID string,
) error {
	_, err := target.Caller.Call(ctx, PersistentRequest{
		Command: command,
		Sequence: &PersistentSequenceRequest{
			ShardID: target.ShardID, SequenceID: sequenceID, OwnerID: ownerID,
		},
	})
	if err == nil || target.Name == "" {
		return err
	}
	return fmt.Errorf("%s %s: %w", target.Name, command, err)
}

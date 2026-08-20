package generation

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/tensorcheck"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

type TraceVerificationConfig struct {
	RTol           float64
	ATol           float64
	ForwardTimeout time.Duration
	Observer       TraceObserver
}

type TraceObserver func(TraceSample)

type TraceSample struct {
	Operation string
	Position  uint64
	Memory    workerproc.StageMemory
}

// VerifyTrace loads one independent full-model shard and compares its logits
// and greedy tokens with a previously captured distributed generation trace.
// This keeps the correctness oracle independent without requiring both full
// and sharded checkpoint copies to coexist in memory.
func VerifyTrace(
	ctx context.Context,
	caller workerproc.PersistentCaller,
	expectedModel workerproc.PersistentModelResult,
	config TraceVerificationConfig,
	prompt string,
	expectedPromptTokenIDs []int32,
	expectedGeneratedTokenIDs []int32,
	distributedLogits []workerproc.WireTensor,
) (verification Verification, returnErr error) {
	if caller == nil {
		return verification, errors.New("reference caller is required")
	}
	if prompt == "" || len(expectedPromptTokenIDs) == 0 {
		return verification, errors.New("reference prompt plan is empty")
	}
	if len(expectedGeneratedTokenIDs) == 0 || len(distributedLogits) != len(expectedGeneratedTokenIDs) {
		return verification, fmt.Errorf(
			"reference trace has %d logits for %d generated tokens",
			len(distributedLogits), len(expectedGeneratedTokenIDs),
		)
	}
	if config.RTol < 0 || config.ATol < 0 ||
		math.IsNaN(config.RTol) || math.IsNaN(config.ATol) ||
		math.IsInf(config.RTol, 0) || math.IsInf(config.ATol, 0) {
		return verification, errors.New("numeric tolerances must be finite and non-negative")
	}
	if config.ForwardTimeout == 0 {
		config.ForwardTimeout = DefaultForwardTimeout
	}
	if config.ForwardTimeout < time.Millisecond {
		return verification, errors.New("forward timeout must be at least 1ms")
	}

	actualModel, err := modelInfo(ctx, caller, expectedModel.ModelID)
	if err != nil {
		return verification, fmt.Errorf("reference model info: %w", err)
	}
	if err := matchModel(&expectedModel, actualModel, "reference"); err != nil {
		return verification, err
	}
	shardID := "generate-reference-" + modelHashSuffix(
		expectedModel.ModelID, expectedModel.CheckpointFingerprint,
	)
	if _, _, err := ensureShard(ctx, caller, workerproc.PersistentLoadShardRequest{
		ModelID: expectedModel.ModelID, ShardID: shardID,
		CheckpointFingerprint: expectedModel.CheckpointFingerprint,
		LayerStart:            0, LayerEnd: expectedModel.LayerCount,
		OwnsInput: true, OwnsOutput: true,
	}); err != nil {
		return verification, fmt.Errorf("reference shard: %w", err)
	}

	tokenized, err := tokenize(ctx, caller, expectedModel.ModelID, prompt)
	if err != nil {
		return verification, fmt.Errorf("tokenize reference prompt: %w", err)
	}
	if !slices.Equal(tokenized.TokenIDs, expectedPromptTokenIDs) {
		return verification, errors.New("reference prompt tokens do not match distributed trace")
	}
	sequenceID, err := randomSequenceID()
	if err != nil {
		return verification, err
	}
	sequences, err := workerproc.OpenSequences(ctx, []workerproc.SequenceTarget{{
		Name: "reference", Caller: caller, ShardID: shardID,
	}}, sequenceID)
	if sequences != nil {
		defer func() {
			if cleanupErr := sequences.Cleanup(); cleanupErr != nil {
				returnErr = errors.Join(returnErr, cleanupErr)
			}
		}()
	}
	if err != nil {
		return verification, fmt.Errorf("open reference sequence: %w", err)
	}

	position := uint64(len(expectedPromptTokenIDs))
	result, _, err := measuredInfer(
		ctx, config.ForwardTimeout, caller,
		"prefill", shardID, sequenceID, 0, "tokens", tokenTensor(expectedPromptTokenIDs),
		false,
	)
	if err != nil {
		return verification, fmt.Errorf("reference prefill: %w", err)
	}
	verification.GreedyTokenIDsMatch = true
	for index, expectedToken := range expectedGeneratedTokenIDs {
		operation := "decode"
		samplePosition := position - 1
		if index == 0 {
			operation = "prefill"
			samplePosition = 0
		}
		if config.Observer != nil {
			config.Observer(TraceSample{
				Operation: operation, Position: samplePosition, Memory: result.Memory,
			})
		}
		difference, compareErr := tensorcheck.CompareFinalLogits(
			distributedLogits[index], result.Output, config.RTol, config.ATol,
		)
		verification.MaxAbsoluteDifference = math.Max(
			verification.MaxAbsoluteDifference, difference.Absolute,
		)
		verification.MaxRelativeDifference = math.Max(
			verification.MaxRelativeDifference, difference.Relative,
		)
		if compareErr != nil {
			return verification, fmt.Errorf("reference step %d logits: %w", index, compareErr)
		}
		referenceToken, err := greedyToken(result.Output)
		if err != nil {
			return verification, fmt.Errorf("sample reference step %d: %w", index, err)
		}
		if referenceToken != expectedToken {
			verification.GreedyTokenIDsMatch = false
			return verification, fmt.Errorf(
				"reference step %d chose token %d; distributed trace chose %d",
				index, referenceToken, expectedToken,
			)
		}
		verification.ComparedTokens++
		if index == len(expectedGeneratedTokenIDs)-1 {
			break
		}
		result, _, err = measuredInfer(
			ctx, config.ForwardTimeout, caller,
			"decode", shardID, sequenceID, position, "tokens", tokenTensor([]int32{expectedToken}),
			false,
		)
		if err != nil {
			return verification, fmt.Errorf("reference decode step %d: %w", index+1, err)
		}
		position++
	}
	return verification, nil
}

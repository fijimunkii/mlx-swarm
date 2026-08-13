package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"slices"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/benchmark"
	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/smoke"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

const (
	defaultModelID = "mlx-community/gemma-3-270m-it-4bit"
	proofPrompt    = "Write a short story about two computers working together:"
)

type stageTeardown struct {
	Index                int    `json:"index"`
	TargetID             string `json:"targetID"`
	PostRunKVCacheBytes  int    `json:"postRunKVCacheBytes"`
	PostRunRetainedBytes int    `json:"postRunRetainedBytes"`
}

type smokeSummary struct {
	SchemaVersion                string                   `json:"schemaVersion"`
	Model                        string                   `json:"model"`
	ModelType                    string                   `json:"modelType"`
	CheckpointFingerprint        string                   `json:"checkpointFingerprint"`
	StageCount                   int                      `json:"stageCount"`
	VerificationPlan             generation.ExecutionPlan `json:"verificationPlan"`
	ServingPlan                  generation.ExecutionPlan `json:"servingPlan"`
	Prompt                       string                   `json:"prompt"`
	GeneratedTokenCount          int                      `json:"generatedTokenCount"`
	GeneratedTokenIDs            []int32                  `json:"generatedTokenIDs"`
	GeneratedText                string                   `json:"generatedText"`
	StopReason                   string                   `json:"stopReason"`
	AllFinalLogitsMatch          bool                     `json:"allFinalLogitsMatch"`
	GreedyTokenIDsMatch          bool                     `json:"greedyTokenIDsMatch"`
	RTol                         float64                  `json:"rtol"`
	ATol                         float64                  `json:"atol"`
	MaxAbsoluteDifference        float64                  `json:"maxAbsoluteDifference"`
	MaxRelativeDifference        float64                  `json:"maxRelativeDifference"`
	ServingTokenIDsMatchVerified bool                     `json:"servingTokenIDsMatchVerified"`
	NormalServingReturnsNoLogits bool                     `json:"normalServingReturnsNoLogits"`
	ComparedTokens               int                      `json:"comparedTokens"`
	SequenceTeardownValidated    bool                     `json:"sequenceTeardownValidated"`
	VerificationEvidence         benchmark.PlannedSummary `json:"verificationEvidence"`
	ServingEvidence              benchmark.PlannedSummary `json:"servingEvidence"`
	Teardown                     []stageTeardown          `json:"teardown"`
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	worker := flag.String("worker", workerproc.DefaultPath(), "path to the built MLXWorker executable")
	model := flag.String("model", defaultModelID, "checkpoint model ID")
	stageCount := flag.Int("stages", 5, "number of local retained execution stages")
	steps := flag.Int("steps", 32, "number of generated tokens required by the proof")
	rtol := flag.Float64("rtol", 1e-4, "relative reference-logit tolerance")
	atol := flag.Float64("atol", 1e-4, "absolute reference-logit tolerance")
	forwardTimeout := flag.Duration("forward-timeout", generation.DefaultForwardTimeout, "per-stage inference timeout")
	timeout := flag.Duration("timeout", 15*time.Minute, "overall proof timeout")
	flag.Parse()
	if *stageCount < 2 || *stageCount > 5 {
		return errors.New("-stages must be between 2 and 5")
	}
	if *steps < 32 {
		return errors.New("-steps must be at least 32")
	}

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	targets := make([]*workerproc.PersistentTarget, 0, *stageCount)
	callers := make([]workerproc.PersistentCaller, 0, *stageCount+1)
	executionTargets := make([]generation.ExecutionTarget, 0, *stageCount)
	for index := 0; index < *stageCount; index++ {
		target, err := workerproc.OpenPersistentTarget(*worker, "")
		if err != nil {
			cleanupTargets(targets)
			return fmt.Errorf("stage %d worker: %w", index, err)
		}
		targets = append(targets, target)
		callers = append(callers, target.Caller)
		executionTargets = append(executionTargets, generation.ExecutionTarget{
			TargetID: fmt.Sprintf("local-mlx-stage-%d", index), Caller: target.Caller,
		})
	}
	defer cleanupTargets(targets)
	reference, err := workerproc.OpenPersistentTarget(*worker, "")
	if err != nil {
		return fmt.Errorf("reference worker: %w", err)
	}
	defer reference.Cleanup()
	callers = append(callers, reference.Caller)

	var verificationSamples []generation.PlannedStageSample
	verificationSession, err := generation.NewBalancedPlannedSession(
		ctx, *model, executionTargets, reference.Caller,
		generation.StageResponseTensor,
		generation.PlannedSessionConfig{
			RTol: *rtol, ATol: *atol, ForwardTimeout: *forwardTimeout,
			Observer: func(sample generation.PlannedStageSample) {
				verificationSamples = append(verificationSamples, sample)
			},
		},
	)
	if err != nil {
		return err
	}
	verificationResult, err := verificationSession.Generate(ctx, generation.Request{
		Prompt: proofPrompt, MaxTokens: *steps,
		SequenceID: fmt.Sprintf("n-stage-verify-%d", *stageCount), IgnoreEOS: true,
	})
	if err != nil {
		return err
	}
	if len(verificationResult.GeneratedTokenIDs) != *steps || verificationResult.StopReason != "max_tokens" {
		return fmt.Errorf(
			"generated %d tokens and stopped on %s; want %d maximum-length tokens",
			len(verificationResult.GeneratedTokenIDs), verificationResult.StopReason, *steps,
		)
	}
	if verificationResult.Verification == nil ||
		!verificationResult.Verification.GreedyTokenIDsMatch ||
		verificationResult.Verification.ComparedTokens != *steps {
		return errors.New("N-stage logits or greedy tokens did not match the full-checkpoint reference")
	}
	verificationEvidence, err := benchmark.SummarizePlanned(verificationSamples)
	if err != nil {
		return err
	}

	var servingSamples []generation.PlannedStageSample
	servingSession, err := generation.NewBalancedPlannedSession(
		ctx, *model, executionTargets, nil,
		generation.StageResponseSampledToken,
		generation.PlannedSessionConfig{
			ForwardTimeout: *forwardTimeout,
			Observer: func(sample generation.PlannedStageSample) {
				servingSamples = append(servingSamples, sample)
			},
		},
	)
	if err != nil {
		return err
	}
	servingResult, err := servingSession.Generate(ctx, generation.Request{
		Prompt: proofPrompt, MaxTokens: *steps,
		SequenceID: fmt.Sprintf("n-stage-serve-%d", *stageCount), IgnoreEOS: true,
	})
	if err != nil {
		return err
	}
	servingMatches := slices.Equal(
		servingResult.GeneratedTokenIDs, verificationResult.GeneratedTokenIDs,
	)
	if !servingMatches {
		return errors.New("terminal-sampling token sequence differs from verified logits path")
	}
	servingEvidence, err := benchmark.SummarizePlanned(servingSamples)
	if err != nil {
		return err
	}
	noServingLogits := servingReturnsNoTerminalLogits(servingSamples)
	if !noServingLogits {
		return errors.New("normal serving returned a terminal full-vocabulary tensor")
	}
	if err := smoke.RequireNoSequenceState(ctx, callers...); err != nil {
		return err
	}

	teardown := make([]stageTeardown, *stageCount)
	for index, target := range executionTargets {
		state, err := smoke.State(ctx, target.Caller)
		if err != nil {
			return fmt.Errorf("stage %d final state: %w", index, err)
		}
		teardown[index] = stageTeardown{
			Index: index, TargetID: target.TargetID,
			PostRunKVCacheBytes:  state.KVCacheBytes,
			PostRunRetainedBytes: state.RetainedBytes,
		}
	}

	summary := smokeSummary{
		SchemaVersion: "1", Model: verificationResult.Model,
		ModelType:                    verificationResult.ModelType,
		CheckpointFingerprint:        verificationResult.CheckpointFingerprint,
		StageCount:                   len(verificationResult.ExecutionPlan.Stages),
		VerificationPlan:             verificationResult.ExecutionPlan,
		ServingPlan:                  servingResult.ExecutionPlan,
		Prompt:                       verificationResult.Prompt,
		GeneratedTokenCount:          len(verificationResult.GeneratedTokenIDs),
		GeneratedTokenIDs:            verificationResult.GeneratedTokenIDs,
		GeneratedText:                verificationResult.Text,
		StopReason:                   verificationResult.StopReason,
		AllFinalLogitsMatch:          verificationResult.Verification.ComparedTokens == *steps,
		GreedyTokenIDsMatch:          verificationResult.Verification.GreedyTokenIDsMatch,
		RTol:                         verificationResult.RTol,
		ATol:                         verificationResult.ATol,
		MaxAbsoluteDifference:        verificationResult.Verification.MaxAbsoluteDifference,
		MaxRelativeDifference:        verificationResult.Verification.MaxRelativeDifference,
		ServingTokenIDsMatchVerified: servingMatches,
		NormalServingReturnsNoLogits: noServingLogits,
		ComparedTokens:               verificationResult.Verification.ComparedTokens,
		SequenceTeardownValidated:    true,
		VerificationEvidence:         verificationEvidence, ServingEvidence: servingEvidence,
		Teardown: teardown,
	}
	encoded, err := json.Marshal(summary)
	if err != nil {
		return err
	}
	fmt.Println(string(encoded))
	return nil
}

func servingReturnsNoTerminalLogits(samples []generation.PlannedStageSample) bool {
	if len(samples) == 0 {
		return false
	}
	for _, sample := range samples {
		if len(sample.Stages) == 0 {
			return false
		}
		for index, stage := range sample.Stages {
			wantTerminal := index == len(sample.Stages)-1
			if stage.TerminalSampling != wantTerminal {
				return false
			}
			if wantTerminal && stage.ResponseTensorBytes != 0 {
				return false
			}
		}
	}
	return true
}

func cleanupTargets(targets []*workerproc.PersistentTarget) {
	for index := len(targets) - 1; index >= 0; index-- {
		targets[index].Cleanup()
	}
}

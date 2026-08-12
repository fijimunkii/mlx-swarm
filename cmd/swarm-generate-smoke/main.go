package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/smoke"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

const (
	defaultModelID = "mlx-community/gemma-3-270m-it-4bit"
	proofPrompt    = "Write a short story about two computers working together:"
)

type smokeSummary struct {
	Model                     string `json:"model"`
	ModelType                 string `json:"modelType"`
	Distributed               bool   `json:"distributed"`
	Prompt                    string `json:"prompt"`
	GeneratedTokenCount       int    `json:"generatedTokenCount"`
	GeneratedText             string `json:"generatedText"`
	StopReason                string `json:"stopReason"`
	GreedyTokenIDsMatch       bool   `json:"greedyTokenIDsMatch"`
	RepeatedRequestValidated  bool   `json:"repeatedRequestValidated"`
	SequenceTeardownValidated bool   `json:"sequenceTeardownValidated"`
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	worker := flag.String("worker", workerproc.DefaultPath(), "path to the built MLXWorker executable")
	peer := flag.String("peer", "", "optional consumer swarmd base URL")
	model := flag.String("model", defaultModelID, "checkpoint model ID")
	steps := flag.Int("steps", 32, "number of generated tokens required by the proof")
	rtol := flag.Float64("rtol", 1e-4, "relative reference-logit tolerance")
	atol := flag.Float64("atol", 1e-4, "absolute reference-logit tolerance")
	timeout := flag.Duration("timeout", 10*time.Minute, "overall proof timeout")
	flag.Parse()
	if *steps < 32 {
		return errors.New("-steps must be at least 32")
	}

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	producer, err := workerproc.OpenPersistentTarget(*worker, "")
	if err != nil {
		return fmt.Errorf("producer: %w", err)
	}
	defer producer.Cleanup()
	consumer, err := workerproc.OpenPersistentTarget(*worker, *peer)
	if err != nil {
		return fmt.Errorf("consumer: %w", err)
	}
	defer consumer.Cleanup()
	reference, err := workerproc.OpenPersistentTarget(*worker, "")
	if err != nil {
		return fmt.Errorf("reference: %w", err)
	}
	defer reference.Cleanup()

	session, err := generation.NewSession(
		ctx,
		producer.Caller,
		consumer.Caller,
		reference.Caller,
		generation.SessionConfig{Model: *model, RTol: *rtol, ATol: *atol},
	)
	if err != nil {
		return err
	}
	loadedBefore, err := loadCounts(ctx, producer.Caller, consumer.Caller, reference.Caller)
	if err != nil {
		return err
	}

	proof, err := session.Generate(ctx, generation.Request{
		Prompt: proofPrompt, MaxTokens: *steps, SequenceID: "generation-proof",
	})
	if err != nil {
		return err
	}
	if len(proof.GeneratedTokenIDs) != *steps || proof.StopReason != "max_tokens" {
		return fmt.Errorf(
			"proof generated %d tokens and stopped on %s; want %d maximum-length tokens",
			len(proof.GeneratedTokenIDs), proof.StopReason, *steps,
		)
	}
	if proof.Verification == nil || !proof.Verification.GreedyTokenIDsMatch ||
		proof.Verification.ComparedTokens != *steps {
		return errors.New("distributed greedy tokens did not match the cached reference")
	}
	if err := smoke.RequireNoSequenceState(ctx, producer.Caller, consumer.Caller, reference.Caller); err != nil {
		return err
	}

	repeatedSession, err := generation.NewSession(
		ctx,
		producer.Caller,
		consumer.Caller,
		reference.Caller,
		generation.SessionConfig{Model: *model, RTol: *rtol, ATol: *atol},
	)
	if err != nil {
		return fmt.Errorf("prepare repeated generation session: %w", err)
	}
	repeated, err := repeatedSession.Generate(ctx, generation.Request{
		Prompt: "Continue this phrase: distributed systems", MaxTokens: 1,
		SequenceID: "generation-repeat-proof",
	})
	if err != nil {
		return fmt.Errorf("repeated generation request: %w", err)
	}
	if len(repeated.GeneratedTokenIDs) != 1 || repeated.Verification == nil ||
		!repeated.Verification.GreedyTokenIDsMatch {
		return errors.New("repeated generation request did not produce one verified token")
	}
	loadedAfter, err := loadCounts(ctx, producer.Caller, consumer.Caller, reference.Caller)
	if err != nil {
		return err
	}
	if loadedAfter != loadedBefore {
		return fmt.Errorf("repeated request reloaded a shard: before=%v after=%v", loadedBefore, loadedAfter)
	}
	if err := smoke.RequireNoSequenceState(ctx, producer.Caller, consumer.Caller, reference.Caller); err != nil {
		return err
	}

	summary := smokeSummary{
		Model: *model, ModelType: proof.ModelType, Distributed: *peer != "",
		Prompt: proofPrompt, GeneratedTokenCount: len(proof.GeneratedTokenIDs),
		GeneratedText: proof.Text, StopReason: proof.StopReason,
		GreedyTokenIDsMatch:      proof.Verification.GreedyTokenIDsMatch,
		RepeatedRequestValidated: true, SequenceTeardownValidated: true,
	}
	encoded, err := json.Marshal(summary)
	if err != nil {
		return err
	}
	fmt.Println(string(encoded))
	return nil
}

func loadCounts(
	ctx context.Context,
	callers ...workerproc.PersistentCaller,
) ([3]int, error) {
	var counts [3]int
	for index, caller := range callers {
		state, err := smoke.State(ctx, caller)
		if err != nil {
			return counts, err
		}
		counts[index] = state.LoadCount
	}
	return counts, nil
}

package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/benchmark"
	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/smoke"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

const (
	defaultModelID = "mlx-community/gemma-3-270m-it-4bit"
	defaultPrompt  = "Write a short story about two computers working together:"
)

type configuration struct {
	Model                   string               `json:"model"`
	ModelType               string               `json:"modelType"`
	CheckpointFingerprint   string               `json:"checkpointFingerprint"`
	Prompt                  string               `json:"prompt"`
	PromptTokenIDs          []int32              `json:"promptTokenIDs"`
	PromptTokenCount        int                  `json:"promptTokenCount"`
	GeneratedTokenIDs       []int32              `json:"generatedTokenIDs"`
	GeneratedTokenCount     int                  `json:"generatedTokenCount"`
	ShardPlan               generation.ShardPlan `json:"shardPlan"`
	Hardware                string               `json:"hardware"`
	Hostname                string               `json:"hostname"`
	Route                   string               `json:"route"`
	Peer                    string               `json:"peer,omitempty"`
	Distributed             bool                 `json:"distributed"`
	WarmupDecodeSteps       int                  `json:"warmupDecodeSteps"`
	PrefillSamplesRequested int                  `json:"prefillSamplesRequested"`
	DecodeSamplesRequested  int                  `json:"decodeSamplesRequested"`
	SamplingPolicy          string               `json:"samplingPolicy"`
	RTol                    float64              `json:"rtol"`
	ATol                    float64              `json:"atol"`
	SessionSetupMicros      int64                `json:"sessionSetupMicrosExcluded"`
}

type warmupSummary struct {
	GeneratedTokenCount int   `json:"generatedTokenCount"`
	DecodeStepCount     int   `json:"decodeStepCount"`
	TotalMicros         int64 `json:"totalMicrosExcluded"`
}

type verificationSummary struct {
	AllFinalLogitsMatch   bool    `json:"allFinalLogitsMatch"`
	GreedyTokenIDsMatch   bool    `json:"greedyTokenIDsMatch"`
	ComparedTokens        int     `json:"comparedTokens"`
	MaxAbsoluteDifference float64 `json:"maxAbsoluteDifference"`
	MaxRelativeDifference float64 `json:"maxRelativeDifference"`
}

type workerMemory struct {
	ShardID                string                 `json:"shardID"`
	WeightKeyCount         int                    `json:"weightKeyCount"`
	LoadedShardMemory      workerproc.StageMemory `json:"loadedShardMemory"`
	RetainedByteBudget     int                    `json:"retainedByteBudget"`
	MaxKVCacheBytes        int                    `json:"maxKVCacheBytes"`
	MaxActiveMemoryBytes   int                    `json:"maxActiveMemoryBytes"`
	MaxAllocatorCacheBytes int                    `json:"maxAllocatorCacheBytes"`
	PeakMemoryBytes        int                    `json:"peakMemoryBytes"`
	PostRunKVCacheBytes    int                    `json:"postRunKVCacheBytes"`
	PostRunRetainedBytes   int                    `json:"postRunRetainedBytes"`
}

type memorySummary struct {
	Producer  workerMemory `json:"producer"`
	Consumer  workerMemory `json:"consumer"`
	Reference workerMemory `json:"reference"`
}

type outputSummary struct {
	SchemaVersion string                   `json:"schemaVersion"`
	RecordedAt    time.Time                `json:"recordedAt"`
	Configuration configuration            `json:"configuration"`
	Warmup        warmupSummary            `json:"warmup"`
	Prefill       benchmark.StageSummary   `json:"prefill"`
	Decode        benchmark.StageSummary   `json:"decode"`
	Memory        memorySummary            `json:"memory"`
	Verification  verificationSummary      `json:"verification"`
	Samples       []generation.StageSample `json:"samples"`
}

type sampleCollector struct {
	enabled bool
	samples []generation.StageSample
}

func (collector *sampleCollector) observe(sample generation.StageSample) {
	if collector.enabled {
		collector.samples = append(collector.samples, sample)
	}
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
	prompt := flag.String("prompt", defaultPrompt, "benchmark prompt")
	warmupDecodes := flag.Int("warmup-decodes", 8, "unmeasured cached decode steps after shard loading")
	prefillSamples := flag.Int("prefill-samples", 5, "number of measured fresh-sequence prefills")
	decodeSamples := flag.Int("decode-samples", 100, "number of measured cached decode steps")
	hardware := flag.String("hardware", runtime.GOOS+"/"+runtime.GOARCH, "hardware description recorded in output")
	route := flag.String("route", "", "network route description recorded in output")
	rtol := flag.Float64("rtol", 1e-4, "relative reference-logit tolerance")
	atol := flag.Float64("atol", 1e-4, "absolute reference-logit tolerance")
	timeout := flag.Duration("timeout", 15*time.Minute, "overall benchmark timeout")
	flag.Parse()

	if *prompt == "" {
		return errors.New("-prompt is required")
	}
	if *warmupDecodes < 0 {
		return errors.New("-warmup-decodes must be non-negative")
	}
	if *prefillSamples <= 0 {
		return errors.New("-prefill-samples must be positive")
	}
	if *decodeSamples <= 0 {
		return errors.New("-decode-samples must be positive")
	}
	if *timeout <= 0 {
		return errors.New("-timeout must be positive")
	}
	if *route == "" {
		if *peer == "" {
			*route = "local"
		} else {
			*route = "unspecified"
		}
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

	collector := &sampleCollector{}
	session, err := generation.NewSession(
		ctx, producer.Caller, consumer.Caller, reference.Caller,
		generation.SessionConfig{
			Model: *model, RTol: *rtol, ATol: *atol, Observer: collector.observe,
		},
	)
	if err != nil {
		return err
	}
	info := session.Info()
	loadedMemory, err := captureLoadedMemory(
		ctx,
		[]workerTarget{
			{name: "producer", caller: producer.Caller, shardID: info.ShardPlan.Producer.ID},
			{name: "consumer", caller: consumer.Caller, shardID: info.ShardPlan.Consumer.ID},
			{name: "reference", caller: reference.Caller, shardID: info.ReferenceShardID},
		},
	)
	if err != nil {
		return err
	}

	warmup, err := session.Generate(ctx, generation.Request{
		Prompt: *prompt, MaxTokens: *warmupDecodes + 1,
		SequenceID: "benchmark-warmup", IgnoreEOS: true,
	})
	if err != nil {
		return fmt.Errorf("warmup: %w", err)
	}
	if warmup.StopReason != "max_tokens" || len(warmup.GeneratedTokenIDs) != *warmupDecodes+1 {
		return fmt.Errorf("warmup produced %d tokens and stopped on %s", len(warmup.GeneratedTokenIDs), warmup.StopReason)
	}

	collector.enabled = true
	verification := verificationSummary{AllFinalLogitsMatch: true, GreedyTokenIDsMatch: true}
	var measured generation.Result
	for index := 0; index < *prefillSamples; index++ {
		maxTokens := 1
		if index == *prefillSamples-1 {
			maxTokens = *decodeSamples + 1
		}
		measured, err = session.Generate(ctx, generation.Request{
			Prompt: *prompt, MaxTokens: maxTokens,
			SequenceID: fmt.Sprintf("benchmark-measured-%d", index), IgnoreEOS: true,
		})
		if err != nil {
			return fmt.Errorf("measurement %d: %w", index+1, err)
		}
		if measured.StopReason != "max_tokens" || len(measured.GeneratedTokenIDs) != maxTokens {
			return fmt.Errorf("measurement %d produced %d tokens and stopped on %s", index+1, len(measured.GeneratedTokenIDs), measured.StopReason)
		}
		if measured.Verification == nil || !measured.Verification.GreedyTokenIDsMatch {
			return fmt.Errorf("measurement %d did not match the cached reference", index+1)
		}
		mergeVerification(&verification, measured.Verification)
	}
	collector.enabled = false

	prefill, decode := splitSamples(collector.samples)
	if len(prefill) != *prefillSamples || len(decode) != *decodeSamples {
		return fmt.Errorf(
			"recorded prefill/decode samples %d/%d, want %d/%d",
			len(prefill), len(decode), *prefillSamples, *decodeSamples,
		)
	}
	if err := smoke.RequireNoSequenceState(ctx, producer.Caller, consumer.Caller, reference.Caller); err != nil {
		return err
	}
	finalStates, err := captureStates(ctx, producer.Caller, consumer.Caller, reference.Caller)
	if err != nil {
		return err
	}
	hostname, _ := os.Hostname()
	summary := outputSummary{
		SchemaVersion: "1", RecordedAt: time.Now().UTC(),
		Configuration: configuration{
			Model: info.Model.ModelID, ModelType: info.Model.ModelType,
			CheckpointFingerprint: info.Model.CheckpointFingerprint,
			Prompt:                *prompt, PromptTokenIDs: measured.PromptTokenIDs,
			PromptTokenCount:    len(measured.PromptTokenIDs),
			GeneratedTokenIDs:   measured.GeneratedTokenIDs,
			GeneratedTokenCount: len(measured.GeneratedTokenIDs),
			ShardPlan:           info.ShardPlan, Hardware: *hardware, Hostname: hostname,
			Route: *route, Peer: *peer, Distributed: *peer != "",
			WarmupDecodeSteps:       *warmupDecodes,
			PrefillSamplesRequested: *prefillSamples, DecodeSamplesRequested: *decodeSamples,
			SamplingPolicy: "greedy lowest-token-ID tie-break; EOS ignored for fixed sample count",
			RTol:           *rtol, ATol: *atol, SessionSetupMicros: info.SessionSetupMicros,
		},
		Warmup: warmupSummary{
			GeneratedTokenCount: len(warmup.GeneratedTokenIDs),
			DecodeStepCount:     max(0, len(warmup.GeneratedTokenIDs)-1),
			TotalMicros:         warmup.Timing.TotalMicros,
		},
		Prefill: benchmark.Summarize(prefill), Decode: benchmark.Summarize(decode),
		Memory: memorySummary{
			Producer:  memoryForRole(loadedMemory[0], finalStates[0], collector.samples, "producer"),
			Consumer:  memoryForRole(loadedMemory[1], finalStates[1], collector.samples, "consumer"),
			Reference: memoryForRole(loadedMemory[2], finalStates[2], collector.samples, "reference"),
		},
		Verification: verification, Samples: collector.samples,
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(summary)
}

type workerTarget struct {
	name    string
	caller  workerproc.PersistentCaller
	shardID string
}

type loadedWorkerMemory struct {
	shard  workerproc.PersistentShardSnapshot
	budget int
}

func captureLoadedMemory(ctx context.Context, targets []workerTarget) ([3]loadedWorkerMemory, error) {
	var captured [3]loadedWorkerMemory
	for index, target := range targets {
		state, err := smoke.State(ctx, target.caller)
		if err != nil {
			return captured, fmt.Errorf("%s state: %w", target.name, err)
		}
		found := false
		for _, shard := range state.LoadedShards {
			if shard.ShardID == target.shardID {
				captured[index] = loadedWorkerMemory{shard: shard, budget: state.RetainedByteBudget}
				found = true
				break
			}
		}
		if !found {
			return captured, fmt.Errorf("%s state does not contain shard %s", target.name, target.shardID)
		}
	}
	return captured, nil
}

func captureStates(
	ctx context.Context,
	callers ...workerproc.PersistentCaller,
) ([3]*workerproc.PersistentWorkerState, error) {
	var states [3]*workerproc.PersistentWorkerState
	for index, caller := range callers {
		state, err := smoke.State(ctx, caller)
		if err != nil {
			return states, err
		}
		states[index] = state
	}
	return states, nil
}

func splitSamples(samples []generation.StageSample) ([]generation.StageSample, []generation.StageSample) {
	var prefill, decode []generation.StageSample
	for _, sample := range samples {
		switch sample.Operation {
		case "prefill":
			prefill = append(prefill, sample)
		case "decode":
			decode = append(decode, sample)
		}
	}
	return prefill, decode
}

func mergeVerification(summary *verificationSummary, verification *generation.Verification) {
	summary.GreedyTokenIDsMatch = summary.GreedyTokenIDsMatch && verification.GreedyTokenIDsMatch
	summary.ComparedTokens += verification.ComparedTokens
	summary.MaxAbsoluteDifference = math.Max(summary.MaxAbsoluteDifference, verification.MaxAbsoluteDifference)
	summary.MaxRelativeDifference = math.Max(summary.MaxRelativeDifference, verification.MaxRelativeDifference)
}

func memoryForRole(
	loaded loadedWorkerMemory,
	final *workerproc.PersistentWorkerState,
	samples []generation.StageSample,
	role string,
) workerMemory {
	memory := workerMemory{
		ShardID: loaded.shard.ShardID, WeightKeyCount: loaded.shard.WeightKeyCount,
		LoadedShardMemory: loaded.shard.LoadedMemory, RetainedByteBudget: loaded.budget,
		PostRunKVCacheBytes: final.KVCacheBytes, PostRunRetainedBytes: final.RetainedBytes,
	}
	for _, sample := range samples {
		var kvBytes int
		var stage workerproc.StageMemory
		switch role {
		case "producer":
			kvBytes, stage = sample.ProducerKVCacheBytes, sample.ProducerMemory
		case "consumer":
			kvBytes, stage = sample.ConsumerKVCacheBytes, sample.ConsumerMemory
		case "reference":
			kvBytes, stage = sample.ReferenceKVCacheBytes, sample.ReferenceMemory
		}
		memory.MaxKVCacheBytes = max(memory.MaxKVCacheBytes, kvBytes)
		memory.MaxActiveMemoryBytes = max(memory.MaxActiveMemoryBytes, stage.ActiveBytes)
		memory.MaxAllocatorCacheBytes = max(memory.MaxAllocatorCacheBytes, stage.CacheBytes)
		memory.PeakMemoryBytes = max(memory.PeakMemoryBytes, stage.PeakBytes)
	}
	return memory
}

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"slices"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/smoke"
	"github.com/fijimunkii/mlx-swarm/internal/tensorcheck"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

const (
	defaultModelID = "mlx-community/gemma-3-270m-it-4bit"
	producerShard  = "cache-producer"
	consumerShard  = "cache-consumer"
	referenceShard = "cache-reference"
	sequenceA      = "cache-sequence-a"
	sequenceB      = "cache-sequence-b"
)

type sequencePlan struct {
	id       string
	prompt   []int32
	next     uint64
	tokenAdd int32
}

type smokeSummary struct {
	Model                       string  `json:"model"`
	ModelType                   string  `json:"modelType"`
	Layers                      int     `json:"layers"`
	SplitLayer                  int     `json:"splitLayer"`
	DecodeStepsPerSequence      int     `json:"decodeStepsPerSequence"`
	SequenceCount               int     `json:"sequenceCount"`
	Distributed                 bool    `json:"distributed"`
	RTol                        float64 `json:"rtol"`
	ATol                        float64 `json:"atol"`
	LogitComparisons            int     `json:"logitComparisons"`
	MaxAbsoluteDifference       float64 `json:"maxAbsoluteDifference"`
	MaxRelativeDifference       float64 `json:"maxRelativeDifference"`
	AllFinalLogitsMatch         bool    `json:"allFinalLogitsMatch"`
	PositionsValidated          bool    `json:"positionsValidated"`
	MutationReplayValidated     bool    `json:"mutationReplayValidated"`
	ResourceLimitsValidated     bool    `json:"resourceLimitsValidated"`
	SequenceIsolationValidated  bool    `json:"sequenceIsolationValidated"`
	ProducerKVCacheBytes        int     `json:"producerKVCacheBytes"`
	ConsumerKVCacheBytes        int     `json:"consumerKVCacheBytes"`
	ReferenceKVCacheBytes       int     `json:"referenceKVCacheBytes"`
	ProducerAfterTeardownBytes  int     `json:"producerAfterTeardownBytes"`
	ConsumerAfterTeardownBytes  int     `json:"consumerAfterTeardownBytes"`
	ReferenceAfterTeardownBytes int     `json:"referenceAfterTeardownBytes"`
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	worker := flag.String("worker", workerproc.DefaultPath(), "path to the built MLXWorker executable")
	peer := flag.String("peer", "", "optional swarmd base URL for the consumer shard")
	model := flag.String("model", defaultModelID, "real checkpoint model ID")
	layers := flag.Int("layers", 18, "checkpoint transformer layer count")
	split := flag.Int("split", 9, "first layer owned by the consumer")
	steps := flag.Int("steps", 32, "cached decode steps per interleaved sequence")
	rtol := flag.Float64("rtol", 1e-4, "relative final-logit tolerance")
	atol := flag.Float64("atol", 1e-4, "absolute final-logit tolerance")
	timeout := flag.Duration("timeout", 10*time.Minute, "overall smoke timeout")
	flag.Parse()

	if *layers < 2 || *split <= 0 || *split >= *layers {
		return fmt.Errorf("split %d must be internal to %d layers", *split, *layers)
	}
	if *steps < 32 {
		return errors.New("steps must be at least 32")
	}
	if *rtol < 0 || *atol < 0 {
		return errors.New("numeric tolerances must be non-negative")
	}

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	producer, err := smoke.OpenWorker(*worker, "")
	if err != nil {
		return fmt.Errorf("start producer worker: %w", err)
	}
	defer producer.Cleanup()

	reference, err := smoke.OpenWorker(*worker, "")
	if err != nil {
		return fmt.Errorf("start reference worker: %w", err)
	}
	defer reference.Cleanup()

	consumer, err := smoke.OpenWorker(*worker, *peer)
	if err != nil {
		return fmt.Errorf("open consumer worker: %w", err)
	}
	defer consumer.Cleanup()

	for name, client := range map[string]*smoke.Worker{
		"producer":  producer,
		"consumer":  consumer,
		"reference": reference,
	} {
		if _, err := smoke.Call(ctx, client.Caller, workerproc.PersistentRequest{Command: "health"}); err != nil {
			return fmt.Errorf("%s %w", name, err)
		}
	}

	producerSnapshot, err := smoke.LoadShard(ctx, producer.Caller, workerproc.PersistentLoadShardRequest{
		ModelID: *model, ShardID: producerShard,
		LayerStart: 0, LayerEnd: *split, OwnsInput: true,
	})
	if err != nil {
		return fmt.Errorf("load producer: %w", err)
	}
	consumerSnapshot, err := smoke.LoadShard(ctx, consumer.Caller, workerproc.PersistentLoadShardRequest{
		ModelID: *model, ShardID: consumerShard,
		LayerStart: *split, LayerEnd: *layers, OwnsOutput: true,
	})
	if err != nil {
		return fmt.Errorf("load consumer: %w", err)
	}
	referenceSnapshot, err := smoke.LoadShard(ctx, reference.Caller, workerproc.PersistentLoadShardRequest{
		ModelID: *model, ShardID: referenceShard,
		LayerStart: 0, LayerEnd: *layers, OwnsInput: true, OwnsOutput: true,
	})
	if err != nil {
		return fmt.Errorf("load reference: %w", err)
	}
	if producerSnapshot.ModelType != consumerSnapshot.ModelType ||
		producerSnapshot.ModelType != referenceSnapshot.ModelType {
		return fmt.Errorf(
			"adapter mismatch: producer=%s consumer=%s reference=%s",
			producerSnapshot.ModelType,
			consumerSnapshot.ModelType,
			referenceSnapshot.ModelType,
		)
	}

	plans := []*sequencePlan{
		{id: sequenceA, prompt: []int32{1, 2, 3, 4, 5, 6}, tokenAdd: 32},
		{id: sequenceB, prompt: []int32{11, 12, 13, 14}, tokenAdd: 96},
	}
	targets := []smoke.SequenceTarget{
		{Name: "producer", Caller: producer.Caller, ShardID: producerShard},
		{Name: "consumer", Caller: consumer.Caller, ShardID: consumerShard},
		{Name: "reference", Caller: reference.Caller, ShardID: referenceShard},
	}
	sequences, err := smoke.OpenSequences(ctx, targets, sequenceA, sequenceB)
	if sequences != nil {
		defer sequences.Cleanup()
	}
	if err != nil {
		return err
	}

	var logitMetrics tensorcheck.Metrics
	var lastBoundary workerproc.WireTensor
	mutationReplayValidated := true
	for _, plan := range plans {
		prompt := smoke.TokenTensor(plan.prompt)
		producerResult, err := smoke.Infer(
			ctx, producer.Caller, "prefill", producerShard, plan.id, 0, "tokens", prompt,
		)
		if err != nil {
			return err
		}
		if err := proveMutationReplay(
			ctx, producer.Caller, "prefill", producerShard, plan.id, 0, "tokens", prompt,
			producerResult,
		); err != nil {
			return fmt.Errorf("producer prefill replay for %s: %w", plan.id, err)
		}
		lastBoundary = producerResult.Output
		consumerResult, err := smoke.Infer(
			ctx, consumer.Caller, "prefill", consumerShard, plan.id, 0, "hidden",
			producerResult.Output,
		)
		if err != nil {
			return err
		}
		if err := proveMutationReplay(
			ctx, consumer.Caller, "prefill", consumerShard, plan.id, 0, "hidden",
			producerResult.Output, consumerResult,
		); err != nil {
			return fmt.Errorf("consumer prefill replay for %s: %w", plan.id, err)
		}
		referenceResult, err := smoke.Infer(
			ctx, reference.Caller, "prefill", referenceShard, plan.id, 0, "tokens", prompt,
		)
		if err != nil {
			return err
		}
		if err := proveMutationReplay(
			ctx, reference.Caller, "prefill", referenceShard, plan.id, 0, "tokens", prompt,
			referenceResult,
		); err != nil {
			return fmt.Errorf("reference prefill replay for %s: %w", plan.id, err)
		}
		if err := logitMetrics.Compare(
			consumerResult.Output,
			referenceResult.Output,
			*rtol,
			*atol,
		); err != nil {
			return fmt.Errorf("prefill logits for %s: %w", plan.id, err)
		}
		plan.next = uint64(len(plan.prompt))
		if producerResult.NextPosition != plan.next ||
			consumerResult.NextPosition != plan.next ||
			referenceResult.NextPosition != plan.next {
			return fmt.Errorf("prefill next position mismatch for %s", plan.id)
		}
	}

	positionsValidated, err := provePositionValidation(ctx, producer.Caller, consumer.Caller, plans[0])
	if err != nil {
		return err
	}
	resourceLimitsValidated, err := proveResourceLimits(
		ctx,
		producer.Caller,
		consumer.Caller,
		plans[0],
	)
	if err != nil {
		return err
	}

	producerAfterPrefill, err := smoke.State(ctx, producer.Caller)
	if err != nil {
		return err
	}
	consumerAfterPrefill, err := smoke.State(ctx, consumer.Caller)
	if err != nil {
		return err
	}
	referenceAfterPrefill, err := smoke.State(ctx, reference.Caller)
	if err != nil {
		return err
	}
	if producerAfterPrefill.KVCacheBytes == 0 || consumerAfterPrefill.KVCacheBytes == 0 ||
		referenceAfterPrefill.KVCacheBytes == 0 {
		return fmt.Errorf(
			"prefill did not allocate all caches: producer=%d consumer=%d reference=%d",
			producerAfterPrefill.KVCacheBytes,
			consumerAfterPrefill.KVCacheBytes,
			referenceAfterPrefill.KVCacheBytes,
		)
	}

	for step := 0; step < *steps; step++ {
		for _, plan := range plans {
			token := smoke.TokenTensor([]int32{plan.tokenAdd + int32(step)})
			producerResult, err := smoke.Infer(
				ctx, producer.Caller, "decode", producerShard, plan.id, plan.next, "tokens", token,
			)
			if err != nil {
				return fmt.Errorf("producer decode %s step %d: %w", plan.id, step, err)
			}
			if step == 0 {
				if err := proveMutationReplay(
					ctx, producer.Caller, "decode", producerShard, plan.id, plan.next, "tokens",
					token, producerResult,
				); err != nil {
					return fmt.Errorf("producer decode replay for %s: %w", plan.id, err)
				}
			}
			lastBoundary = producerResult.Output
			consumerResult, err := smoke.Infer(
				ctx, consumer.Caller, "decode", consumerShard, plan.id, plan.next, "hidden",
				producerResult.Output,
			)
			if err != nil {
				return fmt.Errorf("consumer decode %s step %d: %w", plan.id, step, err)
			}
			if step == 0 {
				if err := proveMutationReplay(
					ctx, consumer.Caller, "decode", consumerShard, plan.id, plan.next, "hidden",
					producerResult.Output, consumerResult,
				); err != nil {
					return fmt.Errorf("consumer decode replay for %s: %w", plan.id, err)
				}
			}
			referenceResult, err := smoke.Infer(
				ctx, reference.Caller, "decode", referenceShard, plan.id, plan.next, "tokens", token,
			)
			if err != nil {
				return fmt.Errorf("reference decode %s step %d: %w", plan.id, step, err)
			}
			if step == 0 {
				if err := proveMutationReplay(
					ctx, reference.Caller, "decode", referenceShard, plan.id, plan.next, "tokens",
					token, referenceResult,
				); err != nil {
					return fmt.Errorf("reference decode replay for %s: %w", plan.id, err)
				}
			}
			if err := logitMetrics.Compare(
				consumerResult.Output,
				referenceResult.Output,
				*rtol,
				*atol,
			); err != nil {
				return fmt.Errorf("decode logits for %s step %d: %w", plan.id, step, err)
			}
			plan.next++
			if producerResult.NextPosition != plan.next ||
				consumerResult.NextPosition != plan.next ||
				referenceResult.NextPosition != plan.next {
				return fmt.Errorf("decode next position mismatch for %s step %d", plan.id, step)
			}
		}
	}
	expectedComparisons := len(plans) * (*steps + 1)
	if logitMetrics.Comparisons != expectedComparisons {
		return fmt.Errorf(
			"compared %d final-logit pairs, want %d",
			logitMetrics.Comparisons,
			expectedComparisons,
		)
	}

	producerAfterDecode, err := smoke.State(ctx, producer.Caller)
	if err != nil {
		return err
	}
	consumerAfterDecode, err := smoke.State(ctx, consumer.Caller)
	if err != nil {
		return err
	}
	referenceAfterDecode, err := smoke.State(ctx, reference.Caller)
	if err != nil {
		return err
	}

	sequenceIsolation := true
	if err := sequences.CloseSequence(ctx, sequenceB); err != nil {
		return err
	}
	for _, target := range targets {
		if !smoke.ExpectWorkerError(ctx, target.Caller, workerproc.PersistentRequest{
			Command: "decode",
			Forward: &workerproc.PersistentForwardRequest{
				ShardID: target.ShardID, SequenceID: sequenceB, Position: plans[1].next,
				InputKind: inputKind(target.ShardID), Input: closedInput(target.ShardID, lastBoundary),
			},
		}, "is not open") {
			sequenceIsolation = false
		}
	}
	if err := sequences.Close(ctx); err != nil {
		return err
	}
	if !sequenceIsolation {
		return errors.New("a closed sequence accepted another decode")
	}

	producerAfterTeardown, err := smoke.State(ctx, producer.Caller)
	if err != nil {
		return err
	}
	consumerAfterTeardown, err := smoke.State(ctx, consumer.Caller)
	if err != nil {
		return err
	}
	referenceAfterTeardown, err := smoke.State(ctx, reference.Caller)
	if err != nil {
		return err
	}
	for name, snapshot := range map[string]*workerproc.PersistentWorkerState{
		"producer":  producerAfterTeardown,
		"consumer":  consumerAfterTeardown,
		"reference": referenceAfterTeardown,
	} {
		if snapshot.KVCacheBytes != 0 || len(snapshot.LoadedShards) != 1 {
			return fmt.Errorf(
				"%s teardown did not release only sequence cache: kv=%d shards=%d",
				name,
				snapshot.KVCacheBytes,
				len(snapshot.LoadedShards),
			)
		}
	}

	for _, target := range targets {
		if _, err := smoke.Call(ctx, target.Caller, workerproc.PersistentRequest{
			Command: "unloadShard",
			Shard:   &workerproc.PersistentShardRequest{ShardID: target.ShardID},
		}); err != nil {
			return err
		}
	}

	for _, client := range []*smoke.Worker{producer, consumer, reference} {
		if err := client.Shutdown(ctx); err != nil {
			return fmt.Errorf("shutdown worker: %w", err)
		}
	}

	summary := smokeSummary{
		Model:                       *model,
		ModelType:                   producerSnapshot.ModelType,
		Layers:                      *layers,
		SplitLayer:                  *split,
		DecodeStepsPerSequence:      *steps,
		SequenceCount:               len(plans),
		Distributed:                 *peer != "",
		RTol:                        *rtol,
		ATol:                        *atol,
		LogitComparisons:            logitMetrics.Comparisons,
		MaxAbsoluteDifference:       logitMetrics.MaxAbsoluteDifference,
		MaxRelativeDifference:       logitMetrics.MaxRelativeDifference,
		AllFinalLogitsMatch:         true,
		PositionsValidated:          positionsValidated,
		MutationReplayValidated:     mutationReplayValidated,
		ResourceLimitsValidated:     resourceLimitsValidated,
		SequenceIsolationValidated:  sequenceIsolation,
		ProducerKVCacheBytes:        producerAfterDecode.KVCacheBytes,
		ConsumerKVCacheBytes:        consumerAfterDecode.KVCacheBytes,
		ReferenceKVCacheBytes:       referenceAfterDecode.KVCacheBytes,
		ProducerAfterTeardownBytes:  producerAfterTeardown.KVCacheBytes,
		ConsumerAfterTeardownBytes:  consumerAfterTeardown.KVCacheBytes,
		ReferenceAfterTeardownBytes: referenceAfterTeardown.KVCacheBytes,
	}
	encoded, err := json.Marshal(summary)
	if err != nil {
		return fmt.Errorf("encode summary: %w", err)
	}
	fmt.Println(string(encoded))
	return nil
}

func proveMutationReplay(
	ctx context.Context,
	client workerproc.PersistentCaller,
	command string,
	shardID string,
	sequenceID string,
	position uint64,
	inputKind string,
	input workerproc.WireTensor,
	want *workerproc.PersistentForwardResult,
) error {
	before, err := smoke.State(ctx, client)
	if err != nil {
		return fmt.Errorf("state before replay: %w", err)
	}
	got, err := smoke.Infer(ctx, client, command, shardID, sequenceID, position, inputKind, input)
	if err != nil {
		return err
	}
	after, err := smoke.State(ctx, client)
	if err != nil {
		return fmt.Errorf("state after replay: %w", err)
	}
	if got.ShardID != want.ShardID || got.SequenceID != want.SequenceID ||
		got.Operation != want.Operation || got.Position != want.Position ||
		got.NextPosition != want.NextPosition || got.ComputeMicros != want.ComputeMicros ||
		got.KVCacheBytes != want.KVCacheBytes || got.Memory != want.Memory ||
		!equalTensor(got.Output, want.Output) {
		return errors.New("replayed mutation returned a different result")
	}
	if after.ForwardCount != before.ForwardCount {
		return fmt.Errorf(
			"replayed mutation advanced forward count from %d to %d",
			before.ForwardCount,
			after.ForwardCount,
		)
	}
	if after.KVCacheBytes != before.KVCacheBytes {
		return fmt.Errorf(
			"replayed mutation changed KV bytes from %d to %d",
			before.KVCacheBytes,
			after.KVCacheBytes,
		)
	}
	return nil
}

func equalTensor(left workerproc.WireTensor, right workerproc.WireTensor) bool {
	return left.DType == right.DType &&
		slices.Equal(left.Shape, right.Shape) &&
		bytes.Equal(left.Data, right.Data)
}

func provePositionValidation(
	ctx context.Context,
	producer workerproc.PersistentCaller,
	consumer workerproc.PersistentCaller,
	plan *sequencePlan,
) (bool, error) {
	producerInput := smoke.TokenTensor([]int32{7})
	probe, err := smoke.Infer(
		ctx,
		producer,
		"forward",
		producerShard,
		plan.id,
		0,
		"tokens",
		producerInput,
	)
	if err != nil {
		return false, fmt.Errorf("build consumer validation input: %w", err)
	}
	consumerInput := probe.Output
	checks := []struct {
		client  workerproc.PersistentCaller
		request workerproc.PersistentRequest
		want    string
	}{
		{producer, smoke.InferenceRequest("decode", producerShard, plan.id, plan.next-1, "tokens", producerInput), "expected"},
		{producer, smoke.InferenceRequest("decode", producerShard, plan.id, plan.next+1, "tokens", producerInput), "expected"},
		{producer, smoke.InferenceRequest("prefill", producerShard, plan.id, 0, "tokens", producerInput), "already prefilled"},
		{producer, smoke.InferenceRequest("decode", producerShard, "unknown-sequence", 0, "tokens", producerInput), "is not open"},
		{consumer, smoke.InferenceRequest("decode", consumerShard, plan.id, plan.next+1, "hidden", consumerInput), "expected"},
	}
	for _, check := range checks {
		if !smoke.ExpectWorkerError(ctx, check.client, check.request, check.want) {
			return false, fmt.Errorf(
				"%s %s did not fail with %q",
				check.request.Command,
				check.request.Forward.SequenceID,
				check.want,
			)
		}
	}
	return true, nil
}

func proveResourceLimits(
	ctx context.Context,
	producer workerproc.PersistentCaller,
	consumer workerproc.PersistentCaller,
	plan *sequencePlan,
) (bool, error) {
	producerTarget := smoke.SequenceTarget{Caller: producer, ShardID: producerShard}
	consumerTarget := smoke.SequenceTarget{Caller: consumer, ShardID: consumerShard}
	const cacheBudgetSequence = "cache-kv-budget"
	if err := smoke.SequenceCommand(ctx, producerTarget, "openSequence", cacheBudgetSequence); err != nil {
		return false, err
	}
	cacheBudgetOpen := true
	defer func() {
		if cacheBudgetOpen {
			_ = smoke.SequenceCommand(ctx, producerTarget, "closeSequence", cacheBudgetSequence)
		}
	}()

	longPrompt := make([]int32, 8_192)
	for index := range longPrompt {
		longPrompt[index] = int32(index%128 + 1)
	}
	producerBefore, err := smoke.State(ctx, producer)
	if err != nil {
		return false, err
	}
	if !smoke.ExpectWorkerError(
		ctx,
		producer,
		smoke.InferenceRequest(
			"prefill",
			producerShard,
			cacheBudgetSequence,
			0,
			"tokens",
			smoke.TokenTensor(longPrompt),
		),
		"retained sequence state",
	) {
		return false, errors.New("oversized initial rotating-cache write was not rejected")
	}
	producerAfter, err := smoke.State(ctx, producer)
	if err != nil {
		return false, err
	}
	if producerAfter.ForwardCount != producerBefore.ForwardCount ||
		producerAfter.KVCacheBytes != producerBefore.KVCacheBytes ||
		producerAfter.RetainedBytes != producerBefore.RetainedBytes {
		return false, errors.New("rotating-cache budget rejection mutated worker state")
	}
	if err := smoke.SequenceCommand(ctx, producerTarget, "closeSequence", cacheBudgetSequence); err != nil {
		return false, err
	}
	cacheBudgetOpen = false

	const budgetSequence = "cache-resource-budget"
	if err := smoke.SequenceCommand(ctx, consumerTarget, "openSequence", budgetSequence); err != nil {
		return false, err
	}
	budgetOpen := true
	defer func() {
		if budgetOpen {
			_ = smoke.SequenceCommand(ctx, consumerTarget, "closeSequence", budgetSequence)
		}
	}()

	longTokens := make([]int32, 129)
	for index := range longTokens {
		longTokens[index] = int32(index + 1)
	}
	boundary, err := smoke.Infer(
		ctx,
		producer,
		"forward",
		producerShard,
		plan.id,
		0,
		"tokens",
		smoke.TokenTensor(longTokens),
	)
	if err != nil {
		return false, fmt.Errorf("build retained-budget input: %w", err)
	}
	before, err := smoke.State(ctx, consumer)
	if err != nil {
		return false, err
	}
	if !smoke.ExpectWorkerError(
		ctx,
		consumer,
		smoke.InferenceRequest(
			"prefill",
			consumerShard,
			budgetSequence,
			0,
			"hidden",
			boundary.Output,
		),
		"retained sequence state",
	) {
		return false, errors.New("oversized retained state was not rejected")
	}
	after, err := smoke.State(ctx, consumer)
	if err != nil {
		return false, err
	}
	if after.ForwardCount != before.ForwardCount || after.KVCacheBytes != before.KVCacheBytes ||
		after.RetainedBytes != before.RetainedBytes {
		return false, errors.New("retained-budget rejection mutated worker state")
	}
	if err := smoke.SequenceCommand(ctx, consumerTarget, "closeSequence", budgetSequence); err != nil {
		return false, err
	}
	budgetOpen = false

	current, err := smoke.State(ctx, consumer)
	if err != nil {
		return false, err
	}
	shard, err := shardState(current, consumerShard)
	if err != nil {
		return false, err
	}
	if shard.MaxOpenSequenceCount <= shard.OpenSequenceCount {
		return false, fmt.Errorf(
			"invalid open sequence limit %d for %d open sequences",
			shard.MaxOpenSequenceCount,
			shard.OpenSequenceCount,
		)
	}
	extraCount := shard.MaxOpenSequenceCount - shard.OpenSequenceCount
	extraIDs := make([]string, 0, extraCount)
	defer func() {
		for _, sequenceID := range extraIDs {
			_ = smoke.SequenceCommand(ctx, consumerTarget, "closeSequence", sequenceID)
		}
	}()
	for index := 0; index < extraCount; index++ {
		sequenceID := fmt.Sprintf("cache-sequence-limit-%d", index)
		if err := smoke.SequenceCommand(ctx, consumerTarget, "openSequence", sequenceID); err != nil {
			return false, err
		}
		extraIDs = append(extraIDs, sequenceID)
	}
	if !smoke.ExpectWorkerError(ctx, consumer, workerproc.PersistentRequest{
		Command: "openSequence",
		Sequence: &workerproc.PersistentSequenceRequest{
			ShardID: consumerShard, SequenceID: "cache-sequence-limit-overflow",
		},
	}, "open sequence limit") {
		return false, errors.New("open sequence limit was not enforced")
	}
	for _, sequenceID := range extraIDs {
		if err := smoke.SequenceCommand(ctx, consumerTarget, "closeSequence", sequenceID); err != nil {
			return false, err
		}
	}
	extraIDs = nil
	return true, nil
}

func shardState(
	state *workerproc.PersistentWorkerState,
	shardID string,
) (*workerproc.PersistentShardSnapshot, error) {
	for index := range state.LoadedShards {
		if state.LoadedShards[index].ShardID == shardID {
			return &state.LoadedShards[index], nil
		}
	}
	return nil, fmt.Errorf("state did not contain shard %s", shardID)
}

func inputKind(shardID string) string {
	if shardID == consumerShard {
		return "hidden"
	}
	return "tokens"
}

func closedInput(shardID string, boundary workerproc.WireTensor) workerproc.WireTensor {
	if shardID != consumerShard {
		return smoke.TokenTensor([]int32{1})
	}
	return boundary
}

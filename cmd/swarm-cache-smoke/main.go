package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"time"

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
	MaxAbsoluteDifference       float64 `json:"maxAbsoluteDifference"`
	MaxRelativeDifference       float64 `json:"maxRelativeDifference"`
	AllFinalLogitsMatch         bool    `json:"allFinalLogitsMatch"`
	PositionsValidated          bool    `json:"positionsValidated"`
	MutationReplayValidated     bool    `json:"mutationReplayValidated"`
	SequenceIsolationValidated  bool    `json:"sequenceIsolationValidated"`
	ProducerKVCacheBytes        int     `json:"producerKVCacheBytes"`
	ConsumerKVCacheBytes        int     `json:"consumerKVCacheBytes"`
	ReferenceKVCacheBytes       int     `json:"referenceKVCacheBytes"`
	ProducerAfterTeardownBytes  int     `json:"producerAfterTeardownBytes"`
	ConsumerAfterTeardownBytes  int     `json:"consumerAfterTeardownBytes"`
	ReferenceAfterTeardownBytes int     `json:"referenceAfterTeardownBytes"`
}

type managedClient struct {
	caller workerproc.PersistentCaller
	direct *workerproc.PersistentClient
	clean  bool
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

	producer, err := startDirect(*worker)
	if err != nil {
		return fmt.Errorf("start producer worker: %w", err)
	}
	defer producer.terminate()

	reference, err := startDirect(*worker)
	if err != nil {
		return fmt.Errorf("start reference worker: %w", err)
	}
	defer reference.terminate()

	var consumer *managedClient
	if *peer == "" {
		consumer, err = startDirect(*worker)
		if err != nil {
			return fmt.Errorf("start consumer worker: %w", err)
		}
		defer consumer.terminate()
	} else {
		httpClient, configureErr := workerproc.NewHTTPPersistentClient(*peer, nil)
		if configureErr != nil {
			return fmt.Errorf("configure consumer worker: %w", configureErr)
		}
		consumer = &managedClient{caller: httpClient}
	}

	for name, client := range map[string]*managedClient{
		"producer":  producer,
		"consumer":  consumer,
		"reference": reference,
	} {
		if _, err := call(ctx, client.caller, workerproc.PersistentRequest{Command: "health"}); err != nil {
			return fmt.Errorf("%s %w", name, err)
		}
	}

	producerSnapshot, err := loadShard(ctx, producer.caller, workerproc.PersistentLoadShardRequest{
		ModelID: *model, ShardID: producerShard,
		LayerStart: 0, LayerEnd: *split, OwnsInput: true,
	})
	if err != nil {
		return fmt.Errorf("load producer: %w", err)
	}
	consumerSnapshot, err := loadShard(ctx, consumer.caller, workerproc.PersistentLoadShardRequest{
		ModelID: *model, ShardID: consumerShard,
		LayerStart: *split, LayerEnd: *layers, OwnsOutput: true,
	})
	if err != nil {
		return fmt.Errorf("load consumer: %w", err)
	}
	referenceSnapshot, err := loadShard(ctx, reference.caller, workerproc.PersistentLoadShardRequest{
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
	for _, plan := range plans {
		for _, target := range []struct {
			client workerproc.PersistentCaller
			shard  string
		}{
			{producer.caller, producerShard},
			{consumer.caller, consumerShard},
			{reference.caller, referenceShard},
		} {
			if err := sequenceCommand(ctx, target.client, "openSequence", target.shard, plan.id); err != nil {
				return err
			}
		}
	}

	maxAbsolute := 0.0
	maxRelative := 0.0
	var lastBoundary workerproc.WireTensor
	mutationReplayValidated := true
	for _, plan := range plans {
		prompt := tokenTensor(plan.prompt)
		producerResult, err := infer(
			ctx, producer.caller, "prefill", producerShard, plan.id, 0, "tokens", prompt,
		)
		if err != nil {
			return err
		}
		if err := proveMutationReplay(
			ctx, producer.caller, "prefill", producerShard, plan.id, 0, "tokens", prompt,
			producerResult,
		); err != nil {
			return fmt.Errorf("producer prefill replay for %s: %w", plan.id, err)
		}
		lastBoundary = producerResult.Output
		consumerResult, err := infer(
			ctx, consumer.caller, "prefill", consumerShard, plan.id, 0, "hidden",
			producerResult.Output,
		)
		if err != nil {
			return err
		}
		if err := proveMutationReplay(
			ctx, consumer.caller, "prefill", consumerShard, plan.id, 0, "hidden",
			producerResult.Output, consumerResult,
		); err != nil {
			return fmt.Errorf("consumer prefill replay for %s: %w", plan.id, err)
		}
		referenceResult, err := infer(
			ctx, reference.caller, "prefill", referenceShard, plan.id, 0, "tokens", prompt,
		)
		if err != nil {
			return err
		}
		if err := proveMutationReplay(
			ctx, reference.caller, "prefill", referenceShard, plan.id, 0, "tokens", prompt,
			referenceResult,
		); err != nil {
			return fmt.Errorf("reference prefill replay for %s: %w", plan.id, err)
		}
		absolute, relative, err := compareFinalLogits(
			consumerResult.Output,
			referenceResult.Output,
			*rtol,
			*atol,
		)
		if err != nil {
			return fmt.Errorf("prefill logits for %s: %w", plan.id, err)
		}
		maxAbsolute = math.Max(maxAbsolute, absolute)
		maxRelative = math.Max(maxRelative, relative)
		plan.next = uint64(len(plan.prompt))
		if producerResult.NextPosition != plan.next ||
			consumerResult.NextPosition != plan.next ||
			referenceResult.NextPosition != plan.next {
			return fmt.Errorf("prefill next position mismatch for %s", plan.id)
		}
	}

	positionsValidated, err := provePositionValidation(ctx, producer.caller, consumer.caller, plans[0])
	if err != nil {
		return err
	}

	producerAfterPrefill, err := state(ctx, producer.caller)
	if err != nil {
		return err
	}
	consumerAfterPrefill, err := state(ctx, consumer.caller)
	if err != nil {
		return err
	}
	referenceAfterPrefill, err := state(ctx, reference.caller)
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
			token := tokenTensor([]int32{plan.tokenAdd + int32(step)})
			producerResult, err := infer(
				ctx, producer.caller, "decode", producerShard, plan.id, plan.next, "tokens", token,
			)
			if err != nil {
				return fmt.Errorf("producer decode %s step %d: %w", plan.id, step, err)
			}
			if step == 0 {
				if err := proveMutationReplay(
					ctx, producer.caller, "decode", producerShard, plan.id, plan.next, "tokens",
					token, producerResult,
				); err != nil {
					return fmt.Errorf("producer decode replay for %s: %w", plan.id, err)
				}
			}
			lastBoundary = producerResult.Output
			consumerResult, err := infer(
				ctx, consumer.caller, "decode", consumerShard, plan.id, plan.next, "hidden",
				producerResult.Output,
			)
			if err != nil {
				return fmt.Errorf("consumer decode %s step %d: %w", plan.id, step, err)
			}
			if step == 0 {
				if err := proveMutationReplay(
					ctx, consumer.caller, "decode", consumerShard, plan.id, plan.next, "hidden",
					producerResult.Output, consumerResult,
				); err != nil {
					return fmt.Errorf("consumer decode replay for %s: %w", plan.id, err)
				}
			}
			referenceResult, err := infer(
				ctx, reference.caller, "decode", referenceShard, plan.id, plan.next, "tokens", token,
			)
			if err != nil {
				return fmt.Errorf("reference decode %s step %d: %w", plan.id, step, err)
			}
			if step == 0 {
				if err := proveMutationReplay(
					ctx, reference.caller, "decode", referenceShard, plan.id, plan.next, "tokens",
					token, referenceResult,
				); err != nil {
					return fmt.Errorf("reference decode replay for %s: %w", plan.id, err)
				}
			}
			absolute, relative, err := compareFinalLogits(
				consumerResult.Output,
				referenceResult.Output,
				*rtol,
				*atol,
			)
			if err != nil {
				return fmt.Errorf("decode logits for %s step %d: %w", plan.id, step, err)
			}
			maxAbsolute = math.Max(maxAbsolute, absolute)
			maxRelative = math.Max(maxRelative, relative)
			plan.next++
			if producerResult.NextPosition != plan.next ||
				consumerResult.NextPosition != plan.next ||
				referenceResult.NextPosition != plan.next {
				return fmt.Errorf("decode next position mismatch for %s step %d", plan.id, step)
			}
		}
	}

	producerAfterDecode, err := state(ctx, producer.caller)
	if err != nil {
		return err
	}
	consumerAfterDecode, err := state(ctx, consumer.caller)
	if err != nil {
		return err
	}
	referenceAfterDecode, err := state(ctx, reference.caller)
	if err != nil {
		return err
	}

	sequenceIsolation := true
	for _, target := range []struct {
		client workerproc.PersistentCaller
		shard  string
	}{
		{producer.caller, producerShard},
		{consumer.caller, consumerShard},
		{reference.caller, referenceShard},
	} {
		if err := sequenceCommand(ctx, target.client, "closeSequence", target.shard, sequenceB); err != nil {
			return err
		}
		if !expectWorkerError(ctx, target.client, workerproc.PersistentRequest{
			Command: "decode",
			Forward: &workerproc.PersistentForwardRequest{
				ShardID: target.shard, SequenceID: sequenceB, Position: plans[1].next,
				InputKind: inputKind(target.shard), Input: closedInput(target.shard, lastBoundary),
			},
		}, "is not open") {
			sequenceIsolation = false
		}
		if err := sequenceCommand(ctx, target.client, "closeSequence", target.shard, sequenceA); err != nil {
			return err
		}
	}
	if !sequenceIsolation {
		return errors.New("a closed sequence accepted another decode")
	}

	producerAfterTeardown, err := state(ctx, producer.caller)
	if err != nil {
		return err
	}
	consumerAfterTeardown, err := state(ctx, consumer.caller)
	if err != nil {
		return err
	}
	referenceAfterTeardown, err := state(ctx, reference.caller)
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

	for _, target := range []struct {
		client workerproc.PersistentCaller
		shard  string
	}{
		{producer.caller, producerShard},
		{consumer.caller, consumerShard},
		{reference.caller, referenceShard},
	} {
		if _, err := call(ctx, target.client, workerproc.PersistentRequest{
			Command: "unloadShard",
			Shard:   &workerproc.PersistentShardRequest{ShardID: target.shard},
		}); err != nil {
			return err
		}
	}

	for _, client := range []*managedClient{producer, consumer, reference} {
		if client.direct == nil {
			continue
		}
		if err := client.direct.Shutdown(ctx); err != nil {
			return fmt.Errorf("shutdown worker: %w", err)
		}
		client.clean = true
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
		MaxAbsoluteDifference:       maxAbsolute,
		MaxRelativeDifference:       maxRelative,
		AllFinalLogitsMatch:         true,
		PositionsValidated:          positionsValidated,
		MutationReplayValidated:     mutationReplayValidated,
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

func startDirect(worker string) (*managedClient, error) {
	client, err := workerproc.StartPersistent(worker)
	if err != nil {
		return nil, err
	}
	return &managedClient{caller: client, direct: client}, nil
}

func (c *managedClient) terminate() {
	if c == nil || c.direct == nil || c.clean {
		return
	}
	_ = c.direct.Kill()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = c.direct.Wait(ctx)
}

func call(
	ctx context.Context,
	client workerproc.PersistentCaller,
	request workerproc.PersistentRequest,
) (workerproc.PersistentResponse, error) {
	response, err := client.Call(ctx, request)
	if err != nil {
		return workerproc.PersistentResponse{}, fmt.Errorf("%s: %w", request.Command, err)
	}
	return response, nil
}

func loadShard(
	ctx context.Context,
	client workerproc.PersistentCaller,
	request workerproc.PersistentLoadShardRequest,
) (*workerproc.PersistentShardSnapshot, error) {
	response, err := call(ctx, client, workerproc.PersistentRequest{
		Command:   "loadShard",
		LoadShard: &request,
	})
	if err != nil {
		return nil, err
	}
	if response.Result == nil || response.Result.Shard == nil {
		return nil, errors.New("loadShard returned no shard snapshot")
	}
	return response.Result.Shard, nil
}

func sequenceCommand(
	ctx context.Context,
	client workerproc.PersistentCaller,
	command string,
	shardID string,
	sequenceID string,
) error {
	_, err := call(ctx, client, workerproc.PersistentRequest{
		Command: command,
		Sequence: &workerproc.PersistentSequenceRequest{
			ShardID: shardID, SequenceID: sequenceID,
		},
	})
	return err
}

func infer(
	ctx context.Context,
	client workerproc.PersistentCaller,
	command string,
	shardID string,
	sequenceID string,
	position uint64,
	inputKind string,
	input workerproc.WireTensor,
) (*workerproc.PersistentForwardResult, error) {
	response, err := call(ctx, client, workerproc.PersistentRequest{
		Command: command,
		Forward: &workerproc.PersistentForwardRequest{
			ShardID: shardID, SequenceID: sequenceID, Position: position,
			InputKind: inputKind, Input: input,
		},
	})
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
			command,
			result.Operation,
			result.Position,
			result.KVCacheBytes,
		)
	}
	return result, nil
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
	before, err := state(ctx, client)
	if err != nil {
		return fmt.Errorf("state before replay: %w", err)
	}
	got, err := infer(ctx, client, command, shardID, sequenceID, position, inputKind, input)
	if err != nil {
		return err
	}
	after, err := state(ctx, client)
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

func state(
	ctx context.Context,
	client workerproc.PersistentCaller,
) (*workerproc.PersistentWorkerState, error) {
	response, err := call(ctx, client, workerproc.PersistentRequest{Command: "state"})
	if err != nil {
		return nil, err
	}
	if response.Result == nil || response.Result.State == nil {
		return nil, errors.New("state returned no snapshot")
	}
	return response.Result.State, nil
}

func provePositionValidation(
	ctx context.Context,
	producer workerproc.PersistentCaller,
	consumer workerproc.PersistentCaller,
	plan *sequencePlan,
) (bool, error) {
	producerInput := tokenTensor([]int32{7})
	probe, err := infer(
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
		{producer, inferenceRequest("decode", producerShard, plan.id, plan.next-1, "tokens", producerInput), "expected"},
		{producer, inferenceRequest("decode", producerShard, plan.id, plan.next+1, "tokens", producerInput), "expected"},
		{producer, inferenceRequest("prefill", producerShard, plan.id, 0, "tokens", producerInput), "already prefilled"},
		{producer, inferenceRequest("decode", producerShard, "unknown-sequence", 0, "tokens", producerInput), "is not open"},
		{consumer, inferenceRequest("decode", consumerShard, plan.id, plan.next+1, "hidden", consumerInput), "expected"},
	}
	for _, check := range checks {
		if !expectWorkerError(ctx, check.client, check.request, check.want) {
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

func inferenceRequest(
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

func expectWorkerError(
	ctx context.Context,
	client workerproc.PersistentCaller,
	request workerproc.PersistentRequest,
	want string,
) bool {
	_, err := client.Call(ctx, request)
	var responseError *workerproc.WorkerResponseError
	return errors.As(err, &responseError) && strings.Contains(responseError.Message, want)
}

func tokenTensor(tokens []int32) workerproc.WireTensor {
	data := make([]byte, len(tokens)*4)
	for index, token := range tokens {
		binary.LittleEndian.PutUint32(data[index*4:], uint32(token))
	}
	return workerproc.WireTensor{
		Shape: []int{1, len(tokens)}, DType: "int32", Data: data,
	}
}

func inputKind(shardID string) string {
	if shardID == consumerShard {
		return "hidden"
	}
	return "tokens"
}

func closedInput(shardID string, boundary workerproc.WireTensor) workerproc.WireTensor {
	if shardID != consumerShard {
		return tokenTensor([]int32{1})
	}
	return boundary
}

func compareFinalLogits(
	got workerproc.WireTensor,
	want workerproc.WireTensor,
	rtol float64,
	atol float64,
) (float64, float64, error) {
	if len(got.Shape) == 0 || len(want.Shape) == 0 {
		return 0, 0, errors.New("logit tensor has no dimensions")
	}
	if len(got.Shape) != len(want.Shape) {
		return 0, 0, fmt.Errorf("rank mismatch: got %v want %v", got.Shape, want.Shape)
	}
	for index := range got.Shape {
		if got.Shape[index] != want.Shape[index] {
			return 0, 0, fmt.Errorf("shape mismatch: got %v want %v", got.Shape, want.Shape)
		}
	}
	vocabulary := got.Shape[len(got.Shape)-1]
	gotValues, err := finalValues(got, vocabulary)
	if err != nil {
		return 0, 0, fmt.Errorf("decode distributed logits: %w", err)
	}
	wantValues, err := finalValues(want, vocabulary)
	if err != nil {
		return 0, 0, fmt.Errorf("decode reference logits: %w", err)
	}
	maxAbsolute := 0.0
	maxRelative := 0.0
	for index := range gotValues {
		actual := gotValues[index]
		expected := wantValues[index]
		if math.IsNaN(actual) || math.IsNaN(expected) {
			return 0, 0, fmt.Errorf("NaN at vocabulary index %d", index)
		}
		absolute := math.Abs(actual - expected)
		relative := absolute / math.Max(math.Abs(expected), math.SmallestNonzeroFloat64)
		maxAbsolute = math.Max(maxAbsolute, absolute)
		maxRelative = math.Max(maxRelative, relative)
		if absolute > atol+rtol*math.Abs(expected) {
			return maxAbsolute, maxRelative, fmt.Errorf(
				"index %d differs: got=%g want=%g abs=%g tolerance=%g",
				index,
				actual,
				expected,
				absolute,
				atol+rtol*math.Abs(expected),
			)
		}
	}
	return maxAbsolute, maxRelative, nil
}

func finalValues(tensor workerproc.WireTensor, count int) ([]float64, error) {
	var elementBytes int
	switch tensor.DType {
	case "bfloat16", "float16":
		elementBytes = 2
	case "float32":
		elementBytes = 4
	default:
		return nil, fmt.Errorf("unsupported logit dtype %q", tensor.DType)
	}
	required := count * elementBytes
	if count <= 0 || len(tensor.Data) < required || len(tensor.Data)%elementBytes != 0 {
		return nil, fmt.Errorf("invalid %s logit payload size %d", tensor.DType, len(tensor.Data))
	}
	data := tensor.Data[len(tensor.Data)-required:]
	values := make([]float64, count)
	for index := range values {
		offset := index * elementBytes
		switch tensor.DType {
		case "bfloat16":
			bits := uint32(binary.LittleEndian.Uint16(data[offset:])) << 16
			values[index] = float64(math.Float32frombits(bits))
		case "float16":
			values[index] = float64(float16(binary.LittleEndian.Uint16(data[offset:])))
		case "float32":
			values[index] = float64(math.Float32frombits(binary.LittleEndian.Uint32(data[offset:])))
		}
	}
	return values, nil
}

func float16(bits uint16) float32 {
	sign := 1.0
	if bits&0x8000 != 0 {
		sign = -1
	}
	exponent := int(bits>>10) & 0x1f
	fraction := int(bits & 0x03ff)
	switch exponent {
	case 0:
		if fraction == 0 {
			return float32(math.Copysign(0, sign))
		}
		return float32(sign * math.Ldexp(float64(fraction), -24))
	case 0x1f:
		if fraction == 0 {
			return float32(math.Inf(int(sign)))
		}
		return float32(math.NaN())
	}
	value := math.Ldexp(1+float64(fraction)/1024, exponent-15)
	return float32(sign * value)
}

package pooledproof

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

const (
	SchemaVersion              = 1
	DefaultModelID             = "mlx-community/gemma-3-text-12b-it-4bit"
	DefaultPrompt              = "Write a short story about two computers working together:"
	DefaultMinimumTokens       = 32
	DefaultWorkerMemoryLimit   = 6 * 1024 * 1024 * 1024
	defaultCapabilitiesMaxBody = 1 << 20
	defaultCleanupTimeout      = 10 * time.Second
)

type Capabilities struct {
	Runtime                   string   `json:"runtime"`
	Device                    string   `json:"device"`
	CheckpointShardModelTypes []string `json:"checkpointShardModelTypes"`
	PhysicalMemoryBytes       uint64   `json:"physicalMemoryBytes"`
	MLXMemoryLimitBytes       int      `json:"mlxMemoryLimitBytes"`
	MLXCacheLimitBytes        int      `json:"mlxCacheLimitBytes"`
}

type MemoryEvidence struct {
	Load             workerproc.StageMemory `json:"load"`
	Prefill          workerproc.StageMemory `json:"prefill"`
	Decode           workerproc.StageMemory `json:"decode"`
	MaxObservedBytes int                    `json:"maxObservedBytes"`
}

type Reference struct {
	SchemaVersion         int                     `json:"schemaVersion"`
	Model                 string                  `json:"model"`
	ModelType             string                  `json:"modelType"`
	LayerCount            int                     `json:"layerCount"`
	CheckpointFingerprint string                  `json:"checkpointFingerprint"`
	CheckpointBytes       uint64                  `json:"checkpointBytes"`
	Prompt                string                  `json:"prompt"`
	PromptTokenIDs        []int32                 `json:"promptTokenIDs"`
	GeneratedTokenIDs     []int32                 `json:"generatedTokenIDs"`
	GeneratedText         string                  `json:"generatedText"`
	StopReason            string                  `json:"stopReason"`
	FullCheckpointMemory  MemoryEvidence          `json:"fullCheckpointMemory"`
	SourceHardware        Capabilities            `json:"sourceHardware"`
	ReferenceVerification generation.Verification `json:"referenceVerification"`
}

type WorkerEvidence struct {
	Endpoint     string                             `json:"endpoint"`
	Capabilities Capabilities                       `json:"capabilities"`
	Shard        workerproc.PersistentShardSnapshot `json:"shard"`
	Memory       MemoryEvidence                     `json:"memory"`
}

type Checks struct {
	CleanWorkersAtStart               bool `json:"cleanWorkersAtStart"`
	CheckpointMatchesReference        bool `json:"checkpointMatchesReference"`
	CheckpointExceedsProducerLimit    bool `json:"checkpointExceedsProducerLimit"`
	CheckpointExceedsConsumerLimit    bool `json:"checkpointExceedsConsumerLimit"`
	FullInferenceExceedsProducerLimit bool `json:"fullInferenceExceedsProducerLimit"`
	FullInferenceExceedsConsumerLimit bool `json:"fullInferenceExceedsConsumerLimit"`
	ProducerUsesConfiguredLimit       bool `json:"producerUsesConfiguredLimit"`
	ConsumerUsesConfiguredLimit       bool `json:"consumerUsesConfiguredLimit"`
	ProducerWithinLimit               bool `json:"producerWithinLimit"`
	ConsumerWithinLimit               bool `json:"consumerWithinLimit"`
	ComplementaryShardsOnly           bool `json:"complementaryShardsOnly"`
	NoServingFullModelOracle          bool `json:"noServingFullModelOracle"`
	PromptTokensMatchReference        bool `json:"promptTokensMatchReference"`
	GeneratedTokensMatchReference     bool `json:"generatedTokensMatchReference"`
	GeneratedAtLeastMinimumTokens     bool `json:"generatedAtLeastMinimumTokens"`
	SequenceStateReleased             bool `json:"sequenceStateReleased"`
	AllPassed                         bool `json:"allPassed"`
}

type Result struct {
	SchemaVersion              int               `json:"schemaVersion"`
	Model                      string            `json:"model"`
	ModelType                  string            `json:"modelType"`
	CheckpointFingerprint      string            `json:"checkpointFingerprint"`
	CheckpointBytes            uint64            `json:"checkpointBytes"`
	ConfiguredMemoryLimitBytes int               `json:"configuredMemoryLimitBytes"`
	MinimumGeneratedTokens     int               `json:"minimumGeneratedTokens"`
	Producer                   WorkerEvidence    `json:"producer"`
	Consumer                   WorkerEvidence    `json:"consumer"`
	Reference                  Reference         `json:"reference"`
	Generation                 generation.Result `json:"generation"`
	Checks                     Checks            `json:"checks"`
}

type ReferenceConfig struct {
	WorkerPath     string
	Model          string
	Prompt         string
	MaxTokens      int
	RTol           float64
	ATol           float64
	ForwardTimeout time.Duration
}

type RunConfig struct {
	ProducerURL              string
	ConsumerURL              string
	ExpectedMemoryLimitBytes int
	MinimumGeneratedTokens   int
	RTol                     float64
	ATol                     float64
	ForwardTimeout           time.Duration
	HTTPClient               *http.Client
}

func LoadReference(path string) (Reference, error) {
	file, err := os.Open(path)
	if err != nil {
		return Reference{}, err
	}
	defer file.Close()
	decoder := json.NewDecoder(io.LimitReader(file, defaultCapabilitiesMaxBody))
	decoder.DisallowUnknownFields()
	var reference Reference
	if err := decoder.Decode(&reference); err != nil {
		return Reference{}, fmt.Errorf("decode pooled-memory reference: %w", err)
	}
	if err := ValidateReference(reference); err != nil {
		return Reference{}, err
	}
	return reference, nil
}

func ValidateReference(reference Reference) error {
	switch {
	case reference.SchemaVersion != SchemaVersion:
		return fmt.Errorf("reference schema version is %d, want %d", reference.SchemaVersion, SchemaVersion)
	case reference.Model == "" || reference.ModelType == "":
		return errors.New("reference model metadata is incomplete")
	case reference.LayerCount < 2:
		return errors.New("reference layer count must be at least two")
	case reference.CheckpointFingerprint == "" || reference.CheckpointBytes == 0:
		return errors.New("reference checkpoint identity is incomplete")
	case reference.Prompt == "" || len(reference.PromptTokenIDs) == 0:
		return errors.New("reference prompt token plan is empty")
	case len(reference.GeneratedTokenIDs) < DefaultMinimumTokens:
		return fmt.Errorf("reference has %d generated tokens; need at least %d", len(reference.GeneratedTokenIDs), DefaultMinimumTokens)
	case !completeMemoryEvidence(reference.FullCheckpointMemory):
		return errors.New("reference full-checkpoint memory evidence is empty")
	case !reference.ReferenceVerification.GreedyTokenIDsMatch:
		return errors.New("reference did not match upstream full-model greedy tokens")
	case reference.ReferenceVerification.ComparedTokens != len(reference.GeneratedTokenIDs):
		return errors.New("reference did not compare every generated token")
	}
	return nil
}

func CreateReference(ctx context.Context, config ReferenceConfig) (Reference, error) {
	if config.WorkerPath == "" {
		return Reference{}, errors.New("worker path is required")
	}
	if config.Model == "" {
		config.Model = DefaultModelID
	}
	if config.Prompt == "" {
		config.Prompt = DefaultPrompt
	}
	if config.MaxTokens < DefaultMinimumTokens {
		return Reference{}, fmt.Errorf("max tokens must be at least %d", DefaultMinimumTokens)
	}
	if config.ForwardTimeout < time.Millisecond {
		return Reference{}, errors.New("forward timeout must be at least 1ms")
	}

	capabilities, err := localCapabilities(ctx, config.WorkerPath)
	if err != nil {
		return Reference{}, fmt.Errorf("reference hardware capabilities: %w", err)
	}
	producer, err := workerproc.OpenPersistentTarget(config.WorkerPath, "")
	if err != nil {
		return Reference{}, fmt.Errorf("reference producer: %w", err)
	}
	defer producer.Cleanup()
	consumer, err := workerproc.OpenPersistentTarget(config.WorkerPath, "")
	if err != nil {
		return Reference{}, fmt.Errorf("reference consumer: %w", err)
	}
	defer consumer.Cleanup()
	oracle, err := workerproc.OpenPersistentTarget(config.WorkerPath, "")
	if err != nil {
		return Reference{}, fmt.Errorf("full-model oracle: %w", err)
	}
	defer oracle.Cleanup()

	var fullMemory MemoryEvidence
	session, err := generation.NewSession(
		ctx, producer.Caller, consumer.Caller, oracle.Caller,
		generation.SessionConfig{
			Model: config.Model, RTol: config.RTol, ATol: config.ATol,
			ForwardTimeout: config.ForwardTimeout,
			Observer: func(sample generation.StageSample) {
				observePhase(&fullMemory, sample.Operation, sample.ReferenceMemory)
			},
		},
	)
	if err != nil {
		return Reference{}, fmt.Errorf("prepare reference session: %w", err)
	}
	oracleState, err := workerproc.State(ctx, oracle.Caller)
	if err != nil {
		return Reference{}, fmt.Errorf("full-model oracle state: %w", err)
	}
	if len(oracleState.LoadedShards) != 1 {
		return Reference{}, fmt.Errorf("full-model oracle loaded %d shards, want one", len(oracleState.LoadedShards))
	}
	fullMemory.Load = oracleState.LoadedShards[0].LoadedMemory
	updateMaxObserved(&fullMemory, fullMemory.Load)

	generated, err := session.Generate(ctx, generation.Request{
		Prompt: config.Prompt, MaxTokens: config.MaxTokens,
	})
	if err != nil {
		return Reference{}, fmt.Errorf("generate reference: %w", err)
	}
	if generated.Verification == nil {
		return Reference{}, errors.New("reference generation omitted full-model verification")
	}
	info := session.Info()
	reference := Reference{
		SchemaVersion: SchemaVersion,
		Model:         generated.Model, ModelType: generated.ModelType,
		LayerCount:            info.Model.LayerCount,
		CheckpointFingerprint: generated.CheckpointFingerprint,
		CheckpointBytes:       generated.CheckpointBytes,
		Prompt:                generated.Prompt,
		PromptTokenIDs:        append([]int32(nil), generated.PromptTokenIDs...),
		GeneratedTokenIDs:     append([]int32(nil), generated.GeneratedTokenIDs...),
		GeneratedText:         generated.Text, StopReason: generated.StopReason,
		FullCheckpointMemory: fullMemory, SourceHardware: capabilities,
		ReferenceVerification: *generated.Verification,
	}
	if err := ValidateReference(reference); err != nil {
		return Reference{}, err
	}
	return reference, nil
}

func Run(ctx context.Context, reference Reference, config RunConfig) (result Result, returnErr error) {
	if err := ValidateReference(reference); err != nil {
		return Result{}, err
	}
	if config.ProducerURL == "" || config.ConsumerURL == "" {
		return Result{}, errors.New("producer and consumer swarmd URLs are required")
	}
	if strings.TrimRight(config.ProducerURL, "/") == strings.TrimRight(config.ConsumerURL, "/") {
		return Result{}, errors.New("producer and consumer must be different swarmd endpoints")
	}
	if config.ExpectedMemoryLimitBytes <= 0 {
		config.ExpectedMemoryLimitBytes = DefaultWorkerMemoryLimit
	}
	if config.MinimumGeneratedTokens < DefaultMinimumTokens {
		config.MinimumGeneratedTokens = DefaultMinimumTokens
	}
	if config.ForwardTimeout < time.Millisecond {
		return Result{}, errors.New("forward timeout must be at least 1ms")
	}
	if config.HTTPClient == nil {
		config.HTTPClient = http.DefaultClient
	}

	producerCapabilities, err := remoteCapabilities(ctx, config.HTTPClient, config.ProducerURL)
	if err != nil {
		return Result{}, fmt.Errorf("producer capabilities: %w", err)
	}
	consumerCapabilities, err := remoteCapabilities(ctx, config.HTTPClient, config.ConsumerURL)
	if err != nil {
		return Result{}, fmt.Errorf("consumer capabilities: %w", err)
	}
	producer, err := workerproc.OpenPersistentTargetWithHTTPClient(
		"", config.ProducerURL, config.HTTPClient,
	)
	if err != nil {
		return Result{}, fmt.Errorf("producer: %w", err)
	}
	defer producer.Cleanup()
	consumer, err := workerproc.OpenPersistentTargetWithHTTPClient(
		"", config.ConsumerURL, config.HTTPClient,
	)
	if err != nil {
		return Result{}, fmt.Errorf("consumer: %w", err)
	}
	defer consumer.Cleanup()

	producerInitial, err := workerproc.State(ctx, producer.Caller)
	if err != nil {
		return Result{}, fmt.Errorf("producer initial state: %w", err)
	}
	consumerInitial, err := workerproc.State(ctx, consumer.Caller)
	if err != nil {
		return Result{}, fmt.Errorf("consumer initial state: %w", err)
	}
	cleanAtStart := cleanWorker(producerInitial) && cleanWorker(consumerInitial)
	if !cleanAtStart {
		return Result{}, errors.New("producer and consumer workers must be clean before pooled-memory proof")
	}
	defer func() {
		cleanupErr := cleanupProofShards(
			proofCleanupTarget{
				Name: "producer", Caller: producer.Caller, ShardIDPrefix: "generate-producer-",
			},
			proofCleanupTarget{
				Name: "consumer", Caller: consumer.Caller, ShardIDPrefix: "generate-consumer-",
			},
		)
		if cleanupErr != nil {
			returnErr = errors.Join(returnErr, fmt.Errorf("clean up pooled-memory proof: %w", cleanupErr))
		}
	}()

	producerEvidence := WorkerEvidence{Endpoint: config.ProducerURL, Capabilities: producerCapabilities}
	consumerEvidence := WorkerEvidence{Endpoint: config.ConsumerURL, Capabilities: consumerCapabilities}
	session, err := generation.NewSession(
		ctx, producer.Caller, consumer.Caller, nil,
		generation.SessionConfig{
			Model: reference.Model, RTol: config.RTol, ATol: config.ATol,
			ForwardTimeout: config.ForwardTimeout,
			Observer: func(sample generation.StageSample) {
				observePhase(&producerEvidence.Memory, sample.Operation, sample.ProducerMemory)
				observePhase(&consumerEvidence.Memory, sample.Operation, sample.ConsumerMemory)
			},
		},
	)
	if err != nil {
		return Result{}, fmt.Errorf("prepare pooled-memory session: %w", err)
	}
	producerLoaded, err := workerproc.State(ctx, producer.Caller)
	if err != nil {
		return Result{}, fmt.Errorf("producer loaded state: %w", err)
	}
	consumerLoaded, err := workerproc.State(ctx, consumer.Caller)
	if err != nil {
		return Result{}, fmt.Errorf("consumer loaded state: %w", err)
	}
	if len(producerLoaded.LoadedShards) == 1 {
		producerEvidence.Shard = producerLoaded.LoadedShards[0]
		producerEvidence.Memory.Load = producerEvidence.Shard.LoadedMemory
		updateMaxObserved(&producerEvidence.Memory, producerEvidence.Memory.Load)
	}
	if len(consumerLoaded.LoadedShards) == 1 {
		consumerEvidence.Shard = consumerLoaded.LoadedShards[0]
		consumerEvidence.Memory.Load = consumerEvidence.Shard.LoadedMemory
		updateMaxObserved(&consumerEvidence.Memory, consumerEvidence.Memory.Load)
	}

	generated, generationErr := session.Generate(ctx, generation.Request{
		Prompt: reference.Prompt, MaxTokens: len(reference.GeneratedTokenIDs),
	})
	producerFinal, producerStateErr := workerproc.State(ctx, producer.Caller)
	consumerFinal, consumerStateErr := workerproc.State(ctx, consumer.Caller)
	if generationErr != nil {
		return Result{}, fmt.Errorf("generate pooled-memory proof: %w", generationErr)
	}
	if producerStateErr != nil {
		return Result{}, fmt.Errorf("producer final state: %w", producerStateErr)
	}
	if consumerStateErr != nil {
		return Result{}, fmt.Errorf("consumer final state: %w", consumerStateErr)
	}
	if len(producerFinal.LoadedShards) == 1 {
		producerEvidence.Shard = producerFinal.LoadedShards[0]
	}
	if len(consumerFinal.LoadedShards) == 1 {
		consumerEvidence.Shard = consumerFinal.LoadedShards[0]
	}

	info := session.Info()
	complementary := complementaryShards(
		producerLoaded, consumerLoaded, info.ShardPlan, info.Model.LayerCount,
	)
	noOracle := complementary &&
		!producerEvidence.Shard.OwnsOutput && !consumerEvidence.Shard.OwnsInput
	checks := Checks{
		CleanWorkersAtStart: cleanAtStart,
		CheckpointMatchesReference: generated.CheckpointFingerprint == reference.CheckpointFingerprint &&
			generated.CheckpointBytes == reference.CheckpointBytes &&
			generated.ModelType == reference.ModelType && info.Model.LayerCount == reference.LayerCount,
		CheckpointExceedsProducerLimit:    reference.CheckpointBytes > uint64(producerCapabilities.MLXMemoryLimitBytes),
		CheckpointExceedsConsumerLimit:    reference.CheckpointBytes > uint64(consumerCapabilities.MLXMemoryLimitBytes),
		FullInferenceExceedsProducerLimit: reference.FullCheckpointMemory.MaxObservedBytes > producerCapabilities.MLXMemoryLimitBytes,
		FullInferenceExceedsConsumerLimit: reference.FullCheckpointMemory.MaxObservedBytes > consumerCapabilities.MLXMemoryLimitBytes,
		ProducerUsesConfiguredLimit: configuredLimit(
			producerCapabilities, producerInitial, config.ExpectedMemoryLimitBytes,
		),
		ConsumerUsesConfiguredLimit: configuredLimit(
			consumerCapabilities, consumerInitial, config.ExpectedMemoryLimitBytes,
		),
		ProducerWithinLimit: completeMemoryEvidence(producerEvidence.Memory) &&
			producerEvidence.Memory.MaxObservedBytes <= producerCapabilities.MLXMemoryLimitBytes,
		ConsumerWithinLimit: completeMemoryEvidence(consumerEvidence.Memory) &&
			consumerEvidence.Memory.MaxObservedBytes <= consumerCapabilities.MLXMemoryLimitBytes,
		ComplementaryShardsOnly:       complementary,
		NoServingFullModelOracle:      noOracle,
		PromptTokensMatchReference:    slices.Equal(generated.PromptTokenIDs, reference.PromptTokenIDs),
		GeneratedTokensMatchReference: slices.Equal(generated.GeneratedTokenIDs, reference.GeneratedTokenIDs),
		GeneratedAtLeastMinimumTokens: len(generated.GeneratedTokenIDs) >= config.MinimumGeneratedTokens,
		SequenceStateReleased:         releasedWorker(producerFinal) && releasedWorker(consumerFinal),
	}
	checks.AllPassed = allChecksPassed(checks)
	result = Result{
		SchemaVersion: SchemaVersion,
		Model:         generated.Model, ModelType: generated.ModelType,
		CheckpointFingerprint:      generated.CheckpointFingerprint,
		CheckpointBytes:            generated.CheckpointBytes,
		ConfiguredMemoryLimitBytes: config.ExpectedMemoryLimitBytes,
		MinimumGeneratedTokens:     config.MinimumGeneratedTokens,
		Producer:                   producerEvidence, Consumer: consumerEvidence,
		Reference: reference, Generation: generated, Checks: checks,
	}
	if !checks.AllPassed {
		return result, errors.New("pooled-memory proof did not satisfy every check")
	}
	return result, nil
}

type proofCleanupTarget struct {
	Name          string
	Caller        workerproc.PersistentCaller
	ShardIDPrefix string
}

func cleanupProofShards(targets ...proofCleanupTarget) error {
	var cleanupErr error
	for _, target := range targets {
		ctx, cancel := context.WithTimeout(context.Background(), defaultCleanupTimeout)
		state, err := workerproc.State(ctx, target.Caller)
		cancel()
		if err != nil {
			cleanupErr = errors.Join(cleanupErr, fmt.Errorf("%s state: %w", target.Name, err))
			continue
		}
		for _, shard := range state.LoadedShards {
			if !strings.HasPrefix(shard.ShardID, target.ShardIDPrefix) {
				cleanupErr = errors.Join(cleanupErr, fmt.Errorf(
					"%s has unexpected shard %q after proof", target.Name, shard.ShardID,
				))
				continue
			}
			ctx, cancel := context.WithTimeout(context.Background(), defaultCleanupTimeout)
			_, err := target.Caller.Call(ctx, workerproc.PersistentRequest{
				Command: "unloadShard",
				Shard:   &workerproc.PersistentShardRequest{ShardID: shard.ShardID},
			})
			cancel()
			if err != nil {
				cleanupErr = errors.Join(cleanupErr, fmt.Errorf(
					"%s unload shard %q: %w", target.Name, shard.ShardID, err,
				))
			}
		}
	}
	return cleanupErr
}

func remoteCapabilities(
	ctx context.Context,
	client *http.Client,
	endpoint string,
) (Capabilities, error) {
	request, err := http.NewRequestWithContext(
		ctx, http.MethodGet,
		strings.TrimRight(endpoint, "/")+"/v1/debug/worker/capabilities",
		nil,
	)
	if err != nil {
		return Capabilities{}, err
	}
	response, err := client.Do(request)
	if err != nil {
		return Capabilities{}, err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return Capabilities{}, fmt.Errorf("capabilities returned HTTP %d", response.StatusCode)
	}
	return decodeCapabilities(io.LimitReader(response.Body, defaultCapabilitiesMaxBody))
}

func localCapabilities(ctx context.Context, workerPath string) (Capabilities, error) {
	result, err := (workerproc.Client{Path: workerPath}).Run(ctx, []string{"capabilities"}, nil)
	if err != nil {
		return Capabilities{}, err
	}
	return decodeCapabilities(strings.NewReader(string(result.Output)))
}

func decodeCapabilities(reader io.Reader) (Capabilities, error) {
	var capabilities Capabilities
	if err := json.NewDecoder(reader).Decode(&capabilities); err != nil {
		return Capabilities{}, err
	}
	if capabilities.Runtime == "" || capabilities.Device == "" ||
		capabilities.PhysicalMemoryBytes == 0 || capabilities.MLXMemoryLimitBytes <= 0 ||
		capabilities.MLXCacheLimitBytes <= 0 {
		return Capabilities{}, fmt.Errorf("incomplete worker capabilities: %+v", capabilities)
	}
	return capabilities, nil
}

func observePhase(evidence *MemoryEvidence, operation string, memory workerproc.StageMemory) {
	switch operation {
	case "prefill":
		evidence.Prefill = maximumMemory(evidence.Prefill, memory)
	case "decode":
		evidence.Decode = maximumMemory(evidence.Decode, memory)
	}
	updateMaxObserved(evidence, memory)
}

func updateMaxObserved(evidence *MemoryEvidence, memory workerproc.StageMemory) {
	activeAndCache := saturatedAdd(memory.ActiveBytes, memory.CacheBytes)
	peakAndCache := saturatedAdd(memory.PeakBytes, memory.CacheBytes)
	evidence.MaxObservedBytes = max(evidence.MaxObservedBytes, activeAndCache, peakAndCache)
}

func maximumMemory(left, right workerproc.StageMemory) workerproc.StageMemory {
	return workerproc.StageMemory{
		ActiveBytes: max(left.ActiveBytes, right.ActiveBytes),
		CacheBytes:  max(left.CacheBytes, right.CacheBytes),
		PeakBytes:   max(left.PeakBytes, right.PeakBytes),
	}
}

func completeMemoryEvidence(evidence MemoryEvidence) bool {
	return validMemory(evidence.Load) && validMemory(evidence.Prefill) &&
		validMemory(evidence.Decode) && evidence.MaxObservedBytes > 0 &&
		evidence.MaxObservedBytes >= saturatedAdd(evidence.Load.PeakBytes, evidence.Load.CacheBytes) &&
		evidence.MaxObservedBytes >= saturatedAdd(evidence.Prefill.PeakBytes, evidence.Prefill.CacheBytes) &&
		evidence.MaxObservedBytes >= saturatedAdd(evidence.Decode.PeakBytes, evidence.Decode.CacheBytes)
}

func validMemory(memory workerproc.StageMemory) bool {
	return memory.ActiveBytes > 0 && memory.CacheBytes >= 0 &&
		memory.PeakBytes >= memory.ActiveBytes
}

func saturatedAdd(left, right int) int {
	maximum := int(^uint(0) >> 1)
	if left < 0 || right < 0 || left > maximum-right {
		return maximum
	}
	return left + right
}

func cleanWorker(state *workerproc.PersistentWorkerState) bool {
	return state != nil && len(state.LoadedShards) == 0 && state.KVCacheBytes == 0 && state.RetainedBytes == 0
}

func releasedWorker(state *workerproc.PersistentWorkerState) bool {
	if state == nil || state.KVCacheBytes != 0 || state.RetainedBytes != 0 || len(state.LoadedShards) != 1 {
		return false
	}
	return state.LoadedShards[0].OpenSequenceCount == 0 &&
		state.LoadedShards[0].KVCacheBytes == 0 && state.LoadedShards[0].RetainedBytes == 0
}

func complementaryShards(
	producer, consumer *workerproc.PersistentWorkerState,
	plan generation.ShardPlan,
	layerCount int,
) bool {
	if producer == nil || consumer == nil || len(producer.LoadedShards) != 1 || len(consumer.LoadedShards) != 1 {
		return false
	}
	first, second := producer.LoadedShards[0], consumer.LoadedShards[0]
	return first.ShardID == plan.Producer.ID && second.ShardID == plan.Consumer.ID &&
		first.LayerStart == 0 && first.LayerEnd == plan.Producer.LayerEnd &&
		second.LayerStart == plan.Consumer.LayerStart && second.LayerEnd == layerCount &&
		first.LayerEnd == second.LayerStart && first.OwnsInput && !first.OwnsOutput &&
		!second.OwnsInput && second.OwnsOutput
}

func configuredLimit(
	capabilities Capabilities,
	state *workerproc.PersistentWorkerState,
	expected int,
) bool {
	return capabilities.MLXMemoryLimitBytes == expected &&
		uint64(expected) <= capabilities.PhysicalMemoryBytes && state != nil &&
		state.MLXMemoryLimitBytes == expected &&
		state.PhysicalMemoryBytes == capabilities.PhysicalMemoryBytes &&
		state.MLXCacheLimitBytes == capabilities.MLXCacheLimitBytes
}

func allChecksPassed(checks Checks) bool {
	return checks.CleanWorkersAtStart && checks.CheckpointMatchesReference &&
		checks.CheckpointExceedsProducerLimit && checks.CheckpointExceedsConsumerLimit &&
		checks.FullInferenceExceedsProducerLimit && checks.FullInferenceExceedsConsumerLimit &&
		checks.ProducerUsesConfiguredLimit && checks.ConsumerUsesConfiguredLimit &&
		checks.ProducerWithinLimit && checks.ConsumerWithinLimit &&
		checks.ComplementaryShardsOnly && checks.NoServingFullModelOracle &&
		checks.PromptTokensMatchReference && checks.GeneratedTokensMatchReference &&
		checks.GeneratedAtLeastMinimumTokens && checks.SequenceStateReleased
}

package generation

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/tensorcheck"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

type SessionConfig struct {
	Model            string
	RTol             float64
	ATol             float64
	ForwardTimeout   time.Duration
	TerminalSampling bool
	Observer         StageObserver
	LogitsObserver   LogitsObserver
}

const DefaultForwardTimeout = 30 * time.Second

// StageObserver receives one synchronous observation for each prefill or
// decode operation after the resulting token has been sampled. Observer work
// is excluded from the reported stage durations.
type StageObserver func(StageSample)

// LogitsObserver receives a private copy of the distributed final-logit
// tensor immediately before each greedy token is sampled. It is intended for
// offline reference verification where the distributed shards and full-model
// oracle cannot coexist in memory.
type LogitsObserver func(step int, logits workerproc.WireTensor)

// StageSample separates the distributed hot path from the cached full-model
// correctness oracle. Wire byte counts measure representative JSON payloads,
// excluding transport headers and generated request IDs.
type StageSample struct {
	Operation                           string                 `json:"operation"`
	Position                            uint64                 `json:"position"`
	InputTokenCount                     int                    `json:"inputTokenCount"`
	ProducerWallMicros                  int64                  `json:"producerWallMicros"`
	ProducerComputeMicros               uint64                 `json:"producerComputeMicros"`
	ProducerOverheadMicros              int64                  `json:"producerOverheadMicros"`
	BoundarySerializationMicros         int64                  `json:"boundarySerializationMicros"`
	ConsumerResponseSerializationMicros int64                  `json:"consumerResponseSerializationMicros"`
	ConsumerRoundTripMicros             int64                  `json:"consumerRoundTripMicros"`
	ConsumerComputeMicros               uint64                 `json:"consumerComputeMicros"`
	TransportOverheadMicros             int64                  `json:"transportOverheadMicros"`
	DistributedEndToEndMicros           int64                  `json:"distributedEndToEndMicros"`
	SamplingMicros                      int64                  `json:"samplingMicros"`
	TokenLatencyMicros                  int64                  `json:"tokenLatencyMicros"`
	ReferenceWallMicros                 int64                  `json:"referenceWallMicros,omitempty"`
	ReferenceComputeMicros              uint64                 `json:"referenceComputeMicros,omitempty"`
	ReferenceSamplingMicros             int64                  `json:"referenceSamplingMicros,omitempty"`
	ReferenceTokenLatencyMicros         int64                  `json:"referenceTokenLatencyMicros,omitempty"`
	BoundaryTensorBytes                 int                    `json:"boundaryTensorBytes"`
	BoundaryWireBytes                   int                    `json:"boundaryWireBytes"`
	ConsumerResponseTensorBytes         int                    `json:"consumerResponseTensorBytes"`
	ConsumerResponseWireBytes           int                    `json:"consumerResponseWireBytes"`
	TerminalSampling                    bool                   `json:"terminalSampling"`
	ProducerKVCacheBytes                int                    `json:"producerKVCacheBytes"`
	ConsumerKVCacheBytes                int                    `json:"consumerKVCacheBytes"`
	ReferenceKVCacheBytes               int                    `json:"referenceKVCacheBytes,omitempty"`
	ProducerMemory                      workerproc.StageMemory `json:"producerMemory"`
	ConsumerMemory                      workerproc.StageMemory `json:"consumerMemory"`
	ReferenceMemory                     workerproc.StageMemory `json:"referenceMemory,omitempty"`
}

type Request struct {
	Prompt     string
	MaxTokens  int
	SequenceID string
	IgnoreEOS  bool
}

type Shard struct {
	ID         string `json:"id"`
	LayerStart int    `json:"layerStart"`
	LayerEnd   int    `json:"layerEnd"`
}

type ShardPlan struct {
	Producer Shard `json:"producer"`
	Consumer Shard `json:"consumer"`
}

type Timing struct {
	SessionSetupMicros     int64  `json:"sessionSetupMicros"`
	TokenizeMicros         int64  `json:"tokenizeMicros"`
	PrefillMicros          int64  `json:"prefillMicros"`
	DecodeMicros           int64  `json:"decodeMicros"`
	DetokenizeMicros       int64  `json:"detokenizeMicros"`
	TotalMicros            int64  `json:"totalMicros"`
	ProducerComputeMicros  uint64 `json:"producerComputeMicros"`
	ConsumerComputeMicros  uint64 `json:"consumerComputeMicros"`
	ReferenceComputeMicros uint64 `json:"referenceComputeMicros,omitempty"`
}

type Verification struct {
	GreedyTokenIDsMatch   bool    `json:"greedyTokenIDsMatch"`
	ComparedTokens        int     `json:"comparedTokens"`
	MaxAbsoluteDifference float64 `json:"maxAbsoluteDifference"`
	MaxRelativeDifference float64 `json:"maxRelativeDifference"`
}

type Result struct {
	Model                 string        `json:"model"`
	ModelType             string        `json:"modelType"`
	CheckpointFingerprint string        `json:"checkpointFingerprint"`
	CheckpointBytes       uint64        `json:"checkpointBytes"`
	ShardPlan             ShardPlan     `json:"shardPlan"`
	SequenceID            string        `json:"sequenceID"`
	Prompt                string        `json:"prompt"`
	PromptTokenIDs        []int32       `json:"promptTokenIDs"`
	GeneratedTokenIDs     []int32       `json:"generatedTokenIDs"`
	Text                  string        `json:"text"`
	MaxTokens             int           `json:"maxTokens"`
	StopReason            string        `json:"stopReason"`
	EOSTokenID            *int32        `json:"eosTokenID,omitempty"`
	RTol                  float64       `json:"rtol"`
	ATol                  float64       `json:"atol"`
	ForwardTimeoutMillis  int64         `json:"forwardTimeoutMillis"`
	ProducerKVCacheBytes  int           `json:"producerKVCacheBytes"`
	ConsumerKVCacheBytes  int           `json:"consumerKVCacheBytes"`
	Timing                Timing        `json:"timing"`
	Verification          *Verification `json:"verification,omitempty"`
	Failure               *Failure      `json:"failure,omitempty"`
}

type Failure struct {
	SequenceID             string `json:"sequenceID"`
	ShardID                string `json:"shardID,omitempty"`
	Phase                  string `json:"phase"`
	Operation              string `json:"operation,omitempty"`
	Position               uint64 `json:"position"`
	LastAcceptedTokenIndex int    `json:"lastAcceptedTokenIndex"`
	LastAcceptedTokenID    *int32 `json:"lastAcceptedTokenID,omitempty"`
	TimedOut               bool   `json:"timedOut"`
	Canceled               bool   `json:"canceled"`
	Cause                  string `json:"cause"`
}

type GenerationError struct {
	Failure Failure
	Err     error
}

func (err *GenerationError) Error() string {
	return fmt.Sprintf("generation %s failed: %v", err.Failure.Phase, err.Err)
}

func (err *GenerationError) Unwrap() error { return err.Err }

type failurePoint struct {
	phase     string
	shardID   string
	operation string
	position  uint64
}

type Session struct {
	producer       workerproc.PersistentCaller
	consumer       workerproc.PersistentCaller
	reference      workerproc.PersistentCaller
	config         SessionConfig
	model          workerproc.PersistentModelResult
	plan           ShardPlan
	referenceShard string
	setupMicros    int64
}

type SessionInfo struct {
	Model              workerproc.PersistentModelResult `json:"model"`
	ShardPlan          ShardPlan                        `json:"shardPlan"`
	ReferenceShardID   string                           `json:"referenceShardID,omitempty"`
	SessionSetupMicros int64                            `json:"sessionSetupMicros"`
}

func (s *Session) Info() SessionInfo {
	return SessionInfo{
		Model: s.model, ShardPlan: s.plan, ReferenceShardID: s.referenceShard,
		SessionSetupMicros: s.setupMicros,
	}
}

func NewSession(
	ctx context.Context,
	producer workerproc.PersistentCaller,
	consumer workerproc.PersistentCaller,
	reference workerproc.PersistentCaller,
	config SessionConfig,
) (*Session, error) {
	started := time.Now()
	if producer == nil || consumer == nil {
		return nil, errors.New("producer and consumer callers are required")
	}
	if config.Model == "" {
		return nil, errors.New("model is required")
	}
	if config.RTol < 0 || config.ATol < 0 ||
		math.IsNaN(config.RTol) || math.IsNaN(config.ATol) ||
		math.IsInf(config.RTol, 0) || math.IsInf(config.ATol, 0) {
		return nil, errors.New("numeric tolerances must be finite and non-negative")
	}
	if config.ForwardTimeout < 0 {
		return nil, errors.New("forward timeout must be non-negative")
	}
	if config.ForwardTimeout > 0 && config.ForwardTimeout < time.Millisecond {
		return nil, errors.New("forward timeout must be at least 1ms")
	}
	if config.ForwardTimeout == 0 {
		config.ForwardTimeout = DefaultForwardTimeout
	}
	if config.TerminalSampling && (reference != nil || config.LogitsObserver != nil) {
		return nil, errors.New("terminal sampling cannot return logits for reference verification")
	}

	producerModel, err := modelInfo(ctx, producer, config.Model)
	if err != nil {
		return nil, fmt.Errorf("producer model info: %w", err)
	}
	consumerModel, err := modelInfo(ctx, consumer, config.Model)
	if err != nil {
		return nil, fmt.Errorf("consumer model info: %w", err)
	}
	if err := matchModel(producerModel, consumerModel, "consumer"); err != nil {
		return nil, err
	}
	if producerModel.LayerCount < 2 {
		return nil, fmt.Errorf("model %s has %d layers; at least two are required", config.Model, producerModel.LayerCount)
	}

	split := producerModel.LayerCount / 2
	suffix := modelHashSuffix(config.Model, producerModel.CheckpointFingerprint)
	plan := ShardPlan{
		Producer: Shard{ID: "generate-producer-" + suffix, LayerStart: 0, LayerEnd: split},
		Consumer: Shard{ID: "generate-consumer-" + suffix, LayerStart: split, LayerEnd: producerModel.LayerCount},
	}
	if _, err := ensureShard(ctx, producer, workerproc.PersistentLoadShardRequest{
		ModelID: config.Model, ShardID: plan.Producer.ID,
		CheckpointFingerprint: producerModel.CheckpointFingerprint,
		LayerStart:            plan.Producer.LayerStart, LayerEnd: plan.Producer.LayerEnd, OwnsInput: true,
	}); err != nil {
		return nil, fmt.Errorf("producer shard: %w", err)
	}
	if _, err := ensureShard(ctx, consumer, workerproc.PersistentLoadShardRequest{
		ModelID: config.Model, ShardID: plan.Consumer.ID,
		CheckpointFingerprint: producerModel.CheckpointFingerprint,
		LayerStart:            plan.Consumer.LayerStart, LayerEnd: plan.Consumer.LayerEnd, OwnsOutput: true,
	}); err != nil {
		return nil, fmt.Errorf("consumer shard: %w", err)
	}

	referenceShard := ""
	if reference != nil {
		referenceModel, infoErr := modelInfo(ctx, reference, config.Model)
		if infoErr != nil {
			return nil, fmt.Errorf("reference model info: %w", infoErr)
		}
		if err := matchModel(producerModel, referenceModel, "reference"); err != nil {
			return nil, err
		}
		referenceShard = "generate-reference-" + suffix
		if _, err := ensureShard(ctx, reference, workerproc.PersistentLoadShardRequest{
			ModelID: config.Model, ShardID: referenceShard,
			CheckpointFingerprint: producerModel.CheckpointFingerprint,
			LayerStart:            0, LayerEnd: producerModel.LayerCount, OwnsInput: true, OwnsOutput: true,
		}); err != nil {
			return nil, fmt.Errorf("reference shard: %w", err)
		}
	}

	return &Session{
		producer: producer, consumer: consumer, reference: reference,
		config: config, model: *producerModel, plan: plan, referenceShard: referenceShard,
		setupMicros: time.Since(started).Microseconds(),
	}, nil
}

func (s *Session) Generate(ctx context.Context, request Request) (result Result, returnErr error) {
	started := time.Now()
	defer func() { result.Timing.TotalMicros = time.Since(started).Microseconds() }()
	if err := ctx.Err(); err != nil {
		return Result{}, err
	}
	if request.Prompt == "" {
		return Result{}, errors.New("prompt is required")
	}
	if request.MaxTokens <= 0 {
		return Result{}, errors.New("max tokens must be positive")
	}
	if request.SequenceID == "" {
		sequenceID, err := randomSequenceID()
		if err != nil {
			return Result{}, err
		}
		request.SequenceID = sequenceID
	}
	result = Result{
		Model: s.config.Model, ModelType: s.model.ModelType,
		CheckpointFingerprint: s.model.CheckpointFingerprint,
		CheckpointBytes:       s.model.CheckpointBytes, ShardPlan: s.plan,
		SequenceID: request.SequenceID, Prompt: request.Prompt, MaxTokens: request.MaxTokens,
		RTol: s.config.RTol, ATol: s.config.ATol,
		ForwardTimeoutMillis: s.config.ForwardTimeout.Milliseconds(),
		Timing:               Timing{SessionSetupMicros: s.setupMicros},
	}
	point := failurePoint{phase: "tokenize", shardID: s.plan.Producer.ID}
	defer func() {
		if returnErr == nil || result.Failure != nil {
			return
		}
		failure := failureFrom(point, result, returnErr)
		result.Failure = &failure
		returnErr = &GenerationError{Failure: failure, Err: returnErr}
	}()

	tokenizeStarted := time.Now()
	tokenized, err := tokenize(ctx, s.producer, s.config.Model, request.Prompt)
	result.Timing.TokenizeMicros = time.Since(tokenizeStarted).Microseconds()
	if err != nil {
		return result, fmt.Errorf("tokenize prompt: %w", err)
	}
	if len(tokenized.TokenIDs) == 0 {
		return result, errors.New("tokenizer returned no prompt tokens")
	}
	result.PromptTokenIDs = tokenized.TokenIDs
	result.EOSTokenID = tokenized.EOSTokenID

	targets := []workerproc.SequenceTarget{
		{Name: "producer", Caller: s.producer, ShardID: s.plan.Producer.ID},
		{Name: "consumer", Caller: s.consumer, ShardID: s.plan.Consumer.ID},
	}
	if s.reference != nil {
		targets = append(targets, workerproc.SequenceTarget{
			Name: "reference", Caller: s.reference, ShardID: s.referenceShard,
		})
	}
	point = failurePoint{phase: "open_sequences"}
	sequences, err := workerproc.OpenSequences(ctx, targets, request.SequenceID)
	if sequences != nil {
		defer func() {
			cleanupErr := sequences.Cleanup()
			if cleanupErr != nil {
				if returnErr == nil {
					point = failurePoint{phase: "sequence_cleanup"}
				}
				if returnErr == nil {
					returnErr = cleanupErr
				} else {
					returnErr = errors.Join(returnErr, cleanupErr)
				}
			}
		}()
	}
	if err != nil {
		return result, fmt.Errorf("open generation sequences: %w", err)
	}

	prompt := tokenTensor(tokenized.TokenIDs)
	prefillStarted := time.Now()
	distributedStarted := time.Now()
	point = failurePoint{
		phase: "producer_prefill", shardID: s.plan.Producer.ID,
		operation: "prefill", position: 0,
	}
	producerResult, producerWallMicros, err := measuredInfer(
		ctx, s.config.ForwardTimeout, s.producer,
		"prefill", s.plan.Producer.ID, request.SequenceID, 0, "tokens", prompt, false,
	)
	if err != nil {
		return result, fmt.Errorf("producer prefill: %w", err)
	}
	point = failurePoint{
		phase: "consumer_prefill", shardID: s.plan.Consumer.ID,
		operation: "prefill", position: 0,
	}
	consumerResult, consumerWallMicros, err := measuredInfer(
		ctx, s.config.ForwardTimeout, s.consumer,
		"prefill", s.plan.Consumer.ID, request.SequenceID, 0, "hidden", producerResult.Output,
		s.config.TerminalSampling,
	)
	if err != nil {
		return result, fmt.Errorf("consumer prefill: %w", err)
	}
	distributedEndToEndMicros := time.Since(distributedStarted).Microseconds()
	var referenceResult *workerproc.PersistentForwardResult
	var referenceWallMicros int64
	if s.reference != nil {
		point = failurePoint{
			phase: "reference_prefill", shardID: s.referenceShard,
			operation: "prefill", position: 0,
		}
		referenceResult, referenceWallMicros, err = measuredInfer(
			ctx, s.config.ForwardTimeout, s.reference,
			"prefill", s.referenceShard, request.SequenceID, 0, "tokens", prompt, false,
		)
		if err != nil {
			return result, fmt.Errorf("reference prefill: %w", err)
		}
	}
	result.Timing.PrefillMicros = time.Since(prefillStarted).Microseconds()
	result.Timing.ProducerComputeMicros += producerResult.ComputeMicros
	result.Timing.ConsumerComputeMicros += consumerResult.ComputeMicros
	result.ProducerKVCacheBytes = producerResult.KVCacheBytes
	result.ConsumerKVCacheBytes = consumerResult.KVCacheBytes
	if referenceResult != nil {
		result.Timing.ReferenceComputeMicros += referenceResult.ComputeMicros
		result.Verification = &Verification{GreedyTokenIDsMatch: true}
	}
	var pendingSample *StageSample
	if s.config.Observer != nil {
		sample := newStageSample(
			"prefill", 0, len(tokenized.TokenIDs), request.SequenceID,
			producerResult, producerWallMicros, consumerResult, consumerWallMicros,
			distributedEndToEndMicros, referenceResult, referenceWallMicros,
		)
		pendingSample = &sample
	}

	position := uint64(len(tokenized.TokenIDs))
	distributedLogits := consumerResult.Output
	distributedSampledToken := consumerResult.SampledTokenID
	var referenceLogits workerproc.WireTensor
	if referenceResult != nil {
		referenceLogits = referenceResult.Output
	}
	decodeStarted := time.Now()
	for len(result.GeneratedTokenIDs) < request.MaxTokens {
		point = failurePoint{phase: "sample", operation: "sample", position: position}
		if err := ctx.Err(); err != nil {
			return result, err
		}
		if s.config.LogitsObserver != nil {
			s.config.LogitsObserver(
				len(result.GeneratedTokenIDs),
				cloneWireTensor(distributedLogits),
			)
		}
		samplingStarted := time.Now()
		nextToken, err := sampledToken(distributedSampledToken, distributedLogits)
		samplingMicros := time.Since(samplingStarted).Microseconds()
		if err != nil {
			return result, fmt.Errorf("sample distributed logits: %w", err)
		}
		if pendingSample != nil {
			pendingSample.SamplingMicros = samplingMicros
			pendingSample.TokenLatencyMicros = pendingSample.DistributedEndToEndMicros + samplingMicros
		}
		if result.Verification != nil {
			difference, compareErr := tensorcheck.CompareFinalLogits(
				distributedLogits, referenceLogits, s.config.RTol, s.config.ATol,
			)
			if compareErr != nil {
				return result, fmt.Errorf("generation step %d logits: %w", len(result.GeneratedTokenIDs), compareErr)
			}
			result.Verification.MaxAbsoluteDifference = math.Max(
				result.Verification.MaxAbsoluteDifference, difference.Absolute,
			)
			result.Verification.MaxRelativeDifference = math.Max(
				result.Verification.MaxRelativeDifference, difference.Relative,
			)
			referenceSamplingStarted := time.Now()
			referenceToken, sampleErr := greedyToken(referenceLogits)
			referenceSamplingMicros := time.Since(referenceSamplingStarted).Microseconds()
			if sampleErr != nil {
				return result, fmt.Errorf("sample reference logits: %w", sampleErr)
			}
			if pendingSample != nil {
				pendingSample.ReferenceSamplingMicros = referenceSamplingMicros
				pendingSample.ReferenceTokenLatencyMicros =
					pendingSample.ReferenceWallMicros + referenceSamplingMicros
			}
			if nextToken != referenceToken {
				result.Verification.GreedyTokenIDsMatch = false
				return result, fmt.Errorf(
					"generation step %d chose token %d; reference chose %d",
					len(result.GeneratedTokenIDs), nextToken, referenceToken,
				)
			}
			result.Verification.ComparedTokens++
		}
		if pendingSample != nil {
			s.config.Observer(*pendingSample)
			pendingSample = nil
		}
		result.GeneratedTokenIDs = append(result.GeneratedTokenIDs, nextToken)
		if !request.IgnoreEOS && result.EOSTokenID != nil && nextToken == *result.EOSTokenID {
			result.StopReason = "eos"
			break
		}
		if len(result.GeneratedTokenIDs) == request.MaxTokens {
			result.StopReason = "max_tokens"
			break
		}

		token := tokenTensor([]int32{nextToken})
		distributedStarted = time.Now()
		point = failurePoint{
			phase: "producer_decode", shardID: s.plan.Producer.ID,
			operation: "decode", position: position,
		}
		producerResult, producerWallMicros, err = measuredInfer(
			ctx, s.config.ForwardTimeout, s.producer,
			"decode", s.plan.Producer.ID, request.SequenceID, position, "tokens", token, false,
		)
		if err != nil {
			return result, fmt.Errorf("producer decode step %d: %w", len(result.GeneratedTokenIDs), err)
		}
		point = failurePoint{
			phase: "consumer_decode", shardID: s.plan.Consumer.ID,
			operation: "decode", position: position,
		}
		consumerResult, consumerWallMicros, err = measuredInfer(
			ctx, s.config.ForwardTimeout, s.consumer,
			"decode", s.plan.Consumer.ID, request.SequenceID, position, "hidden", producerResult.Output,
			s.config.TerminalSampling,
		)
		if err != nil {
			return result, fmt.Errorf("consumer decode step %d: %w", len(result.GeneratedTokenIDs), err)
		}
		distributedEndToEndMicros = time.Since(distributedStarted).Microseconds()
		result.Timing.ProducerComputeMicros += producerResult.ComputeMicros
		result.Timing.ConsumerComputeMicros += consumerResult.ComputeMicros
		result.ProducerKVCacheBytes = producerResult.KVCacheBytes
		result.ConsumerKVCacheBytes = consumerResult.KVCacheBytes
		distributedLogits = consumerResult.Output
		distributedSampledToken = consumerResult.SampledTokenID
		referenceResult = nil
		referenceWallMicros = 0
		if s.reference != nil {
			point = failurePoint{
				phase: "reference_decode", shardID: s.referenceShard,
				operation: "decode", position: position,
			}
			referenceResult, referenceWallMicros, err = measuredInfer(
				ctx, s.config.ForwardTimeout, s.reference,
				"decode", s.referenceShard, request.SequenceID, position, "tokens", token, false,
			)
			if err != nil {
				return result, fmt.Errorf("reference decode step %d: %w", len(result.GeneratedTokenIDs), err)
			}
			result.Timing.ReferenceComputeMicros += referenceResult.ComputeMicros
			referenceLogits = referenceResult.Output
		}
		if s.config.Observer != nil {
			sample := newStageSample(
				"decode", position, 1, request.SequenceID,
				producerResult, producerWallMicros, consumerResult, consumerWallMicros,
				distributedEndToEndMicros, referenceResult, referenceWallMicros,
			)
			pendingSample = &sample
		}
		position++
	}
	result.Timing.DecodeMicros = time.Since(decodeStarted).Microseconds()

	point = failurePoint{phase: "detokenize", shardID: s.plan.Producer.ID}
	detokenizeStarted := time.Now()
	text, err := detokenize(ctx, s.producer, s.config.Model, result.GeneratedTokenIDs)
	result.Timing.DetokenizeMicros = time.Since(detokenizeStarted).Microseconds()
	if err != nil {
		return result, fmt.Errorf("detokenize generated tokens: %w", err)
	}
	result.Text = text
	point = failurePoint{}
	return result, nil
}

func modelInfo(
	ctx context.Context,
	caller workerproc.PersistentCaller,
	modelID string,
) (*workerproc.PersistentModelResult, error) {
	response, err := call(ctx, caller, workerproc.PersistentRequest{
		Command: "modelInfo", Model: &workerproc.PersistentModelRequest{ModelID: modelID},
	})
	if err != nil {
		return nil, err
	}
	if response.Result == nil || response.Result.Model == nil {
		return nil, errors.New("modelInfo returned no model metadata")
	}
	if response.Result.Model.ModelID != modelID || response.Result.Model.ModelType == "" ||
		response.Result.Model.LayerCount <= 0 || response.Result.Model.CheckpointFingerprint == "" ||
		response.Result.Model.CheckpointBytes == 0 {
		return nil, fmt.Errorf("modelInfo returned invalid metadata: %+v", *response.Result.Model)
	}
	return response.Result.Model, nil
}

func matchModel(expected, actual *workerproc.PersistentModelResult, role string) error {
	if expected.ModelID != actual.ModelID || expected.ModelType != actual.ModelType ||
		expected.LayerCount != actual.LayerCount ||
		expected.CheckpointFingerprint != actual.CheckpointFingerprint ||
		expected.CheckpointBytes != actual.CheckpointBytes {
		return fmt.Errorf(
			"%s model mismatch: producer=%+v %s=%+v", role, *expected, role, *actual,
		)
	}
	return nil
}

func ensureShard(
	ctx context.Context,
	caller workerproc.PersistentCaller,
	request workerproc.PersistentLoadShardRequest,
) (*workerproc.PersistentShardSnapshot, error) {
	state, err := workerState(ctx, caller)
	if err != nil {
		return nil, err
	}
	if shard, found, err := findLoadedShard(state, request); found || err != nil {
		return shard, err
	}
	response, err := call(ctx, caller, workerproc.PersistentRequest{Command: "loadShard", LoadShard: &request})
	if err != nil {
		var workerResponseErr *workerproc.WorkerResponseError
		if errors.As(err, &workerResponseErr) {
			// A stable shard may have been loaded after the state snapshot by a
			// concurrent session. Reconcile any explicit worker rejection with
			// current state before failing this session.
			if refreshed, stateErr := workerState(ctx, caller); stateErr == nil {
				if shard, found, validationErr := findLoadedShard(refreshed, request); found || validationErr != nil {
					return shard, validationErr
				}
			}
		}
		return nil, err
	}
	if response.Result == nil || response.Result.Shard == nil {
		return nil, errors.New("loadShard returned no shard snapshot")
	}
	if err := validateLoadedShard(response.Result.Shard, request); err != nil {
		return nil, err
	}
	return response.Result.Shard, nil
}

func findLoadedShard(
	state *workerproc.PersistentWorkerState,
	request workerproc.PersistentLoadShardRequest,
) (*workerproc.PersistentShardSnapshot, bool, error) {
	for index := range state.LoadedShards {
		shard := &state.LoadedShards[index]
		if shard.ShardID != request.ShardID {
			continue
		}
		if err := validateLoadedShard(shard, request); err != nil {
			return nil, true, err
		}
		return shard, true, nil
	}
	return nil, false, nil
}

func validateLoadedShard(
	shard *workerproc.PersistentShardSnapshot,
	request workerproc.PersistentLoadShardRequest,
) error {
	if shard.ModelID != request.ModelID ||
		shard.CheckpointFingerprint != request.CheckpointFingerprint ||
		shard.LayerStart != request.LayerStart || shard.LayerEnd != request.LayerEnd ||
		shard.OwnsInput != request.OwnsInput || shard.OwnsOutput != request.OwnsOutput {
		return fmt.Errorf("loaded shard %s does not match requested plan", request.ShardID)
	}
	return nil
}

func tokenize(
	ctx context.Context,
	caller workerproc.PersistentCaller,
	modelID string,
	prompt string,
) (*workerproc.PersistentTextResult, error) {
	addSpecialTokens := true
	response, err := call(ctx, caller, workerproc.PersistentRequest{
		Command: "tokenize",
		Text: &workerproc.PersistentTextRequest{
			ModelID: modelID, Text: &prompt, AddSpecialTokens: &addSpecialTokens,
		},
	})
	if err != nil {
		return nil, err
	}
	if response.Result == nil || response.Result.Text == nil {
		return nil, errors.New("tokenize returned no tokenization result")
	}
	if response.Result.Text.ModelID != modelID {
		return nil, fmt.Errorf(
			"tokenize returned model %q, want %q", response.Result.Text.ModelID, modelID,
		)
	}
	for _, token := range response.Result.Text.TokenIDs {
		if token < 0 {
			return nil, fmt.Errorf("tokenize returned negative token ID %d", token)
		}
	}
	return response.Result.Text, nil
}

func detokenize(
	ctx context.Context,
	caller workerproc.PersistentCaller,
	modelID string,
	tokens []int32,
) (string, error) {
	skipSpecialTokens := true
	response, err := call(ctx, caller, workerproc.PersistentRequest{
		Command: "detokenize",
		Text: &workerproc.PersistentTextRequest{
			ModelID: modelID, TokenIDs: tokens, SkipSpecialTokens: &skipSpecialTokens,
		},
	})
	if err != nil {
		return "", err
	}
	if response.Result == nil || response.Result.Text == nil || response.Result.Text.Text == nil {
		return "", errors.New("detokenize returned no text")
	}
	if response.Result.Text.ModelID != modelID {
		return "", fmt.Errorf(
			"detokenize returned model %q, want %q", response.Result.Text.ModelID, modelID,
		)
	}
	return *response.Result.Text.Text, nil
}

func workerState(
	ctx context.Context,
	caller workerproc.PersistentCaller,
) (*workerproc.PersistentWorkerState, error) {
	return workerproc.State(ctx, caller)
}

func measuredInfer(
	ctx context.Context,
	timeout time.Duration,
	caller workerproc.PersistentCaller,
	command string,
	shardID string,
	sequenceID string,
	position uint64,
	inputKind string,
	input workerproc.WireTensor,
	returnSampledToken bool,
) (*workerproc.PersistentForwardResult, int64, error) {
	callContext, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	started := time.Now()
	result, err := infer(
		callContext, caller, command, shardID, sequenceID, position, inputKind, input,
		returnSampledToken,
	)
	return result, time.Since(started).Microseconds(), err
}

func newStageSample(
	operation string,
	position uint64,
	inputTokenCount int,
	sequenceID string,
	producer *workerproc.PersistentForwardResult,
	producerWallMicros int64,
	consumer *workerproc.PersistentForwardResult,
	consumerWallMicros int64,
	distributedEndToEndMicros int64,
	reference *workerproc.PersistentForwardResult,
	referenceWallMicros int64,
) StageSample {
	serializationStarted := time.Now()
	wirePayload, _ := json.Marshal(workerproc.PersistentRequest{
		Command: operation, DeadlineUnixMillis: time.Now().UnixMilli(),
		Forward: &workerproc.PersistentForwardRequest{
			ShardID: consumer.ShardID, SequenceID: sequenceID, Position: position,
			InputKind: "hidden", Input: producer.Output,
			ReturnSampledToken: consumer.SampledTokenID != nil,
		},
	})
	boundarySerializationMicros := time.Since(serializationStarted).Microseconds()
	responseSerializationStarted := time.Now()
	responsePayload, _ := json.Marshal(workerproc.PersistentResponse{
		OK: true,
		Result: &workerproc.PersistentWorkerResult{
			Forward: consumer,
		},
	})
	sample := StageSample{
		Operation: operation, Position: position, InputTokenCount: inputTokenCount,
		ProducerWallMicros: producerWallMicros, ProducerComputeMicros: producer.ComputeMicros,
		ProducerOverheadMicros:              positiveDifference(producerWallMicros, producer.ComputeMicros),
		BoundarySerializationMicros:         boundarySerializationMicros,
		ConsumerResponseSerializationMicros: time.Since(responseSerializationStarted).Microseconds(),
		ConsumerRoundTripMicros:             consumerWallMicros,
		ConsumerComputeMicros:               consumer.ComputeMicros,
		TransportOverheadMicros:             positiveDifference(consumerWallMicros, consumer.ComputeMicros),
		DistributedEndToEndMicros:           distributedEndToEndMicros,
		BoundaryTensorBytes:                 len(producer.Output.Data),
		BoundaryWireBytes:                   len(wirePayload),
		ConsumerResponseTensorBytes:         len(consumer.Output.Data),
		ConsumerResponseWireBytes:           len(responsePayload),
		TerminalSampling:                    consumer.SampledTokenID != nil,
		ProducerKVCacheBytes:                producer.KVCacheBytes,
		ConsumerKVCacheBytes:                consumer.KVCacheBytes,
		ProducerMemory:                      producer.Memory,
		ConsumerMemory:                      consumer.Memory,
	}
	if reference != nil {
		sample.ReferenceWallMicros = referenceWallMicros
		sample.ReferenceComputeMicros = reference.ComputeMicros
		sample.ReferenceKVCacheBytes = reference.KVCacheBytes
		sample.ReferenceMemory = reference.Memory
	}
	return sample
}

func positiveDifference(wallMicros int64, computeMicros uint64) int64 {
	if computeMicros >= uint64(wallMicros) {
		return 0
	}
	return wallMicros - int64(computeMicros)
}

func infer(
	ctx context.Context,
	caller workerproc.PersistentCaller,
	command string,
	shardID string,
	sequenceID string,
	position uint64,
	inputKind string,
	input workerproc.WireTensor,
	returnSampledToken bool,
) (*workerproc.PersistentForwardResult, error) {
	deadline, hasDeadline := ctx.Deadline()
	if !hasDeadline {
		return nil, workerproc.ErrInferenceDeadlineRequired
	}
	response, err := call(ctx, caller, workerproc.PersistentRequest{
		Command: command, DeadlineUnixMillis: workerproc.WireDeadlineUnixMillis(deadline),
		Forward: &workerproc.PersistentForwardRequest{
			ShardID: shardID, SequenceID: sequenceID, Position: position,
			InputKind: inputKind, Input: input, ReturnSampledToken: returnSampledToken,
		},
	})
	if err != nil {
		return nil, err
	}
	if response.Result == nil || response.Result.Forward == nil {
		return nil, fmt.Errorf("%s returned no inference result", command)
	}
	result := response.Result.Forward
	if result.Operation != command || result.Position != position || result.KVCacheBytes <= 0 {
		return nil, fmt.Errorf(
			"%s returned inconsistent metadata: operation=%s position=%d kv=%d",
			command, result.Operation, result.Position, result.KVCacheBytes,
		)
	}
	if returnSampledToken {
		if result.SampledTokenID == nil || len(result.Output.Data) != 0 {
			return nil, fmt.Errorf(
				"%s returned an invalid sampled-token response: token=%v outputBytes=%d",
				command, result.SampledTokenID, len(result.Output.Data),
			)
		}
	} else if result.SampledTokenID != nil || len(result.Output.Data) == 0 {
		return nil, fmt.Errorf(
			"%s returned an invalid tensor response: token=%v outputBytes=%d",
			command, result.SampledTokenID, len(result.Output.Data),
		)
	}
	return result, nil
}

func failureFrom(point failurePoint, result Result, err error) Failure {
	failure := Failure{
		SequenceID: result.SequenceID, ShardID: point.shardID,
		Phase: point.phase, Operation: point.operation, Position: point.position,
		LastAcceptedTokenIndex: len(result.GeneratedTokenIDs) - 1,
		TimedOut:               errors.Is(err, context.DeadlineExceeded),
		Canceled:               errors.Is(err, context.Canceled), Cause: err.Error(),
	}
	if len(result.GeneratedTokenIDs) > 0 {
		token := result.GeneratedTokenIDs[len(result.GeneratedTokenIDs)-1]
		failure.LastAcceptedTokenID = &token
	}
	return failure
}

func call(
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

func tokenTensor(tokens []int32) workerproc.WireTensor {
	data := make([]byte, len(tokens)*4)
	for index, token := range tokens {
		binary.LittleEndian.PutUint32(data[index*4:], uint32(token))
	}
	return workerproc.WireTensor{Shape: []int{1, len(tokens)}, DType: "int32", Data: data}
}

func cloneWireTensor(tensor workerproc.WireTensor) workerproc.WireTensor {
	return workerproc.WireTensor{
		Shape: append([]int(nil), tensor.Shape...),
		DType: tensor.DType,
		Data:  append([]byte(nil), tensor.Data...),
	}
}

func randomSequenceID() (string, error) {
	return randomIdentifier("generation-")
}

func randomIdentifier(prefix string) (string, error) {
	bytes := make([]byte, 12)
	if _, err := rand.Read(bytes); err != nil {
		return "", fmt.Errorf("generate sequence ID: %w", err)
	}
	return prefix + hex.EncodeToString(bytes), nil
}

func modelHashSuffix(modelID, fingerprint string) string {
	hash := sha256.Sum256([]byte(modelID + "\x00" + fingerprint))
	return hex.EncodeToString(hash[:6])
}

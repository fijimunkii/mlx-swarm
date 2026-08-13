package generation

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"time"

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
	planned        *PlannedSession
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
	if producer == nil || consumer == nil {
		return nil, errors.New("producer and consumer callers are required")
	}
	responseMode := StageResponseTensor
	if config.TerminalSampling {
		responseMode = StageResponseSampledToken
	}
	plannedConfig := PlannedSessionConfig{
		RTol: config.RTol, ATol: config.ATol,
		ForwardTimeout: config.ForwardTimeout,
		LogitsObserver: config.LogitsObserver,
	}
	if config.Observer != nil {
		plannedConfig.Observer = func(sample PlannedStageSample) {
			config.Observer(legacyStageSample(sample))
		}
	}
	targets := []ExecutionTarget{
		{TargetID: "producer", Caller: producer},
		{TargetID: "consumer", Caller: consumer},
	}
	planned, err := NewBalancedPlannedSession(
		ctx, config.Model, targets, reference, responseMode, plannedConfig,
	)
	if err != nil {
		var loadErr *ExecutionStageLoadError
		if errors.As(err, &loadErr) {
			role := loadErr.Stage.Name
			if loadErr.Index == 0 {
				role = "producer"
			} else if loadErr.Index == 1 {
				role = "consumer"
			}
			return nil, fmt.Errorf("%s shard: %w", role, loadErr.Err)
		}
		return nil, err
	}
	legacyPlan, ok := planned.plan.LegacyShardPlan()
	if !ok {
		return nil, errors.New("balanced two-stage session did not produce a legacy plan")
	}
	return &Session{
		planned: planned, model: planned.model, plan: legacyPlan,
		referenceShard: planned.referenceShard, setupMicros: planned.setupMicros,
	}, nil
}

func (s *Session) Generate(ctx context.Context, request Request) (Result, error) {
	if s == nil || s.planned == nil {
		return Result{}, errors.New("generation session is not initialized")
	}
	planned, err := s.planned.Generate(ctx, request)
	result := legacyResult(planned)
	var generationErr *GenerationError
	if errors.As(err, &generationErr) {
		failure := generationErr.Failure
		failure.Phase = legacyFailurePhase(failure.Phase)
		err = &GenerationError{Failure: failure, Err: generationErr.Err}
	}
	return result, err
}

func legacyResult(planned PlannedResult) Result {
	shardPlan, _ := planned.ExecutionPlan.LegacyShardPlan()
	result := Result{
		Model: planned.Model, ModelType: planned.ModelType,
		CheckpointFingerprint: planned.CheckpointFingerprint,
		CheckpointBytes:       planned.CheckpointBytes, ShardPlan: shardPlan,
		SequenceID: planned.SequenceID, Prompt: planned.Prompt,
		PromptTokenIDs:    planned.PromptTokenIDs,
		GeneratedTokenIDs: planned.GeneratedTokenIDs,
		Text:              planned.Text, MaxTokens: planned.MaxTokens,
		StopReason: planned.StopReason, EOSTokenID: planned.EOSTokenID,
		RTol: planned.RTol, ATol: planned.ATol,
		ForwardTimeoutMillis: planned.ForwardTimeoutMillis,
		Timing: Timing{
			SessionSetupMicros:     planned.Timing.SessionSetupMicros,
			TokenizeMicros:         planned.Timing.TokenizeMicros,
			PrefillMicros:          planned.Timing.PrefillMicros,
			DecodeMicros:           planned.Timing.DecodeMicros,
			DetokenizeMicros:       planned.Timing.DetokenizeMicros,
			TotalMicros:            planned.Timing.TotalMicros,
			ReferenceComputeMicros: planned.Timing.ReferenceComputeMicros,
		},
		Verification: planned.Verification,
	}
	if len(planned.StageKVCacheBytes) == 2 {
		result.ProducerKVCacheBytes = planned.StageKVCacheBytes[0]
		result.ConsumerKVCacheBytes = planned.StageKVCacheBytes[1]
	}
	if len(planned.Timing.StageComputeMicros) == 2 {
		result.Timing.ProducerComputeMicros = planned.Timing.StageComputeMicros[0]
		result.Timing.ConsumerComputeMicros = planned.Timing.StageComputeMicros[1]
	}
	if planned.Failure != nil {
		failure := *planned.Failure
		failure.Phase = legacyFailurePhase(failure.Phase)
		result.Failure = &failure
	}
	return result
}

func legacyStageSample(sample PlannedStageSample) StageSample {
	legacy := StageSample{
		Operation: sample.Operation, Position: sample.Position,
		InputTokenCount:             sample.InputTokenCount,
		DistributedEndToEndMicros:   sample.DistributedEndToEndMicros,
		SamplingMicros:              sample.SamplingMicros,
		TokenLatencyMicros:          sample.TokenLatencyMicros,
		ReferenceWallMicros:         sample.ReferenceWallMicros,
		ReferenceComputeMicros:      sample.ReferenceComputeMicros,
		ReferenceSamplingMicros:     sample.ReferenceSamplingMicros,
		ReferenceTokenLatencyMicros: sample.ReferenceTokenLatencyMicros,
		ReferenceKVCacheBytes:       sample.ReferenceKVCacheBytes,
		ReferenceMemory:             sample.ReferenceMemory,
	}
	if len(sample.Stages) != 2 {
		return legacy
	}
	producer := sample.Stages[0]
	consumer := sample.Stages[1]
	legacy.ProducerWallMicros = producer.WallMicros
	legacy.ProducerComputeMicros = producer.ComputeMicros
	legacy.ProducerOverheadMicros = producer.OverheadMicros
	legacy.BoundarySerializationMicros = consumer.RequestSerializationMicros
	legacy.ConsumerResponseSerializationMicros = consumer.ResponseSerializationMicros
	legacy.ConsumerRoundTripMicros = consumer.WallMicros
	legacy.ConsumerComputeMicros = consumer.ComputeMicros
	legacy.TransportOverheadMicros = consumer.OverheadMicros
	legacy.BoundaryTensorBytes = consumer.InputTensorBytes
	legacy.BoundaryWireBytes = consumer.InputWireBytes
	legacy.ConsumerResponseTensorBytes = consumer.ResponseTensorBytes
	legacy.ConsumerResponseWireBytes = consumer.ResponseWireBytes
	legacy.TerminalSampling = consumer.TerminalSampling
	legacy.ProducerKVCacheBytes = producer.KVCacheBytes
	legacy.ConsumerKVCacheBytes = consumer.KVCacheBytes
	legacy.ProducerMemory = producer.Memory
	legacy.ConsumerMemory = consumer.Memory
	return legacy
}

func legacyFailurePhase(phase string) string {
	switch phase {
	case "stage_0_prefill":
		return "producer_prefill"
	case "stage_1_prefill":
		return "consumer_prefill"
	case "stage_0_decode":
		return "producer_decode"
	case "stage_1_decode":
		return "consumer_decode"
	default:
		return phase
	}
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

package generation

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/tensorcheck"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

// PlannedStageObserver receives one observation for each completed distributed
// prefill/decode traversal after the resulting token has been sampled.
type PlannedStageObserver func(PlannedStageSample)

// PlannedStageSample is the N-stage equivalent of the legacy two-stage
// StageSample. The ordered Stages slice preserves per-hop compute, transport,
// memory, and KV evidence without collapsing it back into producer/consumer
// fields.
type PlannedStageSample struct {
	Operation                   string                 `json:"operation"`
	Position                    uint64                 `json:"position"`
	InputTokenCount             int                    `json:"inputTokenCount"`
	Stages                      []StageExecution       `json:"stages"`
	DistributedEndToEndMicros   int64                  `json:"distributedEndToEndMicros"`
	SamplingMicros              int64                  `json:"samplingMicros"`
	TokenLatencyMicros          int64                  `json:"tokenLatencyMicros"`
	ReferenceWallMicros         int64                  `json:"referenceWallMicros,omitempty"`
	ReferenceComputeMicros      uint64                 `json:"referenceComputeMicros,omitempty"`
	ReferenceSamplingMicros     int64                  `json:"referenceSamplingMicros,omitempty"`
	ReferenceTokenLatencyMicros int64                  `json:"referenceTokenLatencyMicros,omitempty"`
	ReferenceKVCacheBytes       int                    `json:"referenceKVCacheBytes,omitempty"`
	ReferenceMemory             workerproc.StageMemory `json:"referenceMemory,omitempty"`
}

// PlannedSessionConfig mirrors the generation controls that matter to an
// arbitrary execution plan while using an N-stage observer instead of the
// legacy producer/consumer observer.
type PlannedSessionConfig struct {
	RTol           float64
	ATol           float64
	ForwardTimeout time.Duration
	Observer       PlannedStageObserver
	LogitsObserver LogitsObserver
}

type PlannedTiming struct {
	SessionSetupMicros     int64    `json:"sessionSetupMicros"`
	TokenizeMicros         int64    `json:"tokenizeMicros"`
	PrefillMicros          int64    `json:"prefillMicros"`
	DecodeMicros           int64    `json:"decodeMicros"`
	DetokenizeMicros       int64    `json:"detokenizeMicros"`
	TotalMicros            int64    `json:"totalMicros"`
	StageComputeMicros     []uint64 `json:"stageComputeMicros"`
	ReferenceComputeMicros uint64   `json:"referenceComputeMicros,omitempty"`
}

// PlannedResult is the machine-readable result of generation through an
// arbitrary ordered execution plan. Per-stage arrays are aligned with
// ExecutionPlan.Stages.
type PlannedResult struct {
	Model                 string        `json:"model"`
	ModelType             string        `json:"modelType"`
	CheckpointFingerprint string        `json:"checkpointFingerprint"`
	CheckpointBytes       uint64        `json:"checkpointBytes"`
	ExecutionPlan         ExecutionPlan `json:"executionPlan"`
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
	StageKVCacheBytes     []int         `json:"stageKVCacheBytes"`
	Timing                PlannedTiming `json:"timing"`
	Verification          *Verification `json:"verification,omitempty"`
	Failure               *Failure      `json:"failure,omitempty"`
}

// PlannedSession is the canonical generation runtime for an explicit ordered
// execution plan. Session remains a two-stage compatibility facade over this
// implementation so existing proof commands exercise the same code path.
type PlannedSession struct {
	targets        []boundExecutionTarget
	reference      workerproc.PersistentCaller
	config         PlannedSessionConfig
	model          workerproc.PersistentModelResult
	plan           ExecutionPlan
	referenceShard string
	stageLoads     []StageLoad
	setupMicros    int64
}

type PlannedSessionInfo struct {
	Model              workerproc.PersistentModelResult `json:"model"`
	ExecutionPlan      ExecutionPlan                    `json:"executionPlan"`
	ReferenceShardID   string                           `json:"referenceShardID,omitempty"`
	StageLoads         []StageLoad                      `json:"stageLoads"`
	SessionSetupMicros int64                            `json:"sessionSetupMicros"`
}

func (s *PlannedSession) Info() PlannedSessionInfo {
	return PlannedSessionInfo{
		Model: s.model, ExecutionPlan: s.plan, ReferenceShardID: s.referenceShard,
		StageLoads:         append([]StageLoad(nil), s.stageLoads...),
		SessionSetupMicros: s.setupMicros,
	}
}

func NewPlannedSession(
	ctx context.Context,
	plan ExecutionPlan,
	targets []ExecutionTarget,
	reference workerproc.PersistentCaller,
	config PlannedSessionConfig,
) (*PlannedSession, error) {
	started := time.Now()
	if err := validatePlannedSessionConfig(&config, plan, reference); err != nil {
		return nil, err
	}
	bound, model, err := preflightExecutionTargets(ctx, plan, targets)
	if err != nil {
		return nil, err
	}

	referenceShard := ""
	if reference != nil {
		referenceModel, infoErr := modelInfo(ctx, reference, plan.Model.ID)
		if infoErr != nil {
			return nil, fmt.Errorf("reference model info: %w", infoErr)
		}
		if err := matchPlanModel(plan.Model, referenceModel, "reference"); err != nil {
			return nil, err
		}
		if err := matchModel(model, referenceModel, "reference"); err != nil {
			return nil, err
		}
	}

	// No target loads until all distributed and reference metadata agrees with
	// the immutable plan identity.
	stageLoads, err := loadExecutionTargets(ctx, plan, bound)
	if err != nil {
		return nil, err
	}
	if reference != nil {
		referenceShard = "generate-reference-" + modelHashSuffix(plan.Model.ID, model.CheckpointFingerprint)
		if _, _, err := ensureShard(ctx, reference, workerproc.PersistentLoadShardRequest{
			ModelID: plan.Model.ID, ShardID: referenceShard,
			CheckpointFingerprint: model.CheckpointFingerprint,
			LayerStart:            0, LayerEnd: model.LayerCount, OwnsInput: true, OwnsOutput: true,
		}); err != nil {
			return nil, fmt.Errorf("reference shard: %w", err)
		}
	}

	return &PlannedSession{
		targets: bound, reference: reference, config: config,
		model: *model, plan: plan, referenceShard: referenceShard,
		stageLoads:  stageLoads,
		setupMicros: time.Since(started).Microseconds(),
	}, nil
}

func validatePlannedSessionConfig(
	config *PlannedSessionConfig,
	plan ExecutionPlan,
	reference workerproc.PersistentCaller,
) error {
	if err := ValidateExecutionPlan(plan); err != nil {
		return fmt.Errorf("execution plan: %w", err)
	}
	if config.RTol < 0 || config.ATol < 0 ||
		math.IsNaN(config.RTol) || math.IsNaN(config.ATol) ||
		math.IsInf(config.RTol, 0) || math.IsInf(config.ATol, 0) {
		return errors.New("numeric tolerances must be finite and non-negative")
	}
	if config.ForwardTimeout < 0 {
		return errors.New("forward timeout must be non-negative")
	}
	if config.ForwardTimeout > 0 && config.ForwardTimeout < time.Millisecond {
		return errors.New("forward timeout must be at least 1ms")
	}
	if config.ForwardTimeout == 0 {
		config.ForwardTimeout = DefaultForwardTimeout
	}
	terminalSampling := plan.Stages[len(plan.Stages)-1].ResponseMode == StageResponseSampledToken
	if terminalSampling && (reference != nil || config.LogitsObserver != nil) {
		return errors.New("terminal sampling cannot return logits for reference verification")
	}
	return nil
}

// NewBalancedPlannedSession discovers the pinned model identity, constructs a
// deterministic balanced plan for the supplied target IDs, and prepares the
// canonical N-stage session. It exists for explicit correctness experiments;
// automatic placement will construct ExecutionPlan directly.
func NewBalancedPlannedSession(
	ctx context.Context,
	modelID string,
	targets []ExecutionTarget,
	reference workerproc.PersistentCaller,
	terminalResponseMode StageResponseMode,
	config PlannedSessionConfig,
) (*PlannedSession, error) {
	started := time.Now()
	if modelID == "" {
		return nil, errors.New("model is required")
	}
	if len(targets) == 0 || targets[0].Caller == nil {
		return nil, errors.New("at least one execution target is required")
	}
	model, err := modelInfo(ctx, targets[0].Caller, modelID)
	if err != nil {
		return nil, fmt.Errorf("discover model for execution plan: %w", err)
	}
	targetIDs := make([]string, len(targets))
	for index, target := range targets {
		targetIDs[index] = target.TargetID
	}
	plan, err := BuildBalancedExecutionPlan(ExecutionModel{
		ID:                    model.ModelID,
		CheckpointFingerprint: model.CheckpointFingerprint,
		LayerCount:            model.LayerCount,
	}, targetIDs, terminalResponseMode)
	if err != nil {
		return nil, err
	}
	session, err := NewPlannedSession(ctx, plan, targets, reference, config)
	if err != nil {
		return nil, err
	}
	session.setupMicros = time.Since(started).Microseconds()
	return session, nil
}

func (s *PlannedSession) Generate(
	ctx context.Context,
	request Request,
) (result PlannedResult, returnErr error) {
	started := time.Now()
	defer func() { result.Timing.TotalMicros = time.Since(started).Microseconds() }()
	if err := ctx.Err(); err != nil {
		return PlannedResult{}, err
	}
	if request.Prompt == "" {
		return PlannedResult{}, errors.New("prompt is required")
	}
	if request.MaxTokens <= 0 {
		return PlannedResult{}, errors.New("max tokens must be positive")
	}
	if request.SequenceID == "" {
		sequenceID, err := randomSequenceID()
		if err != nil {
			return PlannedResult{}, err
		}
		request.SequenceID = sequenceID
	}

	result = PlannedResult{
		Model: s.plan.Model.ID, ModelType: s.model.ModelType,
		CheckpointFingerprint: s.model.CheckpointFingerprint,
		CheckpointBytes:       s.model.CheckpointBytes, ExecutionPlan: s.plan,
		SequenceID: request.SequenceID, Prompt: request.Prompt, MaxTokens: request.MaxTokens,
		RTol: s.config.RTol, ATol: s.config.ATol,
		ForwardTimeoutMillis: s.config.ForwardTimeout.Milliseconds(),
		StageKVCacheBytes:    make([]int, len(s.targets)),
		Timing: PlannedTiming{
			SessionSetupMicros: s.setupMicros,
			StageComputeMicros: make([]uint64, len(s.targets)),
		},
	}
	point := failurePoint{phase: "tokenize", shardID: s.targets[0].Stage.ShardID}
	defer func() {
		if returnErr == nil || result.Failure != nil {
			return
		}
		failure := failureFromPlanned(point, result, returnErr)
		result.Failure = &failure
		returnErr = &GenerationError{Failure: failure, Err: returnErr}
	}()

	inputCaller := s.targets[0].Caller
	tokenizeStarted := time.Now()
	tokenized, err := tokenize(ctx, inputCaller, s.plan.Model.ID, request.Prompt)
	result.Timing.TokenizeMicros = time.Since(tokenizeStarted).Microseconds()
	if err != nil {
		return result, fmt.Errorf("tokenize prompt: %w", err)
	}
	if len(tokenized.TokenIDs) == 0 {
		return result, errors.New("tokenizer returned no prompt tokens")
	}
	result.PromptTokenIDs = tokenized.TokenIDs
	result.EOSTokenID = tokenized.EOSTokenID

	targets := executionSequenceTargets(s.targets)
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
			if cleanupErr == nil {
				return
			}
			if returnErr == nil {
				point = failurePoint{phase: "sequence_cleanup"}
				returnErr = cleanupErr
			} else {
				returnErr = errors.Join(returnErr, cleanupErr)
			}
		}()
	}
	if err != nil {
		return result, fmt.Errorf("open generation sequences: %w", err)
	}

	prompt := tokenTensor(tokenized.TokenIDs)
	prefillStarted := time.Now()
	distributedStarted := time.Now()
	point = failurePoint{phase: "distributed_prefill", operation: "prefill", position: 0}
	stageExecutions, terminalResult, err := executeBoundStageChain(
		ctx, s.config.ForwardTimeout, s.targets,
		"prefill", request.SequenceID, 0, prompt,
	)
	if err != nil {
		point = failurePointForStageError("prefill", 0, err)
		return result, fmt.Errorf("distributed prefill: %w", err)
	}
	distributedEndToEndMicros := time.Since(distributedStarted).Microseconds()
	accumulatePlannedStageState(&result, stageExecutions)

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
		result.Timing.ReferenceComputeMicros += referenceResult.ComputeMicros
		result.Verification = &Verification{GreedyTokenIDsMatch: true}
	}
	result.Timing.PrefillMicros = time.Since(prefillStarted).Microseconds()

	var pendingSample *PlannedStageSample
	if s.config.Observer != nil {
		sample := newPlannedStageSample(
			"prefill", 0, len(tokenized.TokenIDs), stageExecutions,
			distributedEndToEndMicros, referenceResult, referenceWallMicros,
		)
		pendingSample = &sample
	}

	position := uint64(len(tokenized.TokenIDs))
	distributedLogits := terminalResult.Output
	distributedSampledToken := terminalResult.SampledTokenID
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
				len(result.GeneratedTokenIDs), cloneWireTensor(distributedLogits),
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
			phase: "distributed_decode", operation: "decode", position: position,
		}
		stageExecutions, terminalResult, err = executeBoundStageChain(
			ctx, s.config.ForwardTimeout, s.targets,
			"decode", request.SequenceID, position, token,
		)
		if err != nil {
			point = failurePointForStageError("decode", position, err)
			return result, fmt.Errorf("distributed decode step %d: %w", len(result.GeneratedTokenIDs), err)
		}
		distributedEndToEndMicros = time.Since(distributedStarted).Microseconds()
		accumulatePlannedStageState(&result, stageExecutions)
		distributedLogits = terminalResult.Output
		distributedSampledToken = terminalResult.SampledTokenID

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
			sample := newPlannedStageSample(
				"decode", position, 1, stageExecutions,
				distributedEndToEndMicros, referenceResult, referenceWallMicros,
			)
			pendingSample = &sample
		}
		position++
	}
	result.Timing.DecodeMicros = time.Since(decodeStarted).Microseconds()

	point = failurePoint{phase: "detokenize", shardID: s.targets[0].Stage.ShardID}
	detokenizeStarted := time.Now()
	text, err := detokenize(ctx, inputCaller, s.plan.Model.ID, result.GeneratedTokenIDs)
	result.Timing.DetokenizeMicros = time.Since(detokenizeStarted).Microseconds()
	if err != nil {
		return result, fmt.Errorf("detokenize generated tokens: %w", err)
	}
	result.Text = text
	point = failurePoint{}
	return result, nil
}

func accumulatePlannedStageState(result *PlannedResult, executions []StageExecution) {
	for _, execution := range executions {
		if execution.Index < 0 || execution.Index >= len(result.StageKVCacheBytes) {
			continue
		}
		result.StageKVCacheBytes[execution.Index] = execution.KVCacheBytes
		result.Timing.StageComputeMicros[execution.Index] += execution.ComputeMicros
	}
}

func newPlannedStageSample(
	operation string,
	position uint64,
	inputTokenCount int,
	executions []StageExecution,
	distributedEndToEndMicros int64,
	reference *workerproc.PersistentForwardResult,
	referenceWallMicros int64,
) PlannedStageSample {
	stages := append([]StageExecution(nil), executions...)
	for index := range stages {
		// Observers only need measurements. Retaining forward results here would
		// keep every boundary/logit tensor alive when a benchmark stores samples.
		stages[index].Result = nil
	}
	sample := PlannedStageSample{
		Operation: operation, Position: position, InputTokenCount: inputTokenCount,
		Stages:                    stages,
		DistributedEndToEndMicros: distributedEndToEndMicros,
	}
	if reference != nil {
		sample.ReferenceWallMicros = referenceWallMicros
		sample.ReferenceComputeMicros = reference.ComputeMicros
		sample.ReferenceKVCacheBytes = reference.KVCacheBytes
		sample.ReferenceMemory = reference.Memory
	}
	return sample
}

func failurePointForStageError(operation string, position uint64, err error) failurePoint {
	point := failurePoint{
		phase: "distributed_" + operation, operation: operation, position: position,
	}
	var stageErr *ExecutionStageError
	if errors.As(err, &stageErr) {
		point.phase = fmt.Sprintf("stage_%d_%s", stageErr.Index, operation)
		point.shardID = stageErr.Stage.ShardID
		point.position = stageErr.Position
	}
	return point
}

func failureFromPlanned(point failurePoint, result PlannedResult, err error) Failure {
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

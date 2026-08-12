package generation

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

// ExecutionTarget binds one serializable execution stage to the concrete
// persistent caller that will execute it. Placement owns this binding; the
// model plan itself remains transport- and backend-neutral.
type ExecutionTarget struct {
	Stage  ExecutionStage
	Caller workerproc.PersistentCaller
}

// StageExecution captures one stage invocation in an ordered prefill/decode
// traversal. It is intentionally detailed enough to become the per-stage
// evidence emitted by the generalized generation session.
type StageExecution struct {
	Index              int                    `json:"index"`
	Stage              ExecutionStage         `json:"stage"`
	Operation          string                 `json:"operation"`
	Position           uint64                 `json:"position"`
	InputKind          string                 `json:"inputKind"`
	InputTensorBytes   int                    `json:"inputTensorBytes"`
	InputWireBytes     int                    `json:"inputWireBytes"`
	ResponseTensorBytes int                   `json:"responseTensorBytes"`
	ResponseWireBytes  int                    `json:"responseWireBytes"`
	WallMicros         int64                  `json:"wallMicros"`
	ComputeMicros      uint64                 `json:"computeMicros"`
	OverheadMicros     int64                  `json:"overheadMicros"`
	KVCacheBytes       int                    `json:"kvCacheBytes"`
	Memory             workerproc.StageMemory `json:"memory"`
	TerminalSampling   bool                   `json:"terminalSampling"`
	Result             *workerproc.PersistentForwardResult `json:"-"`
}

// ExecutionStageError identifies the exact stage that failed. The caller can
// translate it into the public generation failure contract without losing the
// ordered-stage context.
type ExecutionStageError struct {
	Index     int
	Stage     ExecutionStage
	Operation string
	Position  uint64
	Err       error
}

func (err *ExecutionStageError) Error() string {
	return fmt.Sprintf(
		"stage %d (%s/%s) %s at position %d: %v",
		err.Index, err.Stage.Name, err.Stage.ShardID, err.Operation, err.Position, err.Err,
	)
}

func (err *ExecutionStageError) Unwrap() error { return err.Err }

// PrepareExecutionTargets validates all model metadata and the complete plan
// before loading any shard. This keeps checkpoint mismatch or invalid topology
// failures side-effect free. Only after every target agrees on the model does
// it materialize the requested complementary ranges.
func PrepareExecutionTargets(
	ctx context.Context,
	modelID string,
	targets []ExecutionTarget,
) (*workerproc.PersistentModelResult, error) {
	if modelID == "" {
		return nil, errors.New("model is required")
	}
	if len(targets) == 0 {
		return nil, errors.New("at least one execution target is required")
	}
	for index, target := range targets {
		if target.Caller == nil {
			return nil, fmt.Errorf("stage %d (%s) has no caller", index, target.Stage.Name)
		}
	}

	models := make([]*workerproc.PersistentModelResult, len(targets))
	for index, target := range targets {
		model, err := modelInfo(ctx, target.Caller, modelID)
		if err != nil {
			return nil, fmt.Errorf("stage %d (%s) model info: %w", index, target.Stage.Name, err)
		}
		models[index] = model
	}
	model := models[0]
	for index := 1; index < len(models); index++ {
		if err := matchModel(model, models[index], targets[index].Stage.Name); err != nil {
			return nil, err
		}
	}

	plan := ExecutionPlan{Stages: make([]ExecutionStage, len(targets))}
	for index, target := range targets {
		plan.Stages[index] = target.Stage
	}
	if err := ValidateExecutionPlan(plan, model.LayerCount); err != nil {
		return nil, fmt.Errorf("execution plan: %w", err)
	}

	for index, target := range targets {
		stage := target.Stage
		if _, err := ensureShard(ctx, target.Caller, workerproc.PersistentLoadShardRequest{
			ModelID:               modelID,
			ShardID:               stage.ShardID,
			CheckpointFingerprint: model.CheckpointFingerprint,
			LayerStart:            stage.LayerStart,
			LayerEnd:              stage.LayerEnd,
			OwnsInput:             stage.OwnsInput,
			OwnsOutput:            stage.OwnsOutput,
		}); err != nil {
			return nil, fmt.Errorf("stage %d (%s) load shard: %w", index, stage.Name, err)
		}
	}
	copy := *model
	return &copy, nil
}

// ExecutionSequenceTargets converts the ordered execution targets into the
// owner-safe lifecycle targets used by workerproc.OpenSequences.
func ExecutionSequenceTargets(targets []ExecutionTarget) []workerproc.SequenceTarget {
	result := make([]workerproc.SequenceTarget, 0, len(targets))
	for index, target := range targets {
		name := target.Stage.Name
		if name == "" {
			name = fmt.Sprintf("stage-%d", index)
		}
		result = append(result, workerproc.SequenceTarget{
			Name: name, Caller: target.Caller, ShardID: target.Stage.ShardID,
		})
	}
	return result
}

// ExecuteStageChain forwards one prefill or decode input through every target
// in order. The first stage receives tokens, all later stages receive hidden
// state, and only the final output-owning stage may return a sampled token.
func ExecuteStageChain(
	ctx context.Context,
	timeout time.Duration,
	targets []ExecutionTarget,
	operation string,
	sequenceID string,
	position uint64,
	input workerproc.WireTensor,
	terminalSampling bool,
) ([]StageExecution, *workerproc.PersistentForwardResult, error) {
	if len(targets) == 0 {
		return nil, nil, errors.New("at least one execution target is required")
	}
	if operation != "prefill" && operation != "decode" {
		return nil, nil, fmt.Errorf("unsupported execution operation %q", operation)
	}
	if sequenceID == "" {
		return nil, nil, errors.New("sequence ID is required")
	}

	executions := make([]StageExecution, 0, len(targets))
	current := input
	inputKind := "tokens"
	for index, target := range targets {
		isTerminal := index == len(targets)-1
		returnSampledToken := terminalSampling && isTerminal
		result, wallMicros, err := measuredInfer(
			ctx, timeout, target.Caller,
			operation, target.Stage.ShardID, sequenceID, position, inputKind, current,
			returnSampledToken,
		)
		if err != nil {
			return executions, nil, &ExecutionStageError{
				Index: index, Stage: target.Stage, Operation: operation, Position: position, Err: err,
			}
		}
		execution := newStageExecution(
			index, target.Stage, operation, position, sequenceID, inputKind, current,
			result, wallMicros,
		)
		executions = append(executions, execution)
		if isTerminal {
			return executions, result, nil
		}
		current = result.Output
		inputKind = "hidden"
	}
	return executions, nil, errors.New("execution chain produced no terminal result")
}

func newStageExecution(
	index int,
	stage ExecutionStage,
	operation string,
	position uint64,
	sequenceID string,
	inputKind string,
	input workerproc.WireTensor,
	result *workerproc.PersistentForwardResult,
	wallMicros int64,
) StageExecution {
	requestPayload, _ := json.Marshal(workerproc.PersistentRequest{
		Command: operation, DeadlineUnixMillis: time.Now().UnixMilli(),
		Forward: &workerproc.PersistentForwardRequest{
			ShardID: stage.ShardID, SequenceID: sequenceID, Position: position,
			InputKind: inputKind, Input: input,
			ReturnSampledToken: result.SampledTokenID != nil,
		},
	})
	responsePayload, _ := json.Marshal(workerproc.PersistentResponse{
		OK: true,
		Result: &workerproc.PersistentWorkerResult{Forward: result},
	})
	return StageExecution{
		Index: index, Stage: stage, Operation: operation, Position: position,
		InputKind: inputKind,
		InputTensorBytes: len(input.Data), InputWireBytes: len(requestPayload),
		ResponseTensorBytes: len(result.Output.Data), ResponseWireBytes: len(responsePayload),
		WallMicros: wallMicros, ComputeMicros: result.ComputeMicros,
		OverheadMicros: positiveDifference(wallMicros, result.ComputeMicros),
		KVCacheBytes: result.KVCacheBytes, Memory: result.Memory,
		TerminalSampling: result.SampledTokenID != nil,
		Result: result,
	}
}

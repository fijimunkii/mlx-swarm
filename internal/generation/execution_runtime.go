package generation

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

// ExecutionTarget binds a stable plan target identity to a concrete caller.
// The stage itself remains solely in ExecutionPlan so plan identity cannot
// drift from its transport binding.
type ExecutionTarget struct {
	TargetID string
	Caller   workerproc.PersistentCaller
}

type boundExecutionTarget struct {
	Stage  ExecutionStage
	Caller workerproc.PersistentCaller
}

// StageExecution captures one stage invocation in an ordered prefill/decode
// traversal.
type StageExecution struct {
	Index                       int                                 `json:"index"`
	Stage                       ExecutionStage                      `json:"stage"`
	Operation                   string                              `json:"operation"`
	Position                    uint64                              `json:"position"`
	InputKind                   string                              `json:"inputKind"`
	InputTensorBytes            int                                 `json:"inputTensorBytes"`
	InputWireBytes              int                                 `json:"inputWireBytes"`
	RequestSerializationMicros  int64                               `json:"requestSerializationMicros"`
	ResponseTensorBytes         int                                 `json:"responseTensorBytes"`
	ResponseWireBytes           int                                 `json:"responseWireBytes"`
	ResponseSerializationMicros int64                               `json:"responseSerializationMicros"`
	WallMicros                  int64                               `json:"wallMicros"`
	ComputeMicros               uint64                              `json:"computeMicros"`
	OverheadMicros              int64                               `json:"overheadMicros"`
	KVCacheBytes                int                                 `json:"kvCacheBytes"`
	Memory                      workerproc.StageMemory              `json:"memory"`
	TerminalSampling            bool                                `json:"terminalSampling"`
	Result                      *workerproc.PersistentForwardResult `json:"-"`
}

// ExecutionStageError identifies the exact stage that failed.
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

// ExecutionStageLoadError identifies a partially prepared plan whose earlier
// stages may need rollback by the owning proof/session.
type ExecutionStageLoadError struct {
	Index int
	Stage ExecutionStage
	Err   error
}

func (err *ExecutionStageLoadError) Error() string {
	return fmt.Sprintf(
		"stage %d (%s/%s) load shard: %v",
		err.Index, err.Stage.Name, err.Stage.ShardID, err.Err,
	)
}

func (err *ExecutionStageLoadError) Unwrap() error { return err.Err }

// PrepareExecutionTargets validates plan identity and every worker's model
// metadata before loading any shard.
func PrepareExecutionTargets(
	ctx context.Context,
	plan ExecutionPlan,
	targets []ExecutionTarget,
) (*workerproc.PersistentModelResult, error) {
	_, model, err := prepareExecutionTargets(ctx, plan, targets)
	return model, err
}

func prepareExecutionTargets(
	ctx context.Context,
	plan ExecutionPlan,
	targets []ExecutionTarget,
) ([]boundExecutionTarget, *workerproc.PersistentModelResult, error) {
	bound, model, err := preflightExecutionTargets(ctx, plan, targets)
	if err != nil {
		return nil, nil, err
	}
	if err := loadExecutionTargets(ctx, plan, bound); err != nil {
		return nil, nil, err
	}
	return bound, model, nil
}

func preflightExecutionTargets(
	ctx context.Context,
	plan ExecutionPlan,
	targets []ExecutionTarget,
) ([]boundExecutionTarget, *workerproc.PersistentModelResult, error) {
	if err := ValidateExecutionPlan(plan); err != nil {
		return nil, nil, fmt.Errorf("execution plan: %w", err)
	}
	bound, err := bindExecutionTargets(plan, targets)
	if err != nil {
		return nil, nil, err
	}

	models := make([]*workerproc.PersistentModelResult, len(bound))
	for index, target := range bound {
		model, modelErr := modelInfo(ctx, target.Caller, plan.Model.ID)
		if modelErr != nil {
			return nil, nil, fmt.Errorf(
				"stage %d (%s) model info: %w", index, target.Stage.Name, modelErr,
			)
		}
		if err := matchPlanModel(plan.Model, model, target.Stage.Name); err != nil {
			return nil, nil, err
		}
		models[index] = model
	}
	model := models[0]
	for index := 1; index < len(models); index++ {
		if err := matchModel(model, models[index], bound[index].Stage.Name); err != nil {
			return nil, nil, err
		}
	}
	copy := *model
	return bound, &copy, nil
}

func loadExecutionTargets(
	ctx context.Context,
	plan ExecutionPlan,
	bound []boundExecutionTarget,
) error {
	for index, target := range bound {
		stage := target.Stage
		if _, err := ensureShard(ctx, target.Caller, workerproc.PersistentLoadShardRequest{
			ModelID:               plan.Model.ID,
			ShardID:               stage.ShardID,
			CheckpointFingerprint: plan.Model.CheckpointFingerprint,
			LayerStart:            stage.LayerStart,
			LayerEnd:              stage.LayerEnd,
			OwnsInput:             stage.OwnsInput,
			OwnsOutput:            stage.OwnsOutput,
		}); err != nil {
			return &ExecutionStageLoadError{Index: index, Stage: stage, Err: err}
		}
	}
	return nil
}

func matchPlanModel(
	expected ExecutionModel,
	actual *workerproc.PersistentModelResult,
	role string,
) error {
	if expected.ID != actual.ModelID ||
		expected.CheckpointFingerprint != actual.CheckpointFingerprint ||
		expected.LayerCount != actual.LayerCount {
		return fmt.Errorf(
			"%s model does not match execution plan: plan=%+v actual=%+v",
			role, expected, *actual,
		)
	}
	return nil
}

func bindExecutionTargets(
	plan ExecutionPlan,
	targets []ExecutionTarget,
) ([]boundExecutionTarget, error) {
	if len(targets) != len(plan.Stages) {
		return nil, fmt.Errorf(
			"execution target count %d does not match stage count %d",
			len(targets), len(plan.Stages),
		)
	}
	byID := make(map[string]workerproc.PersistentCaller, len(targets))
	for index, target := range targets {
		if target.TargetID == "" {
			return nil, fmt.Errorf("execution target %d has no target ID", index)
		}
		if target.Caller == nil {
			return nil, fmt.Errorf("execution target %q has no caller", target.TargetID)
		}
		if _, exists := byID[target.TargetID]; exists {
			return nil, fmt.Errorf("execution target %q is duplicated", target.TargetID)
		}
		byID[target.TargetID] = target.Caller
	}
	bound := make([]boundExecutionTarget, len(plan.Stages))
	for index, stage := range plan.Stages {
		caller, exists := byID[stage.TargetID]
		if !exists {
			return nil, fmt.Errorf(
				"stage %d (%s) target %q is not bound",
				index, stage.Name, stage.TargetID,
			)
		}
		bound[index] = boundExecutionTarget{Stage: stage, Caller: caller}
		delete(byID, stage.TargetID)
	}
	if len(byID) != 0 {
		return nil, errors.New("execution targets contain identities not present in the plan")
	}
	return bound, nil
}

func executionSequenceTargets(targets []boundExecutionTarget) []workerproc.SequenceTarget {
	result := make([]workerproc.SequenceTarget, 0, len(targets))
	for _, target := range targets {
		result = append(result, workerproc.SequenceTarget{
			Name: target.Stage.Name, Caller: target.Caller, ShardID: target.Stage.ShardID,
		})
	}
	return result
}

// ExecuteStageChain validates and binds the plan, then forwards one prefill or
// decode input through each stage in order.
func ExecuteStageChain(
	ctx context.Context,
	timeout time.Duration,
	plan ExecutionPlan,
	targets []ExecutionTarget,
	operation string,
	sequenceID string,
	position uint64,
	input workerproc.WireTensor,
) ([]StageExecution, *workerproc.PersistentForwardResult, error) {
	if err := ValidateExecutionPlan(plan); err != nil {
		return nil, nil, fmt.Errorf("execution plan: %w", err)
	}
	bound, err := bindExecutionTargets(plan, targets)
	if err != nil {
		return nil, nil, err
	}
	return executeBoundStageChain(
		ctx, timeout, bound, operation, sequenceID, position, input,
	)
}

func executeBoundStageChain(
	ctx context.Context,
	timeout time.Duration,
	targets []boundExecutionTarget,
	operation string,
	sequenceID string,
	position uint64,
	input workerproc.WireTensor,
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
		returnSampledToken := target.Stage.ResponseMode == StageResponseSampledToken
		result, wallMicros, err := measuredInfer(
			ctx, timeout, target.Caller,
			operation, target.Stage.ShardID, sequenceID, position, inputKind, current,
			returnSampledToken,
		)
		if err != nil {
			return executions, nil, &ExecutionStageError{
				Index: index, Stage: target.Stage, Operation: operation,
				Position: position, Err: err,
			}
		}
		execution := newStageExecution(
			index, target.Stage, operation, position, sequenceID, inputKind, current,
			result, wallMicros,
		)
		executions = append(executions, execution)
		if index == len(targets)-1 {
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
	requestStarted := time.Now()
	requestPayload, _ := json.Marshal(workerproc.PersistentRequest{
		Command: operation, DeadlineUnixMillis: time.Now().UnixMilli(),
		Forward: &workerproc.PersistentForwardRequest{
			ShardID: stage.ShardID, SequenceID: sequenceID, Position: position,
			InputKind: inputKind, Input: input,
			ReturnSampledToken: stage.ResponseMode == StageResponseSampledToken,
		},
	})
	requestSerializationMicros := time.Since(requestStarted).Microseconds()
	responseStarted := time.Now()
	responsePayload, _ := json.Marshal(workerproc.PersistentResponse{
		OK:     true,
		Result: &workerproc.PersistentWorkerResult{Forward: result},
	})
	responseSerializationMicros := time.Since(responseStarted).Microseconds()
	return StageExecution{
		Index: index, Stage: stage, Operation: operation, Position: position,
		InputKind:        inputKind,
		InputTensorBytes: len(input.Data), InputWireBytes: len(requestPayload),
		RequestSerializationMicros: requestSerializationMicros,
		ResponseTensorBytes:        len(result.Output.Data), ResponseWireBytes: len(responsePayload),
		ResponseSerializationMicros: responseSerializationMicros,
		WallMicros:                  wallMicros, ComputeMicros: result.ComputeMicros,
		OverheadMicros: positiveDifference(wallMicros, result.ComputeMicros),
		KVCacheBytes:   result.KVCacheBytes, Memory: result.Memory,
		TerminalSampling: result.SampledTokenID != nil,
		Result:           result,
	}
}

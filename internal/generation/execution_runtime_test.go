package generation

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

func TestPrepareExecutionTargetsPreflightsAllModelsBeforeLoading(t *testing.T) {
	plan, err := BuildBalancedExecutionPlan(
		ExecutionModel{ID: "test/model", CheckpointFingerprint: "expected", LayerCount: 6},
		testExecutionTargetIDs(3), StageResponseTensor,
	)
	if err != nil {
		t.Fatal(err)
	}
	workers := []*executionFakeWorker{
		newExecutionFakeWorker("a", "expected", 6),
		newExecutionFakeWorker("b", "different", 6),
		newExecutionFakeWorker("c", "expected", 6),
	}
	targets := bindExecutionFakeTargets(plan, workers)

	_, err = PrepareExecutionTargets(context.Background(), plan, targets)
	if err == nil {
		t.Fatal("expected checkpoint mismatch")
	}
	for _, worker := range workers {
		if worker.loadCount != 0 {
			t.Fatalf("worker %s loaded %d shards before preflight failed", worker.name, worker.loadCount)
		}
	}
}

func TestPrepareExecutionTargetsLoadsFiveComplementaryStages(t *testing.T) {
	plan, err := BuildBalancedExecutionPlan(
		ExecutionModel{ID: "test/model", CheckpointFingerprint: "expected", LayerCount: 18},
		testExecutionTargetIDs(5), StageResponseTensor,
	)
	if err != nil {
		t.Fatal(err)
	}
	workers := make([]*executionFakeWorker, 5)
	for index := range workers {
		workers[index] = newExecutionFakeWorker(fmt.Sprintf("worker-%d", index), "expected", 18)
	}
	targets := bindExecutionFakeTargets(plan, workers)

	model, err := PrepareExecutionTargets(context.Background(), plan, targets)
	if err != nil {
		t.Fatal(err)
	}
	if model.LayerCount != 18 || model.CheckpointFingerprint != "expected" {
		t.Fatalf("unexpected model metadata: %+v", model)
	}
	for index, worker := range workers {
		if worker.loadCount != 1 || len(worker.loaded) != 1 {
			t.Fatalf("worker %d load state: count=%d loaded=%d", index, worker.loadCount, len(worker.loaded))
		}
		loaded := worker.loaded[0]
		stage := plan.Stages[index]
		if loaded.ShardID != stage.ShardID || loaded.LayerStart != stage.LayerStart ||
			loaded.LayerEnd != stage.LayerEnd || loaded.OwnsInput != stage.OwnsInput ||
			loaded.OwnsOutput != stage.OwnsOutput {
			t.Fatalf("worker %d loaded %+v, want stage %+v", index, loaded, stage)
		}
	}
}

func TestPrepareExecutionTargetsCanReuseWorkersAcrossPlanSizes(t *testing.T) {
	workers := make([]*executionFakeWorker, 5)
	for index := range workers {
		workers[index] = newExecutionFakeWorker(fmt.Sprintf("worker-%d", index), "expected", 18)
	}
	two, err := BuildBalancedExecutionPlan(
		ExecutionModel{ID: "test/model", CheckpointFingerprint: "expected", LayerCount: 18},
		testExecutionTargetIDs(2), StageResponseTensor,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PrepareExecutionTargets(
		context.Background(), two, bindExecutionFakeTargets(two, workers[:2]),
	); err != nil {
		t.Fatal(err)
	}
	five, err := BuildBalancedExecutionPlan(
		ExecutionModel{ID: "test/model", CheckpointFingerprint: "expected", LayerCount: 18},
		testExecutionTargetIDs(5), StageResponseTensor,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PrepareExecutionTargets(
		context.Background(), five, bindExecutionFakeTargets(five, workers),
	); err != nil {
		t.Fatalf("prepare five-stage plan after two-stage plan: %v", err)
	}
	if workers[0].loadCount != 2 || workers[1].loadCount != 2 {
		t.Fatalf(
			"shared workers did not retain distinct plan shards: loads=%d/%d",
			workers[0].loadCount, workers[1].loadCount,
		)
	}
	if workers[0].loaded[0].ShardID == workers[0].loaded[1].ShardID {
		t.Fatalf("worker reused colliding shard ID %q", workers[0].loaded[0].ShardID)
	}
}

func TestExecuteStageChainTraversesFiveStagesInOrder(t *testing.T) {
	plan, err := BuildBalancedExecutionPlan(
		ExecutionModel{ID: "test/model", CheckpointFingerprint: "expected", LayerCount: 10},
		testExecutionTargetIDs(5), StageResponseSampledToken,
	)
	if err != nil {
		t.Fatal(err)
	}
	workers := make([]*executionFakeWorker, 5)
	for index := range workers {
		workers[index] = newExecutionFakeWorker(fmt.Sprintf("worker-%d", index), "expected", 10)
		workers[index].marker = byte(index + 1)
	}
	targets := bindExecutionFakeTargets(plan, workers)
	if _, err := PrepareExecutionTargets(context.Background(), plan, targets); err != nil {
		t.Fatal(err)
	}

	input := workerproc.WireTensor{Shape: []int{1, 2}, DType: "int32", Data: []byte{1, 2, 3, 4}}
	executions, terminal, err := ExecuteStageChain(
		context.Background(), time.Second, plan, targets,
		"prefill", "sequence", 0, input,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(executions) != 5 {
		t.Fatalf("executions = %d, want 5", len(executions))
	}
	if terminal == nil || terminal.SampledTokenID == nil || *terminal.SampledTokenID != 7 {
		t.Fatalf("unexpected terminal result: %+v", terminal)
	}
	for index, execution := range executions {
		wantKind := "hidden"
		if index == 0 {
			wantKind = "tokens"
		}
		if execution.Index != index || execution.Stage.ShardID != plan.Stages[index].ShardID ||
			execution.InputKind != wantKind || execution.KVCacheBytes <= 0 ||
			execution.InputWireBytes <= execution.InputTensorBytes || execution.ResponseWireBytes <= 0 {
			t.Fatalf("unexpected execution %d: %+v", index, execution)
		}
		if execution.TerminalSampling != (index == 4) {
			t.Fatalf("stage %d terminal sampling = %t", index, execution.TerminalSampling)
		}
	}
	for index, worker := range workers {
		if len(worker.inferenceCalls) != 1 {
			t.Fatalf("worker %d inference calls = %v", index, worker.inferenceCalls)
		}
		if worker.inferenceCalls[0].inputKind != executions[index].InputKind {
			t.Fatalf("worker %d input kind = %s, evidence = %s", index, worker.inferenceCalls[0].inputKind, executions[index].InputKind)
		}
		if worker.inferenceCalls[0].sample != (index == 4) {
			t.Fatalf("worker %d sample request = %t", index, worker.inferenceCalls[0].sample)
		}
	}
}

func TestExecuteStageChainReportsExactFailedStage(t *testing.T) {
	plan, err := BuildBalancedExecutionPlan(
		ExecutionModel{ID: "test/model", CheckpointFingerprint: "expected", LayerCount: 6},
		testExecutionTargetIDs(3), StageResponseTensor,
	)
	if err != nil {
		t.Fatal(err)
	}
	workers := []*executionFakeWorker{
		newExecutionFakeWorker("a", "expected", 6),
		newExecutionFakeWorker("b", "expected", 6),
		newExecutionFakeWorker("c", "expected", 6),
	}
	workers[1].inferErr = context.DeadlineExceeded
	targets := bindExecutionFakeTargets(plan, workers)
	if _, err := PrepareExecutionTargets(context.Background(), plan, targets); err != nil {
		t.Fatal(err)
	}

	executions, terminal, err := ExecuteStageChain(
		context.Background(), time.Second, plan, targets,
		"decode", "sequence", 9,
		workerproc.WireTensor{Shape: []int{1, 1}, DType: "int32", Data: []byte{1, 0, 0, 0}},
	)
	if terminal != nil || len(executions) != 1 {
		t.Fatalf("unexpected partial execution: terminal=%+v executions=%d", terminal, len(executions))
	}
	var stageErr *ExecutionStageError
	if !errors.As(err, &stageErr) || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("unexpected error: %v", err)
	}
	if stageErr.Index != 1 || stageErr.Stage.ShardID != plan.Stages[1].ShardID ||
		stageErr.Operation != "decode" || stageErr.Position != 9 {
		t.Fatalf("unexpected stage error: %+v", stageErr)
	}
}

func TestExecutionSequenceTargetsPreservePlanOrder(t *testing.T) {
	plan, err := BuildBalancedExecutionPlan(
		ExecutionModel{ID: "test/model", CheckpointFingerprint: "expected", LayerCount: 8},
		testExecutionTargetIDs(4), StageResponseTensor,
	)
	if err != nil {
		t.Fatal(err)
	}
	workers := make([]*executionFakeWorker, 4)
	for index := range workers {
		workers[index] = newExecutionFakeWorker(fmt.Sprintf("worker-%d", index), "expected", 8)
	}
	targets := bindExecutionFakeTargets(plan, workers)
	bound, err := bindExecutionTargets(plan, targets)
	if err != nil {
		t.Fatal(err)
	}
	sequenceTargets := executionSequenceTargets(bound)
	if len(sequenceTargets) != len(targets) {
		t.Fatalf("sequence targets = %d, want %d", len(sequenceTargets), len(targets))
	}
	for index := range sequenceTargets {
		if sequenceTargets[index].Name != plan.Stages[index].Name ||
			sequenceTargets[index].ShardID != plan.Stages[index].ShardID ||
			sequenceTargets[index].Caller != workers[index] {
			t.Fatalf("sequence target %d mismatch: %+v", index, sequenceTargets[index])
		}
	}
}

func bindExecutionFakeTargets(plan ExecutionPlan, workers []*executionFakeWorker) []ExecutionTarget {
	targets := make([]ExecutionTarget, len(plan.Stages))
	for index, stage := range plan.Stages {
		targets[index] = ExecutionTarget{TargetID: stage.TargetID, Caller: workers[index]}
	}
	return targets
}

type executionInferenceCall struct {
	operation string
	inputKind string
	sample    bool
}

type executionFakeWorker struct {
	name           string
	fingerprint    string
	layerCount     int
	marker         byte
	loadCount      int
	loaded         []workerproc.PersistentShardSnapshot
	inferenceCalls []executionInferenceCall
	inferErr       error
}

func newExecutionFakeWorker(name, fingerprint string, layerCount int) *executionFakeWorker {
	return &executionFakeWorker{name: name, fingerprint: fingerprint, layerCount: layerCount, marker: 1}
}

func (worker *executionFakeWorker) Call(
	ctx context.Context,
	request workerproc.PersistentRequest,
) (workerproc.PersistentResponse, error) {
	if err := ctx.Err(); err != nil {
		return workerproc.PersistentResponse{}, err
	}
	result := &workerproc.PersistentWorkerResult{}
	switch request.Command {
	case "modelInfo":
		result.Model = &workerproc.PersistentModelResult{
			ModelID:               request.Model.ModelID,
			ModelType:             "test",
			LayerCount:            worker.layerCount,
			CheckpointFingerprint: worker.fingerprint,
			CheckpointBytes:       123,
		}
	case "state":
		result.State = &workerproc.PersistentWorkerState{LoadedShards: worker.loaded, LoadCount: worker.loadCount}
	case "loadShard":
		load := request.LoadShard
		snapshot := workerproc.PersistentShardSnapshot{
			ShardID:               load.ShardID,
			ModelID:               load.ModelID,
			CheckpointFingerprint: worker.fingerprint,
			LayerStart:            load.LayerStart,
			LayerEnd:              load.LayerEnd,
			OwnsInput:             load.OwnsInput,
			OwnsOutput:            load.OwnsOutput,
		}
		worker.loaded = append(worker.loaded, snapshot)
		worker.loadCount++
		result.Shard = &snapshot
	case "prefill", "decode":
		if worker.inferErr != nil {
			return workerproc.PersistentResponse{}, worker.inferErr
		}
		if request.Forward == nil {
			return workerproc.PersistentResponse{}, errors.New("missing forward request")
		}
		worker.inferenceCalls = append(worker.inferenceCalls, executionInferenceCall{
			operation: request.Command,
			inputKind: request.Forward.InputKind,
			sample:    request.Forward.ReturnSampledToken,
		})
		forward := &workerproc.PersistentForwardResult{
			ShardID:       request.Forward.ShardID,
			SequenceID:    request.Forward.SequenceID,
			Operation:     request.Command,
			Position:      request.Forward.Position,
			ComputeMicros: 1,
			KVCacheBytes:  1,
		}
		if request.Forward.ReturnSampledToken {
			token := int32(7)
			forward.SampledTokenID = &token
		} else {
			data := append([]byte(nil), request.Forward.Input.Data...)
			data = append(data, worker.marker)
			forward.Output = workerproc.WireTensor{
				Shape: []int{1, len(data)}, DType: "uint8", Data: data,
			}
		}
		if request.Command == "prefill" {
			forward.NextPosition = uint64(max(1, len(request.Forward.Input.Shape)))
		} else {
			forward.NextPosition = request.Forward.Position + 1
		}
		result.Forward = forward
	default:
		return workerproc.PersistentResponse{}, fmt.Errorf("unexpected command %q", request.Command)
	}
	return workerproc.PersistentResponse{OK: true, Result: result}, nil
}

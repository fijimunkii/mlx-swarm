package scaleproof

import (
	"fmt"
	"slices"
	"testing"

	"github.com/fijimunkii/mlx-swarm/internal/benchmark"
	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/mesh"
	"github.com/fijimunkii/mlx-swarm/internal/placement"
	"github.com/fijimunkii/mlx-swarm/internal/pooledproof"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

func TestHybridSyntheticSpecsAreCheckpointIncompatible(t *testing.T) {
	reference := pooledproof.Reference{
		ModelType: "test-adapter", CheckpointFingerprint: "real-checkpoint",
	}
	specs := hybridSyntheticSpecs(reference, DefaultSyntheticPeerCount)
	if len(specs) != DefaultSyntheticPeerCount {
		t.Fatalf("spec count = %d", len(specs))
	}
	for _, spec := range specs {
		registration := spec.Registration()
		if !slices.Contains(registration.Capabilities.Adapters, reference.ModelType) ||
			slices.Contains(
				registration.Capabilities.CheckpointFingerprints,
				reference.CheckpointFingerprint,
			) {
			t.Fatalf("synthetic registration is not narrowly incompatible: %+v", registration)
		}
	}
}

func TestHybridRequestUsesMeasuredFiveStageRanges(t *testing.T) {
	model := generation.ExecutionModel{
		ID: "test/model", CheckpointFingerprint: "test-checkpoint", LayerCount: 20,
	}
	plan, err := OwnershipAwarePlan(
		model, "explicit", []string{"a", "b", "c", "d", "e"}, 1,
	)
	if err != nil {
		t.Fatal(err)
	}
	pooled := RunEvidence{
		Plan: plan, StageLoads: make([]generation.StageLoad, RequiredNodeCount),
		Prefill: benchmark.PlannedSummary{Stages: make([]benchmark.PlannedStageSummary, RequiredNodeCount)},
		Decode:  benchmark.PlannedSummary{Stages: make([]benchmark.PlannedStageSummary, RequiredNodeCount)},
		All:     benchmark.PlannedSummary{Stages: make([]benchmark.PlannedStageSummary, RequiredNodeCount)},
	}
	for index, stage := range plan.Stages {
		pooled.StageLoads[index] = generation.StageLoad{
			Index: index, Stage: stage,
			Snapshot: workerproc.PersistentShardSnapshot{
				LoadedMemory: workerproc.StageMemory{ActiveBytes: 100 + index, CacheBytes: 10},
			},
		}
		pooled.Prefill.Stages[index] = measuredStage(stage, 1000, 400, 200)
		pooled.Decode.Stages[index] = measuredStage(stage, 100, 40, 20)
		pooled.All.Stages[index] = measuredStage(stage, 0, 0, 0)
		pooled.All.Stages[index].MaxKVCacheBytes = 50 + index
	}
	config := RunConfig{
		Coordinator: CoordinatorEvidence{ID: "coordinator", RunID: "run"},
		Reference: pooledproof.Reference{
			ModelType: "test-adapter", PromptTokenIDs: []int32{1, 2},
			GeneratedTokenIDs: []int32{3, 4, 5},
		},
	}
	request, err := hybridRequest(config, pooled)
	if err != nil {
		t.Fatal(err)
	}
	if request.MaxStages != RequiredNodeCount || len(request.Ranges) != RequiredNodeCount ||
		request.Scoring.Transport.Protocol != workerproc.InstanceBoundHTTPProtocol ||
		request.Scoring.Adapter != "test-adapter" {
		t.Fatalf("request = %+v", request)
	}
	for index, candidateRange := range request.Ranges {
		stage := plan.Stages[index]
		if candidateRange.LayerStart != stage.LayerStart || candidateRange.LayerEnd != stage.LayerEnd ||
			candidateRange.Estimate.LoadMemoryBytes != uint64(110+index) ||
			candidateRange.Estimate.SequenceMemoryBytes != uint64(50+index) {
			t.Fatalf("range %d = %+v", index, candidateRange)
		}
	}
	if request.Scoring.PrefillInputTokens != 2 || request.Scoring.DecodeSteps != 3 ||
		request.TerminalResponseMode != generation.StageResponseSampledToken ||
		request.MaxSearchOperations == 0 {
		t.Fatalf("request bounds = %+v", request)
	}
}

func measuredStage(
	stage generation.ExecutionStage,
	computeMicros, inputWireBytes, responseWireBytes int64,
) benchmark.PlannedStageSummary {
	return benchmark.PlannedStageSummary{
		Stage:             stage,
		ComputeMicros:     benchmark.Distribution{P50Micros: computeMicros},
		InputWireBytes:    benchmark.ByteDistribution{P50Bytes: inputWireBytes},
		ResponseWireBytes: benchmark.ByteDistribution{P50Bytes: responseWireBytes},
	}
}

func TestSelectedTargetsMustBeFiveDistinctRealWorkers(t *testing.T) {
	nodes := testNodes()
	selection := meshSelection(nodeIDs(nodes))
	if !selectedTargetsAreReal(selection, nodes) ||
		!selectedTargetsAreDistinct(selection, RequiredNodeCount) {
		t.Fatalf("selection = %+v", selection)
	}
	selection.Targets[4].WorkerID = "synthetic-linux-00"
	if selectedTargetsAreReal(selection, nodes) {
		t.Fatal("synthetic target was accepted as real")
	}
	selection.Targets[4].WorkerID = selection.Targets[0].WorkerID
	if selectedTargetsAreDistinct(selection, RequiredNodeCount) {
		t.Fatal("duplicate real target was accepted")
	}
}

func TestHybridSyntheticRejectionsRequireEveryRangeToReject(t *testing.T) {
	specs := hybridSyntheticSpecs(pooledproof.Reference{
		ModelType: "adapter", CheckpointFingerprint: "checkpoint",
	}, 1)
	selection := mesh.SequenceSelection{Construction: placement.PlanConstructionResult{
		Ranges: []placement.RangePlanEvaluation{{Eligibility: placement.Evaluation{
			Candidates: []placement.Candidate{{
				WorkerID: specs[0].ID, Rejections: []placement.Rejection{{
					Code: placement.RejectionIncompatibleCheckpoint,
				}},
			}},
		}}},
	}}
	rejections, allRejected := hybridSyntheticRejections(selection, specs)
	if !allRejected || len(rejections) != 1 {
		t.Fatalf("rejections = %+v, all = %t", rejections, allRejected)
	}
	selection.Construction.Ranges = append(
		selection.Construction.Ranges,
		placement.RangePlanEvaluation{Eligibility: placement.Evaluation{
			Candidates: []placement.Candidate{{WorkerID: specs[0].ID, Eligible: true}},
		}},
	)
	_, allRejected = hybridSyntheticRejections(selection, specs)
	if allRejected {
		t.Fatal("synthetic worker eligible on one range was reported as fully rejected")
	}
}

func TestPostRunObservationsMustBeOrderedAndReleased(t *testing.T) {
	workers := make([]HybridWorkerObservation, RequiredNodeCount)
	for index := range workers {
		workers[index] = HybridWorkerObservation{
			WorkerID:                     fmt.Sprintf("worker-%d", index),
			InventoryObservationSequence: 1, WorkerObservationSequence: 2,
		}
	}
	if !postRunObservationsOrdered(workers) {
		t.Fatal("clean ordered observations were rejected")
	}
	workers[0].WorkerObservationSequence = workers[0].InventoryObservationSequence
	if postRunObservationsOrdered(workers) {
		t.Fatal("non-newer observation was accepted")
	}
	workers[0].WorkerObservationSequence = 2
	workers[1].RetainedBytes = 1
	if postRunObservationsOrdered(workers) {
		t.Fatal("retained sequence state was accepted")
	}
}

func meshSelection(ids []string) mesh.SequenceSelection {
	selection := mesh.SequenceSelection{Targets: make([]mesh.TargetBinding, len(ids))}
	for index, id := range ids {
		selection.Targets[index] = mesh.TargetBinding{Index: index, WorkerID: id}
	}
	return selection
}

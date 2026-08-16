package scaleproof

import (
	"reflect"
	"strings"
	"testing"

	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

func TestOwnershipAwarePlanReservesEdgeCapacity(t *testing.T) {
	plan, err := OwnershipAwarePlan(generation.ExecutionModel{
		ID: "model", CheckpointFingerprint: "checkpoint", LayerCount: 48,
	}, "inventory-1", []string{"a", "b", "c", "d", "e"}, 2)
	if err != nil {
		t.Fatal(err)
	}
	got := make([][2]int, len(plan.Stages))
	for index, stage := range plan.Stages {
		got[index] = [2]int{stage.LayerStart, stage.LayerEnd}
	}
	want := [][2]int{{0, 9}, {9, 20}, {20, 30}, {30, 40}, {40, 48}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ranges = %v, want %v", got, want)
	}
	if plan.InventoryRevision != "inventory-1" || plan.Revision == "" {
		t.Fatalf("plan identity is incomplete: %+v", plan)
	}
	if !plan.Stages[0].OwnsInput || plan.Stages[0].OwnsOutput ||
		plan.Stages[4].OwnsInput || !plan.Stages[4].OwnsOutput ||
		plan.Stages[4].ResponseMode != generation.StageResponseSampledToken {
		t.Fatalf("edge ownership is invalid: %+v", plan.Stages)
	}
}

func TestComplementaryLoadsRejectsAFullModelSnapshot(t *testing.T) {
	plan, err := OwnershipAwarePlan(generation.ExecutionModel{
		ID: "model", CheckpointFingerprint: "checkpoint", LayerCount: 48,
	}, "inventory-1", []string{"a", "b", "c", "d", "e"}, 2)
	if err != nil {
		t.Fatal(err)
	}
	loads := make([]generation.StageLoad, len(plan.Stages))
	for index, stage := range plan.Stages {
		loads[index] = generation.StageLoad{
			Index: index, Stage: stage,
			Snapshot: workerproc.PersistentShardSnapshot{
				ShardID: stage.ShardID, ModelID: plan.Model.ID,
				CheckpointFingerprint: plan.Model.CheckpointFingerprint,
				LayerStart:            stage.LayerStart, LayerEnd: stage.LayerEnd,
				OwnsInput: stage.OwnsInput, OwnsOutput: stage.OwnsOutput,
			},
		}
	}
	if !complementaryLoads(plan, loads) {
		t.Fatal("valid complementary snapshots were rejected")
	}
	loads[2].Snapshot.LayerStart = 0
	loads[2].Snapshot.LayerEnd = plan.Model.LayerCount
	if complementaryLoads(plan, loads) {
		t.Fatal("full-model serving snapshot was accepted")
	}
}

func TestOwnershipAwarePlanRejectsExcessiveReserve(t *testing.T) {
	_, err := OwnershipAwarePlan(generation.ExecutionModel{
		ID: "model", CheckpointFingerprint: "checkpoint", LayerCount: 5,
	}, "", []string{"a", "b", "c", "d", "e"}, 10)
	if err == nil || !strings.Contains(err.Error(), "edge reserve") {
		t.Fatalf("error = %v, want edge-reserve rejection", err)
	}
}

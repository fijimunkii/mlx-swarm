package generation

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func TestBuildBalancedExecutionPlanFiveStages(t *testing.T) {
	plan, err := BuildBalancedExecutionPlan(
		testExecutionModel(18), testExecutionTargetIDs(5), StageResponseTensor,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Stages) != 5 {
		t.Fatalf("stage count = %d, want 5", len(plan.Stages))
	}
	gotRanges := make([][2]int, 0, len(plan.Stages))
	for index, stage := range plan.Stages {
		gotRanges = append(gotRanges, [2]int{stage.LayerStart, stage.LayerEnd})
		if stage.OwnsInput != (index == 0) {
			t.Fatalf("stage %d input ownership = %t", index, stage.OwnsInput)
		}
		if stage.OwnsOutput != (index == len(plan.Stages)-1) {
			t.Fatalf("stage %d output ownership = %t", index, stage.OwnsOutput)
		}
		if stage.Name == "" || stage.ShardID == "" {
			t.Fatalf("stage %d missing stable identity: %+v", index, stage)
		}
	}
	wantRanges := [][2]int{{0, 4}, {4, 8}, {8, 12}, {12, 15}, {15, 18}}
	if !reflect.DeepEqual(gotRanges, wantRanges) {
		t.Fatalf("ranges = %v, want %v", gotRanges, wantRanges)
	}
	if err := ValidateExecutionPlan(plan); err != nil {
		t.Fatalf("validate built plan: %v", err)
	}
}

func TestBuildExecutionPlanSupportsExplicitOwnershipAwareRanges(t *testing.T) {
	plan, err := BuildExecutionPlan(
		testExecutionModel(48), "inventory-7",
		[]ExecutionStage{
			{Name: "stage-0", TargetID: "worker-0", ShardID: "ignored", LayerStart: 0, LayerEnd: 8, OwnsInput: true, ResponseMode: StageResponseTensor},
			{Name: "stage-1", TargetID: "worker-1", LayerStart: 8, LayerEnd: 19, ResponseMode: StageResponseTensor},
			{Name: "stage-2", TargetID: "worker-2", LayerStart: 19, LayerEnd: 29, ResponseMode: StageResponseTensor},
			{Name: "stage-3", TargetID: "worker-3", LayerStart: 29, LayerEnd: 40, ResponseMode: StageResponseTensor},
			{Name: "stage-4", TargetID: "worker-4", LayerStart: 40, LayerEnd: 48, OwnsOutput: true, ResponseMode: StageResponseSampledToken},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if plan.InventoryRevision != "inventory-7" || plan.SchemaVersion == "" || plan.Revision == "" {
		t.Fatalf("explicit plan omitted immutable identity: %+v", plan)
	}
	if plan.Stages[0].ShardID == "ignored" || plan.Stages[0].ShardID == "" {
		t.Fatalf("explicit plan did not derive its shard identity: %+v", plan.Stages[0])
	}
	if got := [][2]int{
		{plan.Stages[0].LayerStart, plan.Stages[0].LayerEnd},
		{plan.Stages[1].LayerStart, plan.Stages[1].LayerEnd},
		{plan.Stages[2].LayerStart, plan.Stages[2].LayerEnd},
		{plan.Stages[3].LayerStart, plan.Stages[3].LayerEnd},
		{plan.Stages[4].LayerStart, plan.Stages[4].LayerEnd},
	}; !reflect.DeepEqual(got, [][2]int{{0, 8}, {8, 19}, {19, 29}, {29, 40}, {40, 48}}) {
		t.Fatalf("explicit ranges = %v", got)
	}
}

func TestBuildBalancedExecutionPlanIsDeterministic(t *testing.T) {
	first, err := BuildBalancedExecutionPlan(
		testExecutionModel(18), testExecutionTargetIDs(5), StageResponseTensor,
	)
	if err != nil {
		t.Fatal(err)
	}
	second, err := BuildBalancedExecutionPlan(
		testExecutionModel(18), testExecutionTargetIDs(5), StageResponseTensor,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("balanced plans differ:\nfirst=%+v\nsecond=%+v", first, second)
	}
}

func TestBuildBalancedExecutionPlanSeparatesDifferentTopologies(t *testing.T) {
	two, err := BuildBalancedExecutionPlan(
		testExecutionModel(18), testExecutionTargetIDs(2), StageResponseTensor,
	)
	if err != nil {
		t.Fatal(err)
	}
	five, err := BuildBalancedExecutionPlan(
		testExecutionModel(18), testExecutionTargetIDs(5), StageResponseTensor,
	)
	if err != nil {
		t.Fatal(err)
	}
	if two.Revision == five.Revision {
		t.Fatal("different stage topologies produced the same plan revision")
	}
	if two.Stages[0].ShardID == five.Stages[0].ShardID {
		t.Fatalf("different stage ranges reused shard ID %q", two.Stages[0].ShardID)
	}
}

func TestExecutionShardIdentitySurvivesInventoryRevisionChanges(t *testing.T) {
	stages := []ExecutionStage{
		{Name: "stage-0", TargetID: "worker-0", LayerStart: 0, LayerEnd: 9, OwnsInput: true, ResponseMode: StageResponseTensor},
		{Name: "stage-1", TargetID: "worker-1", LayerStart: 9, LayerEnd: 18, OwnsOutput: true, ResponseMode: StageResponseSampledToken},
	}
	first, err := BuildExecutionPlan(testExecutionModel(18), "7", stages)
	if err != nil {
		t.Fatal(err)
	}
	second, err := BuildExecutionPlan(testExecutionModel(18), "8", stages)
	if err != nil {
		t.Fatal(err)
	}
	if first.Revision == second.Revision {
		t.Fatal("inventory revision did not change plan identity")
	}
	for index := range first.Stages {
		if first.Stages[index].ShardID != second.Stages[index].ShardID {
			t.Fatalf(
				"stage %d shard identity changed across inventory revisions: %q != %q",
				index, first.Stages[index].ShardID, second.Stages[index].ShardID,
			)
		}
	}
}

func TestDeriveExecutionShardIDValidatesLoadShape(t *testing.T) {
	model := testExecutionModel(18)
	derived, err := DeriveExecutionShardID(model, 0, 9, true, false)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := BuildExecutionPlan(model, "7", []ExecutionStage{
		{Name: "stage-0", TargetID: "worker-0", LayerStart: 0, LayerEnd: 9, OwnsInput: true, ResponseMode: StageResponseTensor},
		{Name: "stage-1", TargetID: "worker-1", LayerStart: 9, LayerEnd: 18, OwnsOutput: true, ResponseMode: StageResponseTensor},
	})
	if err != nil {
		t.Fatal(err)
	}
	if derived != plan.Stages[0].ShardID {
		t.Fatalf("derived shard ID = %q, plan has %q", derived, plan.Stages[0].ShardID)
	}
	for _, test := range []struct {
		name       string
		model      ExecutionModel
		start, end int
		input      bool
		output     bool
	}{
		{name: "model", model: ExecutionModel{}, start: 0, end: 1, input: true},
		{name: "range", model: model, start: 9, end: 9},
		{name: "input ownership", model: model, start: 0, end: 9},
		{name: "output ownership", model: model, start: 9, end: 18},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := DeriveExecutionShardID(
				test.model, test.start, test.end, test.input, test.output,
			); err == nil {
				t.Fatal("invalid execution shard shape was accepted")
			}
		})
	}
}

func TestExecutionPlanSerializationPinsIdentityAndResponseMode(t *testing.T) {
	plan, err := BuildBalancedExecutionPlan(
		testExecutionModel(18), testExecutionTargetIDs(5), StageResponseSampledToken,
	)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(plan)
	if err != nil {
		t.Fatal(err)
	}
	var decoded ExecutionPlan
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded, plan) {
		t.Fatalf("serialized plan changed:\nwant=%+v\ngot=%+v", plan, decoded)
	}
	if decoded.Model.ID == "" || decoded.Model.CheckpointFingerprint == "" ||
		decoded.Revision == "" || decoded.Stages[4].TargetID == "" ||
		decoded.Stages[4].ResponseMode != StageResponseSampledToken {
		t.Fatalf("serialized plan omitted required identity: %+v", decoded)
	}
}

func TestValidateExecutionPlanRejectsMutatedRevision(t *testing.T) {
	plan, err := BuildBalancedExecutionPlan(
		testExecutionModel(18), testExecutionTargetIDs(5), StageResponseTensor,
	)
	if err != nil {
		t.Fatal(err)
	}
	plan.Stages[2].TargetID = "replacement-worker"
	err = ValidateExecutionPlan(plan)
	if err == nil || !strings.Contains(err.Error(), "revision") {
		t.Fatalf("error = %v, want revision mismatch", err)
	}
}

func TestBuildBalancedExecutionPlanRejectsInvalidCounts(t *testing.T) {
	for _, test := range []struct {
		name       string
		layers     int
		stages     int
		wantSubstr string
	}{
		{name: "zero layers", layers: 0, stages: 1, wantSubstr: "layer count"},
		{name: "zero stages", layers: 2, stages: 0, wantSubstr: "execution targets"},
		{name: "more stages than layers", layers: 2, stages: 3, wantSubstr: "exceeds layer count"},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := BuildBalancedExecutionPlan(
				testExecutionModel(test.layers),
				testExecutionTargetIDs(test.stages),
				StageResponseTensor,
			)
			if err == nil || !strings.Contains(err.Error(), test.wantSubstr) {
				t.Fatalf("error = %v, want substring %q", err, test.wantSubstr)
			}
		})
	}
}

func TestValidateExecutionPlanRejectsInvalidTopology(t *testing.T) {
	valid, err := BuildBalancedExecutionPlan(
		testExecutionModel(8), testExecutionTargetIDs(4), StageResponseTensor,
	)
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range []struct {
		name       string
		mutate     func(*ExecutionPlan)
		wantSubstr string
	}{
		{
			name: "gap",
			mutate: func(plan *ExecutionPlan) {
				plan.Stages[1].LayerStart++
			},
			wantSubstr: "expected",
		},
		{
			name: "overlap",
			mutate: func(plan *ExecutionPlan) {
				plan.Stages[1].LayerStart--
			},
			wantSubstr: "expected",
		},
		{
			name: "missing final layer",
			mutate: func(plan *ExecutionPlan) {
				plan.Stages[len(plan.Stages)-1].LayerEnd--
			},
			wantSubstr: "ends at layer",
		},
		{
			name: "input owner in middle",
			mutate: func(plan *ExecutionPlan) {
				plan.Stages[1].OwnsInput = true
			},
			wantSubstr: "input ownership",
		},
		{
			name: "missing output owner",
			mutate: func(plan *ExecutionPlan) {
				plan.Stages[len(plan.Stages)-1].OwnsOutput = false
			},
			wantSubstr: "output ownership",
		},
		{
			name: "duplicate output owner",
			mutate: func(plan *ExecutionPlan) {
				plan.Stages[1].OwnsOutput = true
			},
			wantSubstr: "output ownership",
		},
		{
			name: "terminal sampling before final stage",
			mutate: func(plan *ExecutionPlan) {
				plan.Stages[1].ResponseMode = StageResponseSampledToken
			},
			wantSubstr: "intermediate stages",
		},
		{
			name: "invalid stage ordering",
			mutate: func(plan *ExecutionPlan) {
				plan.Stages[0], plan.Stages[1] = plan.Stages[1], plan.Stages[0]
			},
			wantSubstr: "expected",
		},
		{
			name: "duplicate name",
			mutate: func(plan *ExecutionPlan) {
				plan.Stages[1].Name = plan.Stages[0].Name
			},
			wantSubstr: "duplicated",
		},
		{
			name: "duplicate target",
			mutate: func(plan *ExecutionPlan) {
				plan.Stages[1].TargetID = plan.Stages[0].TargetID
			},
			wantSubstr: "duplicated",
		},
		{
			name: "duplicate shard",
			mutate: func(plan *ExecutionPlan) {
				plan.Stages[1].ShardID = plan.Stages[0].ShardID
			},
			wantSubstr: "duplicated",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			plan := valid
			plan.Stages = append([]ExecutionStage(nil), valid.Stages...)
			test.mutate(&plan)
			err := ValidateExecutionPlan(plan)
			if err == nil || !strings.Contains(err.Error(), test.wantSubstr) {
				t.Fatalf("error = %v, want substring %q", err, test.wantSubstr)
			}
		})
	}
}

func TestExecutionPlanLegacyShardPlan(t *testing.T) {
	plan, err := BuildBalancedExecutionPlan(
		testExecutionModel(18), testExecutionTargetIDs(2), StageResponseTensor,
	)
	if err != nil {
		t.Fatal(err)
	}
	legacy, ok := plan.LegacyShardPlan()
	if !ok {
		t.Fatal("two-stage plan did not expose legacy shard plan")
	}
	if legacy.Producer.LayerStart != 0 || legacy.Producer.LayerEnd != 9 ||
		legacy.Consumer.LayerStart != 9 || legacy.Consumer.LayerEnd != 18 {
		t.Fatalf("unexpected legacy plan: %+v", legacy)
	}

	five, err := BuildBalancedExecutionPlan(
		testExecutionModel(18), testExecutionTargetIDs(5), StageResponseTensor,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := five.LegacyShardPlan(); ok {
		t.Fatal("five-stage plan unexpectedly exposed legacy producer/consumer shape")
	}
}

func testExecutionModel(layerCount int) ExecutionModel {
	return ExecutionModel{
		ID: "test/model", CheckpointFingerprint: "fingerprint", LayerCount: layerCount,
	}
}

func testExecutionTargetIDs(count int) []string {
	if count <= 0 {
		return nil
	}
	result := make([]string, count)
	for index := range result {
		result[index] = fmt.Sprintf("worker-%d", index)
	}
	return result
}

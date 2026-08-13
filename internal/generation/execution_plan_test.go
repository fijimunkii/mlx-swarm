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

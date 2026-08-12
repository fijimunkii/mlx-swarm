package generation

import (
	"reflect"
	"strings"
	"testing"
)

func TestBuildBalancedExecutionPlanFiveStages(t *testing.T) {
	plan, err := BuildBalancedExecutionPlan("test/model", "test-checkpoint", 18, 5)
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
	if err := ValidateExecutionPlan(plan, 18); err != nil {
		t.Fatalf("validate built plan: %v", err)
	}
}

func TestBuildBalancedExecutionPlanIsDeterministic(t *testing.T) {
	first, err := BuildBalancedExecutionPlan("test/model", "fingerprint", 18, 5)
	if err != nil {
		t.Fatal(err)
	}
	second, err := BuildBalancedExecutionPlan("test/model", "fingerprint", 18, 5)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("balanced plans differ:\nfirst=%+v\nsecond=%+v", first, second)
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
		{name: "zero stages", layers: 2, stages: 0, wantSubstr: "stage count"},
		{name: "more stages than layers", layers: 2, stages: 3, wantSubstr: "exceeds layer count"},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := BuildBalancedExecutionPlan("test/model", "fingerprint", test.layers, test.stages)
			if err == nil || !strings.Contains(err.Error(), test.wantSubstr) {
				t.Fatalf("error = %v, want substring %q", err, test.wantSubstr)
			}
		})
	}
}

func TestValidateExecutionPlanRejectsInvalidTopology(t *testing.T) {
	valid, err := BuildBalancedExecutionPlan("test/model", "fingerprint", 8, 4)
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
			name: "duplicate name",
			mutate: func(plan *ExecutionPlan) {
				plan.Stages[1].Name = plan.Stages[0].Name
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
			plan := ExecutionPlan{Stages: append([]ExecutionStage(nil), valid.Stages...)}
			test.mutate(&plan)
			err := ValidateExecutionPlan(plan, 8)
			if err == nil || !strings.Contains(err.Error(), test.wantSubstr) {
				t.Fatalf("error = %v, want substring %q", err, test.wantSubstr)
			}
		})
	}
}

func TestExecutionPlanLegacyShardPlan(t *testing.T) {
	plan, err := BuildBalancedExecutionPlan("test/model", "fingerprint", 18, 2)
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

	five, err := BuildBalancedExecutionPlan("test/model", "fingerprint", 18, 5)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := five.LegacyShardPlan(); ok {
		t.Fatal("five-stage plan unexpectedly exposed legacy producer/consumer shape")
	}
}

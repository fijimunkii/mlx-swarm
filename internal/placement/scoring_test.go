package placement

import (
	"reflect"
	"slices"
	"testing"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/registry"
)

func TestScorePlanUsesExactProfilesAndPreservesEligibility(t *testing.T) {
	now := time.Date(2026, time.August, 17, 12, 0, 0, 0, time.UTC)
	plan, inventory, profile, request := testScoringFixture(t, now)

	evaluation, err := ScorePlan(inventory, profile, plan, request)
	if err != nil {
		t.Fatal(err)
	}
	if !evaluation.Eligible || evaluation.SchemaVersion != SchemaVersion ||
		evaluation.InventoryRevision != 7 || evaluation.ProfileRevision != 6 ||
		evaluation.ProfileMaxAgeMillis != time.Minute.Milliseconds() ||
		!evaluation.GeneratedAt.Equal(now) || evaluation.Plan.Revision != plan.Revision {
		t.Fatalf("unexpected plan evaluation metadata: %+v", evaluation)
	}
	if !reflect.DeepEqual(evaluation.Request, request) {
		t.Fatalf("recorded scoring request = %+v, want %+v", evaluation.Request, request)
	}
	if len(evaluation.Stages) != 2 {
		t.Fatalf("stage evaluation count = %d", len(evaluation.Stages))
	}
	for index, stage := range evaluation.Stages {
		if stage.Index != index || len(stage.Eligibility.Candidates) != 2 {
			t.Fatalf("stage %d did not preserve candidate evidence: %+v", index, stage)
		}
		if got := []string{
			stage.Eligibility.Candidates[0].WorkerID,
			stage.Eligibility.Candidates[1].WorkerID,
		}; !slices.Equal(got, []string{"worker-a", "worker-b"}) {
			t.Fatalf("stage %d candidate order = %v", index, got)
		}
		if stage.SelectedCandidate.WorkerID != plan.Stages[index].TargetID ||
			!stage.SelectedCandidate.Eligible {
			t.Fatalf("stage %d selected candidate = %+v", index, stage.SelectedCandidate)
		}
	}
	if got := evaluation.Stages[0].Compute; got.PrefillMicros != 200 ||
		got.DecodeMicros != 30 || got.TotalMicros != 230 ||
		!got.PrefillProfiled || !got.DecodeProfiled || got.PrefillProfile == nil ||
		got.DecodeProfile == nil || got.PrefillProfile.WorkerID != "worker-b" ||
		got.DecodeProfile.Operation != "decode" {
		t.Fatalf("stage 0 compute score = %+v", got)
	}
	if got := evaluation.Stages[1].Compute; got.PrefillMicros != 400 ||
		got.DecodeMicros != 60 || got.TotalMicros != 460 ||
		!got.PrefillProfiled || !got.DecodeProfiled || got.PrefillProfile == nil ||
		got.DecodeProfile == nil || got.PrefillProfile.WorkerID != "worker-a" ||
		got.DecodeProfile.Operation != "decode" {
		t.Fatalf("stage 1 compute score = %+v", got)
	}
	if got := evaluation.Stages[0].Transfer; got.RTTMicros != 10 ||
		got.BytesPerSecond != 10_000_000 || got.PrefillMicros != 20 ||
		got.DecodeMicros != 33 || got.TotalMicros != 53 || !got.RTTProfiled ||
		!got.BandwidthProfiled || got.LinkProfile == nil ||
		got.LinkProfile.TargetID != "worker-b" {
		t.Fatalf("stage 0 transfer score = %+v", got)
	}
	if got := evaluation.Stages[1].Transfer; got.RTTMicros != 20 ||
		got.BytesPerSecond != 20_000_000 || got.PrefillMicros != 30 ||
		got.DecodeMicros != 63 || got.TotalMicros != 93 || !got.RTTProfiled ||
		!got.BandwidthProfiled || got.LinkProfile == nil ||
		got.LinkProfile.TargetID != "worker-a" {
		t.Fatalf("stage 1 transfer score = %+v", got)
	}
	wantScore := PlanScore{
		StageCount: 2, EstimatedMicros: 836, ComputeMicros: 690, TransferMicros: 146,
		RecentFailureCount: 3, RestartCount: 7, MemoryPressureBytes: 12,
		RequiredAdditionalMemoryBytes: 1300, NewLoadMemoryBytes: 1000,
		ReusedStageCount: 1, ProfiledComputeOperationCount: 4,
		ProfiledRTTStageCount: 2, ProfiledBandwidthStageCount: 2,
	}
	if evaluation.Score != wantScore {
		t.Fatalf("plan score = %+v, want %+v", evaluation.Score, wantScore)
	}
	if !evaluation.Stages[1].SelectedCandidate.ReusesRetainedShard {
		t.Fatal("exact selected shard was not scored as retained reuse")
	}

	reversedInventory := inventory
	reversedInventory.Workers = append([]registry.Worker(nil), inventory.Workers...)
	slices.Reverse(reversedInventory.Workers)
	reversedProfile := profile
	reversedProfile.Links = append([]LinkProfile(nil), profile.Links...)
	reversedProfile.Compute = append([]ComputeProfile(nil), profile.Compute...)
	slices.Reverse(reversedProfile.Links)
	slices.Reverse(reversedProfile.Compute)
	repeated, err := ScorePlan(reversedInventory, reversedProfile, plan, request)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(evaluation, repeated) {
		t.Fatalf("score depends on inventory or profile order:\nfirst=%+v\nsecond=%+v", evaluation, repeated)
	}
}

func TestScorePlanUsesExplicitFallbacksWithoutExactEvidence(t *testing.T) {
	now := time.Date(2026, time.August, 17, 12, 0, 0, 0, time.UTC)
	plan, inventory, _, request := testScoringFixture(t, now)
	profile := mustProfileSnapshot(t, mustProfileStore(t, ProfileConfig{
		MaxAge: time.Minute, MaxSeries: 16,
	}), now)

	evaluation, err := ScorePlan(inventory, profile, plan, request)
	if err != nil {
		t.Fatal(err)
	}
	if !evaluation.Eligible || evaluation.Score.ComputeMicros != 1430 ||
		evaluation.Score.TransferMicros != 1190 || evaluation.Score.EstimatedMicros != 2620 ||
		evaluation.Score.ProfiledComputeOperationCount != 0 ||
		evaluation.Score.ProfiledRTTStageCount != 0 ||
		evaluation.Score.ProfiledBandwidthStageCount != 0 {
		t.Fatalf("fallback score = %+v", evaluation.Score)
	}
	if got := evaluation.Stages[0].Compute; got.PrefillMicros != 500 ||
		got.DecodeMicros != 150 || got.PrefillProfiled || got.DecodeProfiled ||
		got.PrefillProfile != nil || got.DecodeProfile != nil {
		t.Fatalf("stage 0 fallback compute = %+v", got)
	}
	if got := evaluation.Stages[0].Transfer; got.RTTMicros != 100 ||
		got.BytesPerSecond != 1_000_000 || got.TotalMicros != 530 ||
		got.RTTProfiled || got.BandwidthProfiled || got.LinkProfile != nil {
		t.Fatalf("stage 0 fallback transfer = %+v", got)
	}
}

func TestScorePlanDoesNotCarryProfilesAcrossWorkerRestart(t *testing.T) {
	now := time.Date(2026, time.August, 17, 12, 0, 0, 0, time.UTC)
	plan, inventory, profile, request := testScoringFixture(t, now)
	for index := range inventory.Workers {
		if inventory.Workers[index].ID == "worker-b" {
			inventory.Workers[index].InstanceID = "worker-b-restarted"
		}
	}

	evaluation, err := ScorePlan(inventory, profile, plan, request)
	if err != nil {
		t.Fatal(err)
	}
	stage := evaluation.Stages[0]
	if stage.Compute.PrefillProfiled || stage.Compute.DecodeProfiled ||
		stage.Transfer.RTTProfiled || stage.Transfer.BandwidthProfiled ||
		stage.Compute.PrefillProfile != nil || stage.Compute.DecodeProfile != nil ||
		stage.Transfer.LinkProfile != nil || stage.Compute.TotalMicros != 650 ||
		stage.Transfer.TotalMicros != 530 {
		t.Fatalf("old process evidence survived restart: %+v", stage)
	}
}

func TestScorePlanReturnsCompleteEvidenceForIneligiblePlan(t *testing.T) {
	now := time.Date(2026, time.August, 17, 12, 0, 0, 0, time.UTC)
	plan, inventory, profile, request := testScoringFixture(t, now)
	for index := range inventory.Workers {
		if inventory.Workers[index].ID == "worker-b" {
			inventory.Workers[index].Status.Health = registry.HealthDegraded
		}
	}

	evaluation, err := ScorePlan(inventory, profile, plan, request)
	if err != nil {
		t.Fatal(err)
	}
	if evaluation.Eligible || len(evaluation.Stages) != 2 || evaluation.Score.StageCount != 2 ||
		evaluation.Score.EstimatedMicros != 0 {
		t.Fatalf("unexpected ineligible evaluation: %+v", evaluation)
	}
	selected := evaluation.Stages[0].SelectedCandidate
	if selected.Eligible || !slices.Equal(rejectionCodes(selected), []RejectionCode{RejectionUnhealthy}) {
		t.Fatalf("selected rejection evidence = %+v", selected)
	}
	if !evaluation.Stages[1].SelectedCandidate.Eligible {
		t.Fatalf("later stage evidence was not evaluated: %+v", evaluation.Stages[1])
	}
}

func TestComparePlanEvaluationsUsesStablePriorityOrder(t *testing.T) {
	base := PlanEvaluation{
		Eligible: true, Plan: generation.ExecutionPlan{
			Revision: "plan-b", Stages: []generation.ExecutionStage{{TargetID: "worker-b"}},
		},
		Score: PlanScore{StageCount: 2, EstimatedMicros: 100},
	}
	tests := []struct {
		name  string
		left  PlanEvaluation
		right PlanEvaluation
	}{
		{name: "eligibility", left: base, right: mutateEvaluation(base, func(value *PlanEvaluation) {
			value.Eligible = false
		})},
		{name: "latency", left: base, right: mutateEvaluation(base, func(value *PlanEvaluation) {
			value.Score.EstimatedMicros++
		})},
		{name: "failures", left: base, right: mutateEvaluation(base, func(value *PlanEvaluation) {
			value.Score.RecentFailureCount++
		})},
		{name: "pressure", left: base, right: mutateEvaluation(base, func(value *PlanEvaluation) {
			value.Score.MemoryPressureBytes++
		})},
		{name: "new load", left: base, right: mutateEvaluation(base, func(value *PlanEvaluation) {
			value.Score.NewLoadMemoryBytes++
		})},
		{name: "additional memory", left: base, right: mutateEvaluation(base, func(value *PlanEvaluation) {
			value.Score.RequiredAdditionalMemoryBytes++
		})},
		{name: "restarts", left: base, right: mutateEvaluation(base, func(value *PlanEvaluation) {
			value.Score.RestartCount++
		})},
		{name: "stage count", left: base, right: mutateEvaluation(base, func(value *PlanEvaluation) {
			value.Score.StageCount++
		})},
		{name: "plan topology", left: mutateEvaluation(base, func(value *PlanEvaluation) {
			value.Plan.Stages[0].TargetID = "worker-a"
		}), right: base},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := ComparePlanEvaluations(test.left, test.right); got >= 0 {
				t.Fatalf("left was not preferred: %d", got)
			}
			if got := ComparePlanEvaluations(test.right, test.left); got <= 0 {
				t.Fatalf("right was not disfavored: %d", got)
			}
		})
	}
	if got := ComparePlanEvaluations(base, base); got != 0 {
		t.Fatalf("identical evaluations compare as %d", got)
	}
}

func TestScorePlanRejectsIncoherentInputs(t *testing.T) {
	now := time.Date(2026, time.August, 17, 12, 0, 0, 0, time.UTC)
	validPlan, validInventory, validProfile, validRequest := testScoringFixture(t, now)
	tests := []struct {
		name   string
		mutate func(*registry.Inventory, *ProfileSnapshot, *generation.ExecutionPlan, *PlanScoringRequest)
	}{
		{name: "inventory revision", mutate: func(_ *registry.Inventory, _ *ProfileSnapshot, plan *generation.ExecutionPlan, _ *PlanScoringRequest) {
			plan.InventoryRevision = "8"
		}},
		{name: "duplicate worker", mutate: func(inventory *registry.Inventory, _ *ProfileSnapshot, _ *generation.ExecutionPlan, _ *PlanScoringRequest) {
			inventory.Workers = append(inventory.Workers, inventory.Workers[0])
		}},
		{name: "profile generation time", mutate: func(_ *registry.Inventory, profile *ProfileSnapshot, _ *generation.ExecutionPlan, _ *PlanScoringRequest) {
			profile.GeneratedAt = profile.GeneratedAt.Add(time.Millisecond)
		}},
		{name: "negative status age", mutate: func(_ *registry.Inventory, _ *ProfileSnapshot, _ *generation.ExecutionPlan, request *PlanScoringRequest) {
			request.StatusMaxAgeMillis = -1
		}},
		{name: "missing workload", mutate: func(_ *registry.Inventory, _ *ProfileSnapshot, _ *generation.ExecutionPlan, request *PlanScoringRequest) {
			request.DecodeSteps = 0
		}},
		{name: "missing fallback", mutate: func(_ *registry.Inventory, _ *ProfileSnapshot, _ *generation.ExecutionPlan, request *PlanScoringRequest) {
			request.FallbackBytesPerSecond = 0
		}},
		{name: "stage estimates", mutate: func(_ *registry.Inventory, _ *ProfileSnapshot, _ *generation.ExecutionPlan, request *PlanScoringRequest) {
			request.Stages = request.Stages[:1]
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			inventory := validInventory
			inventory.Workers = append([]registry.Worker(nil), validInventory.Workers...)
			profile := validProfile
			plan := validPlan
			request := validRequest
			request.Stages = append([]StageCostEstimate(nil), validRequest.Stages...)
			test.mutate(&inventory, &profile, &plan, &request)
			if _, err := ScorePlan(inventory, profile, plan, request); err == nil {
				t.Fatal("incoherent scoring input was accepted")
			}
		})
	}
}

func testScoringFixture(
	t *testing.T,
	now time.Time,
) (generation.ExecutionPlan, registry.Inventory, ProfileSnapshot, PlanScoringRequest) {
	t.Helper()
	plan := testProfilePlan(t)
	workerA := testWorker("worker-a", now)
	workerA.Status.MemoryPressureBytes = 7
	workerA.Status.RecentFailureCount = 2
	workerA.Status.RestartCount = 4
	workerA.Status.RetainedBytes = 100
	workerA.Status.RetainedShards = []registry.RetainedShard{{
		ID: plan.Stages[1].ShardID, ModelID: plan.Model.ID,
		CheckpointFingerprint: plan.Model.CheckpointFingerprint,
		LayerStart:            plan.Stages[1].LayerStart, LayerEnd: plan.Stages[1].LayerEnd,
		OwnsInput: plan.Stages[1].OwnsInput, OwnsOutput: plan.Stages[1].OwnsOutput,
		MemoryBytes: 2000,
	}}
	workerB := testWorker("worker-b", now)
	workerB.Status.MemoryPressureBytes = 5
	workerB.Status.RecentFailureCount = 1
	workerB.Status.RestartCount = 3
	inventory := testInventory(now, workerB, workerA)

	store := mustProfileStore(t, ProfileConfig{MaxAge: time.Minute, MaxSeries: 16})
	for _, observation := range []LinkObservation{
		testLinkObservation("worker-b", now, 10, 1000),
		testLinkObservation("worker-a", now, 20, 2000),
	} {
		if err := store.ObserveLink(now, observation); err != nil {
			t.Fatal(err)
		}
	}
	for stageIndex, worker := range []registry.Worker{workerB, workerA} {
		stage := plan.Stages[stageIndex]
		for _, operation := range []struct {
			name        string
			inputTokens uint64
			micros      uint64
		}{
			{name: "prefill", inputTokens: 4, micros: uint64(100 * (stageIndex + 1))},
			{name: "decode", inputTokens: 1, micros: uint64(10 * (stageIndex + 1))},
		} {
			if err := store.ObserveCompute(now, ComputeObservation{
				WorkerID: worker.ID, WorkerInstanceID: worker.InstanceID,
				Backend: worker.Capabilities.Backend, Model: plan.Model,
				Operation: operation.name, LayerStart: stage.LayerStart, LayerEnd: stage.LayerEnd,
				InputTokenCount: operation.inputTokens, ComputeMicros: operation.micros,
				ObservedAt: now,
			}); err != nil {
				t.Fatal(err)
			}
		}
	}
	profile := mustProfileSnapshot(t, store, now)
	request := PlanScoringRequest{
		Adapter: "adapter-a",
		Transport: TransportRequirement{
			Protocol: "http-json-v1", TensorEncoding: "base64-json",
		},
		CoordinatorID: "coordinator", CoordinatorInstanceID: "coordinator-run",
		PrefillInputTokens: 8, DecodeSteps: 3,
		FallbackRTTMicros: 100, FallbackBytesPerSecond: 1_000_000,
		Stages: []StageCostEstimate{
			{
				LoadMemoryBytes: 1000, SequenceMemoryBytes: 100,
				PrefillWireBytes: 100, DecodeWireBytesPerStep: 10,
				FallbackPrefillComputeMicros:       500,
				FallbackDecodeComputeMicrosPerStep: 50,
			},
			{
				LoadMemoryBytes: 2000, SequenceMemoryBytes: 200,
				PrefillWireBytes: 200, DecodeWireBytesPerStep: 20,
				FallbackPrefillComputeMicros:       600,
				FallbackDecodeComputeMicrosPerStep: 60,
			},
		},
	}
	return plan, inventory, profile, request
}

func mutateEvaluation(
	input PlanEvaluation,
	mutate func(*PlanEvaluation),
) PlanEvaluation {
	input.Plan.Stages = append([]generation.ExecutionStage(nil), input.Plan.Stages...)
	input.Request.Stages = append([]StageCostEstimate(nil), input.Request.Stages...)
	input.Stages = append([]StagePlanEvaluation(nil), input.Stages...)
	mutate(&input)
	return input
}

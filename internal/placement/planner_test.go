package placement

import (
	"errors"
	"reflect"
	"slices"
	"testing"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/registry"
)

func TestConstructPlanSelectsBestValidMeshPlanAndPreservesEvidence(t *testing.T) {
	now := time.Date(2026, time.August, 17, 18, 0, 0, 0, time.UTC)
	inventory, profile, request := testConstructionFixture(t, now)

	result, err := ConstructPlan(inventory, profile, request)
	if err != nil {
		t.Fatal(err)
	}
	if result.SelectedPlan == nil || !result.SelectedPlan.Eligible {
		t.Fatalf("no valid plan selected: %+v", result)
	}
	if result.SchemaVersion != SchemaVersion || result.InventoryRevision != inventory.Revision ||
		result.ProfileRevision != profile.Revision || !result.GeneratedAt.Equal(now) ||
		result.CandidateWorkerCount != 3 || result.CompletePlanCount != 4 ||
		result.SearchLimitReached {
		t.Fatalf("unexpected construction metadata: %+v", result)
	}
	plan := result.SelectedPlan.Plan
	if len(plan.Stages) != 2 ||
		plan.Stages[0].TargetID != "worker-a" || plan.Stages[0].LayerStart != 0 ||
		plan.Stages[0].LayerEnd != 6 || plan.Stages[1].TargetID != "worker-b" ||
		plan.Stages[1].LayerStart != 6 || plan.Stages[1].LayerEnd != 12 ||
		plan.Stages[1].ResponseMode != generation.StageResponseSampledToken {
		t.Fatalf("unexpected selected plan: %+v", plan)
	}
	if result.SelectedPlan.Score.StageCount != 2 ||
		result.SelectedPlan.Score.RecentFailureCount != 0 {
		t.Fatalf("unexpected selected score: %+v", result.SelectedPlan.Score)
	}
	full := findRangeEvaluation(t, result.Ranges, 0, 12)
	if len(full.Eligibility.Candidates) != 3 {
		t.Fatalf("full-range evidence omitted candidates: %+v", full)
	}
	for _, candidate := range full.Eligibility.Candidates {
		if candidate.Eligible || !slices.Contains(
			rejectionCodes(candidate), RejectionInsufficientMemory,
		) {
			t.Fatalf("full-range rejection evidence = %+v", candidate)
		}
	}
	if !slices.IsSortedFunc(result.Request.Ranges, func(left, right RangeCostEstimate) int {
		if left.LayerStart != right.LayerStart {
			return left.LayerStart - right.LayerStart
		}
		return left.LayerEnd - right.LayerEnd
	}) {
		t.Fatalf("normalized ranges are not sorted: %+v", result.Request.Ranges)
	}
	for _, stage := range result.SelectedPlan.Stages {
		if len(stage.Eligibility.Candidates) != 3 {
			t.Fatalf("selected stage omitted worker evidence: %+v", stage)
		}
	}

	reversedInventory := inventory
	reversedInventory.Workers = append([]registry.Worker(nil), inventory.Workers...)
	slices.Reverse(reversedInventory.Workers)
	reversedRequest := request
	reversedRequest.Ranges = append([]RangeCostEstimate(nil), request.Ranges...)
	slices.Reverse(reversedRequest.Ranges)
	repeated, err := ConstructPlan(reversedInventory, profile, reversedRequest)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result, repeated) {
		t.Fatalf("construction depends on input order:\nfirst=%+v\nsecond=%+v", result, repeated)
	}
}

func TestConstructPlanPrefersExactRetainedShardReuse(t *testing.T) {
	now := time.Date(2026, time.August, 17, 18, 0, 0, 0, time.UTC)
	inventory, profile, request := testConstructionFixture(t, now)
	inventory.Workers = slices.DeleteFunc(
		append([]registry.Worker(nil), inventory.Workers...),
		func(worker registry.Worker) bool { return worker.ID == "worker-c" },
	)
	request.MaxStages = 2
	request.Ranges = []RangeCostEstimate{
		testConstructionRange(0, 12, 3000, 100, 10),
		testConstructionRange(0, 6, 1000, 100, 10),
		testConstructionRange(6, 12, 1000, 100, 10),
	}
	shardID, err := generation.DeriveExecutionShardID(request.Model, 0, 6, true, false)
	if err != nil {
		t.Fatal(err)
	}
	for index := range inventory.Workers {
		if inventory.Workers[index].ID == "worker-b" {
			inventory.Workers[index].Status.RetainedShards = []registry.RetainedShard{{
				ID: shardID, ModelID: request.Model.ID,
				CheckpointFingerprint: request.Model.CheckpointFingerprint,
				LayerStart:            0, LayerEnd: 6, OwnsInput: true, MemoryBytes: 1000,
			}}
		}
	}

	result, err := ConstructPlan(inventory, profile, request)
	if err != nil {
		t.Fatal(err)
	}
	if result.SelectedPlan == nil || len(result.SelectedPlan.Plan.Stages) != 2 ||
		result.SelectedPlan.Plan.Stages[0].TargetID != "worker-b" ||
		result.SelectedPlan.Plan.Stages[1].TargetID != "worker-a" ||
		result.SelectedPlan.Score.ReusedStageCount != 1 ||
		result.SelectedPlan.Score.NewLoadMemoryBytes != 1000 {
		t.Fatalf("retained shard was not preferred: %+v", result.SelectedPlan)
	}
}

func TestConstructPlanDeclinesWhenNoCompletePlanIsEligible(t *testing.T) {
	now := time.Date(2026, time.August, 17, 18, 0, 0, 0, time.UTC)
	inventory, profile, request := testConstructionFixture(t, now)
	for index := range inventory.Workers {
		inventory.Workers[index].Status.Health = registry.HealthDraining
	}

	result, err := ConstructPlan(inventory, profile, request)
	if err != nil {
		t.Fatal(err)
	}
	if result.SelectedPlan != nil || result.CandidateWorkerCount != 0 ||
		result.CompletePlanCount != 0 || len(result.Ranges) != len(request.Ranges) {
		t.Fatalf("unexpected declined result: %+v", result)
	}
	for _, rangeEvaluation := range result.Ranges {
		for _, candidate := range rangeEvaluation.Eligibility.Candidates {
			if !slices.Contains(rejectionCodes(candidate), RejectionUnhealthy) {
				t.Fatalf("declined candidate has no health reason: %+v", candidate)
			}
		}
	}
}

func TestConstructPlanSkipsWorkersAtReportedRequestCapacity(t *testing.T) {
	now := time.Date(2026, time.August, 17, 18, 0, 0, 0, time.UTC)
	inventory, profile, request := testConstructionFixture(t, now)
	for index := range inventory.Workers {
		worker := &inventory.Workers[index]
		if worker.ID != "worker-a" {
			continue
		}
		worker.Status.OpenSequenceCount = worker.Capabilities.Admission.MaxConcurrentRequests
		worker.Status.RetainedShards = []registry.RetainedShard{{
			ID: "unrelated-busy-shard", ModelID: "other-model",
			CheckpointFingerprint: "other-checkpoint",
			LayerStart:            0, LayerEnd: 1,
			OpenSequenceCount: worker.Status.OpenSequenceCount,
		}}
	}

	result, err := ConstructPlan(inventory, profile, request)
	if err != nil {
		t.Fatal(err)
	}
	if result.SelectedPlan == nil || planUsesWorker(result.SelectedPlan.Plan, "worker-a") {
		t.Fatalf("full worker prevented fallback placement: %+v", result.SelectedPlan)
	}
	rangeEvidence := findRangeEvaluation(t, result.Ranges, 0, 6)
	for _, candidate := range rangeEvidence.Eligibility.Candidates {
		if candidate.WorkerID == "worker-a" && !slices.Contains(
			rejectionCodes(candidate), RejectionWorkerCapacityExhausted,
		) {
			t.Fatalf("full worker has no capacity rejection: %+v", candidate)
		}
	}
}

func TestConstructPlanReactsToMembershipAndTopologyChanges(t *testing.T) {
	now := time.Date(2026, time.August, 17, 18, 0, 0, 0, time.UTC)
	inventory, profile, request := testConstructionFixture(t, now)
	initial, err := ConstructPlan(inventory, profile, request)
	if err != nil {
		t.Fatal(err)
	}
	if initial.SelectedPlan == nil {
		t.Fatal("initial mesh produced no plan")
	}

	removed := inventory
	removed.Revision++
	removed.GeneratedAt = now.Add(time.Second)
	removed.Workers = slices.DeleteFunc(
		append([]registry.Worker(nil), inventory.Workers...),
		func(worker registry.Worker) bool { return worker.ID == "worker-b" },
	)
	for index := range removed.Workers {
		removed.Workers[index].StatusObservedAt = removed.GeneratedAt
		removed.Workers[index].LastSeen = removed.GeneratedAt
		removed.Workers[index].ExpiresAt = removed.GeneratedAt.Add(30 * time.Second)
	}
	removedProfile := mustProfileSnapshot(t, mustProfileStore(t, ProfileConfig{
		MaxAge: time.Minute, MaxSeries: 16,
	}), removed.GeneratedAt)
	afterRemoval, err := ConstructPlan(removed, removedProfile, request)
	if err != nil {
		t.Fatal(err)
	}
	if afterRemoval.SelectedPlan == nil ||
		planUsesWorker(afterRemoval.SelectedPlan.Plan, "worker-b") ||
		afterRemoval.SelectedPlan.Plan.InventoryRevision != "8" {
		t.Fatalf("removed worker survived replanning: %+v", afterRemoval.SelectedPlan)
	}

	joined := inventory
	joined.Revision += 2
	joined.GeneratedAt = now.Add(2 * time.Second)
	joined.Workers = append([]registry.Worker(nil), inventory.Workers...)
	workerD := testConstructionWorker("worker-d", joined.GeneratedAt)
	joined.Workers = append(joined.Workers, workerD)
	for index := range joined.Workers {
		joined.Workers[index].StatusObservedAt = joined.GeneratedAt
		joined.Workers[index].LastSeen = joined.GeneratedAt
		joined.Workers[index].ExpiresAt = joined.GeneratedAt.Add(30 * time.Second)
	}
	store := mustProfileStore(t, ProfileConfig{MaxAge: time.Minute, MaxSeries: 16})
	if err := store.ObserveLink(joined.GeneratedAt, LinkObservation{
		SourceID:         request.Scoring.CoordinatorID,
		SourceInstanceID: request.Scoring.CoordinatorInstanceID,
		TargetID:         "worker-d", TargetInstanceID: workerD.InstanceID,
		Protocol:       request.Scoring.Transport.Protocol,
		TensorEncoding: request.Scoring.Transport.TensorEncoding,
		ObservedAt:     joined.GeneratedAt, RTTMicros: 1,
	}); err != nil {
		t.Fatal(err)
	}
	joinedProfile := mustProfileSnapshot(t, store, joined.GeneratedAt)
	afterJoin, err := ConstructPlan(joined, joinedProfile, request)
	if err != nil {
		t.Fatal(err)
	}
	if afterJoin.SelectedPlan == nil ||
		!planUsesWorker(afterJoin.SelectedPlan.Plan, "worker-d") ||
		afterJoin.SelectedPlan.Plan.Revision == initial.SelectedPlan.Plan.Revision {
		t.Fatalf("better joined worker did not alter placement: %+v", afterJoin.SelectedPlan)
	}
}

func TestConstructPlanEnforcesSearchOperationLimit(t *testing.T) {
	now := time.Date(2026, time.August, 17, 18, 0, 0, 0, time.UTC)
	inventory, profile, request := testConstructionFixture(t, now)
	request.MaxSearchOperations = uint64(len(request.Ranges)*len(inventory.Workers) - 1)

	result, err := ConstructPlan(inventory, profile, request)
	if !errors.Is(err, ErrPlanSearchLimit) || !result.SearchLimitReached ||
		result.SelectedPlan != nil {
		t.Fatalf("search limit result=%+v error=%v", result, err)
	}
}

func TestConstructPlanRejectsInvalidRequests(t *testing.T) {
	now := time.Date(2026, time.August, 17, 18, 0, 0, 0, time.UTC)
	validInventory, validProfile, validRequest := testConstructionFixture(t, now)
	tests := []struct {
		name   string
		mutate func(*registry.Inventory, *ProfileSnapshot, *PlanConstructionRequest)
	}{
		{name: "scoring stages", mutate: func(_ *registry.Inventory, _ *ProfileSnapshot, request *PlanConstructionRequest) {
			request.Scoring.Stages = []StageCostEstimate{{}}
		}},
		{name: "model", mutate: func(_ *registry.Inventory, _ *ProfileSnapshot, request *PlanConstructionRequest) {
			request.Model.ID = ""
		}},
		{name: "response mode", mutate: func(_ *registry.Inventory, _ *ProfileSnapshot, request *PlanConstructionRequest) {
			request.TerminalResponseMode = "other"
		}},
		{name: "stage count", mutate: func(_ *registry.Inventory, _ *ProfileSnapshot, request *PlanConstructionRequest) {
			request.MaxStages = request.Model.LayerCount + 1
		}},
		{name: "ranges", mutate: func(_ *registry.Inventory, _ *ProfileSnapshot, request *PlanConstructionRequest) {
			request.Ranges = nil
		}},
		{name: "duplicate range", mutate: func(_ *registry.Inventory, _ *ProfileSnapshot, request *PlanConstructionRequest) {
			request.Ranges = append(request.Ranges, request.Ranges[0])
		}},
		{name: "range shape", mutate: func(_ *registry.Inventory, _ *ProfileSnapshot, request *PlanConstructionRequest) {
			request.Ranges[0].LayerEnd = request.Ranges[0].LayerStart
		}},
		{name: "range estimate", mutate: func(_ *registry.Inventory, _ *ProfileSnapshot, request *PlanConstructionRequest) {
			request.Ranges[0].Estimate.LoadMemoryBytes = 0
		}},
		{name: "snapshot time", mutate: func(_ *registry.Inventory, profile *ProfileSnapshot, _ *PlanConstructionRequest) {
			profile.GeneratedAt = profile.GeneratedAt.Add(time.Second)
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			inventory := validInventory
			inventory.Workers = append([]registry.Worker(nil), validInventory.Workers...)
			profile := validProfile
			request := validRequest
			request.Ranges = append([]RangeCostEstimate(nil), validRequest.Ranges...)
			test.mutate(&inventory, &profile, &request)
			if _, err := ConstructPlan(inventory, profile, request); err == nil {
				t.Fatal("invalid construction request was accepted")
			}
		})
	}
}

func testConstructionFixture(
	t *testing.T,
	now time.Time,
) (registry.Inventory, ProfileSnapshot, PlanConstructionRequest) {
	t.Helper()
	workerA := testConstructionWorker("worker-a", now)
	workerB := testConstructionWorker("worker-b", now)
	workerC := testConstructionWorker("worker-c", now)
	workerC.Status.RecentFailureCount = 10
	inventory := testInventory(now, workerB, workerC, workerA)
	profile := mustProfileSnapshot(t, mustProfileStore(t, ProfileConfig{
		MaxAge: time.Minute, MaxSeries: 16,
	}), now)
	request := PlanConstructionRequest{
		Model: testProfileModel(),
		Scoring: PlanScoringRequest{
			Adapter: "adapter-a",
			Transport: TransportRequirement{
				Protocol: "http-json-v1", TensorEncoding: "base64-json",
			},
			CoordinatorID: "coordinator", CoordinatorInstanceID: "coordinator-run",
			PrefillInputTokens: 4, DecodeSteps: 2,
			FallbackRTTMicros: 100, FallbackBytesPerSecond: 1_000_000,
		},
		TerminalResponseMode: generation.StageResponseSampledToken,
		MaxStages:            3, MaxSearchOperations: 1000,
		Ranges: []RangeCostEstimate{
			testConstructionRange(8, 12, 700, 80, 8),
			testConstructionRange(0, 12, 3000, 100, 10),
			testConstructionRange(4, 8, 700, 80, 8),
			testConstructionRange(0, 6, 1000, 100, 10),
			testConstructionRange(6, 12, 1000, 100, 10),
			testConstructionRange(0, 4, 700, 80, 8),
		},
	}
	return inventory, profile, request
}

func testConstructionWorker(id string, now time.Time) registry.Worker {
	worker := testWorker(id, now)
	worker.Status.AvailableMemoryBytes = 1500
	return worker
}

func testConstructionRange(
	start, end int,
	loadMemory, fallbackPrefill, fallbackDecode uint64,
) RangeCostEstimate {
	return RangeCostEstimate{
		LayerStart: start, LayerEnd: end,
		Estimate: StageCostEstimate{
			LoadMemoryBytes: loadMemory, SequenceMemoryBytes: 100,
			PrefillWireBytes: 100, DecodeWireBytesPerStep: 10,
			FallbackPrefillComputeMicros:       fallbackPrefill,
			FallbackDecodeComputeMicrosPerStep: fallbackDecode,
		},
	}
}

func findRangeEvaluation(
	t *testing.T,
	ranges []RangePlanEvaluation,
	start, end int,
) RangePlanEvaluation {
	t.Helper()
	for _, rangeEvaluation := range ranges {
		if rangeEvaluation.LayerStart == start && rangeEvaluation.LayerEnd == end {
			return rangeEvaluation
		}
	}
	t.Fatalf("range [%d,%d) not found", start, end)
	return RangePlanEvaluation{}
}

func planUsesWorker(plan generation.ExecutionPlan, workerID string) bool {
	for _, stage := range plan.Stages {
		if stage.TargetID == workerID {
			return true
		}
	}
	return false
}

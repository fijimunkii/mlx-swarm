package placement

import (
	"reflect"
	"slices"
	"testing"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/registry"
)

func TestEvaluateCandidatesIsDeterministicAndRecognizesReuse(t *testing.T) {
	now := time.Date(2026, time.August, 16, 12, 0, 0, 0, time.UTC)
	requirement := testRequirement()
	requirement.ShardID = "retained-stage"
	reused := testWorker("worker-b", now)
	reused.Status.AvailableMemoryBytes = 150
	reused.Status.RetainedBytes = 100
	reused.Status.RetainedShards = []registry.RetainedShard{testRetainedShard(requirement, 1)}
	load := testWorker("worker-a", now)
	inventory := testInventory(now, reused, load)

	evaluation, err := EvaluateCandidates(inventory, requirement)
	if err != nil {
		t.Fatal(err)
	}
	if evaluation.SchemaVersion != SchemaVersion || evaluation.InventoryRevision != inventory.Revision ||
		evaluation.StatusMaxAgeMillis != inventory.LeaseTTLMillis || len(evaluation.Candidates) != 2 {
		t.Fatalf("unexpected evaluation: %+v", evaluation)
	}
	if got := []string{evaluation.Candidates[0].WorkerID, evaluation.Candidates[1].WorkerID}; !slices.Equal(got, []string{"worker-a", "worker-b"}) {
		t.Fatalf("candidate order = %v", got)
	}
	loadedCandidate := evaluation.Candidates[0]
	if !loadedCandidate.Eligible || loadedCandidate.ReusesRetainedShard ||
		loadedCandidate.RequiredAdditionalMemoryBytes != 1100 || len(loadedCandidate.Rejections) != 0 {
		t.Fatalf("unexpected load candidate: %+v", loadedCandidate)
	}
	reusedCandidate := evaluation.Candidates[1]
	if !reusedCandidate.Eligible || !reusedCandidate.ReusesRetainedShard ||
		reusedCandidate.RetainedShardID != "retained-stage" ||
		reusedCandidate.RequiredAdditionalMemoryBytes != 100 || len(reusedCandidate.Rejections) != 0 {
		t.Fatalf("unexpected retained candidate: %+v", reusedCandidate)
	}

	reversed := testInventory(now, load, reused)
	repeated, err := EvaluateCandidates(reversed, requirement)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(evaluation, repeated) {
		t.Fatalf("evaluation depends on inventory order:\nfirst=%+v\nsecond=%+v", evaluation, repeated)
	}
}

func TestEvaluateCandidatesDoesNotReuseDifferentShardIdentity(t *testing.T) {
	now := time.Date(2026, time.August, 16, 12, 0, 0, 0, time.UTC)
	requirement := testRequirement()
	requirement.ShardID = "next-plan-stage"
	worker := testWorker("worker-a", now)
	worker.Status.AvailableMemoryBytes = 150
	worker.Status.RetainedShards = []registry.RetainedShard{testRetainedShard(requirement, 0)}
	worker.Status.RetainedShards[0].ID = "previous-plan-stage"

	evaluation, err := EvaluateCandidates(testInventory(now, worker), requirement)
	if err != nil {
		t.Fatal(err)
	}
	candidate := evaluation.Candidates[0]
	if candidate.ReusesRetainedShard || candidate.RetainedShardID != "" ||
		candidate.RequiredAdditionalMemoryBytes != 1100 ||
		!slices.Equal(rejectionCodes(candidate), []RejectionCode{RejectionInsufficientMemory}) {
		t.Fatalf("configuration-compatible shard with a different ID was reused: %+v", candidate)
	}
}

func TestEvaluateCandidatesReportsStableRejectionCodes(t *testing.T) {
	now := time.Date(2026, time.August, 16, 12, 0, 0, 0, time.UTC)
	requirement := testRequirement()
	worker := testWorker("worker-a", now.Add(-31*time.Second))
	worker.Status.Health = registry.HealthDegraded
	worker.Status.AvailableMemoryBytes = 10
	worker.Status.RetainedBytes = 950
	worker.Capabilities.Adapters = []string{"other-adapter"}
	worker.Capabilities.CheckpointFingerprints = []string{"sha256:other"}
	worker.Capabilities.Operations = []string{"health"}
	worker.Capabilities.Transports = []registry.Transport{{
		Protocol: "binary-v1", TensorEncodings: []string{"raw"}, MaxRequestBytes: 4096,
	}}

	evaluation, err := EvaluateCandidates(testInventory(now, worker), requirement)
	if err != nil {
		t.Fatal(err)
	}
	candidate := evaluation.Candidates[0]
	if candidate.Eligible {
		t.Fatalf("incompatible worker was eligible: %+v", candidate)
	}
	codes := rejectionCodes(candidate)
	want := []RejectionCode{
		RejectionStaleStatus,
		RejectionUnhealthy,
		RejectionUnsupportedAdapter,
		RejectionIncompatibleCheckpoint,
		RejectionUnsupportedOperation,
		RejectionUnsupportedTransport,
		RejectionInsufficientMemory,
		RejectionRetainedBudgetExceeded,
	}
	if !slices.Equal(codes, want) {
		t.Fatalf("rejection codes = %v, want %v", codes, want)
	}
}

func TestEvaluateCandidatesChecksTransportAndRetainedCapacity(t *testing.T) {
	now := time.Date(2026, time.August, 16, 12, 0, 0, 0, time.UTC)
	requirement := testRequirement()
	requirement.ShardID = "retained-stage"
	requirement.Transport.RequireTLS = true
	worker := testWorker("worker-a", now)
	worker.Capabilities.Transports[0].TensorEncodings = []string{"raw"}
	worker.Status.RetainedShards = []registry.RetainedShard{testRetainedShard(requirement, 2)}

	evaluation, err := EvaluateCandidates(testInventory(now, worker), requirement)
	if err != nil {
		t.Fatal(err)
	}
	want := []RejectionCode{
		RejectionUnsupportedEncoding, RejectionTLSRequired, RejectionSequenceCapacityExhausted,
	}
	if got := rejectionCodes(evaluation.Candidates[0]); !slices.Equal(got, want) {
		t.Fatalf("rejection codes = %v, want %v", got, want)
	}
}

func TestEvaluateCandidatesUsesExplicitStricterFreshness(t *testing.T) {
	now := time.Date(2026, time.August, 16, 12, 0, 0, 0, time.UTC)
	requirement := testRequirement()
	requirement.StatusMaxAgeMillis = (5 * time.Second).Milliseconds()
	worker := testWorker("worker-a", now.Add(-6*time.Second))

	evaluation, err := EvaluateCandidates(testInventory(now, worker), requirement)
	if err != nil {
		t.Fatal(err)
	}
	if evaluation.StatusMaxAgeMillis != requirement.StatusMaxAgeMillis ||
		!slices.Equal(rejectionCodes(evaluation.Candidates[0]), []RejectionCode{RejectionStaleStatus}) {
		t.Fatalf("explicit freshness was not enforced: %+v", evaluation)
	}
}

func TestEvaluateCandidatesRejectsInvalidInputs(t *testing.T) {
	now := time.Date(2026, time.August, 16, 12, 0, 0, 0, time.UTC)
	validInventory := testInventory(now, testWorker("worker-a", now))
	validRequirement := testRequirement()
	tests := []struct {
		name              string
		mutateInventory   func(*registry.Inventory)
		mutateRequirement func(*StageRequirement)
	}{
		{name: "inventory schema", mutateInventory: func(input *registry.Inventory) { input.SchemaVersion++ }},
		{name: "inventory time", mutateInventory: func(input *registry.Inventory) { input.GeneratedAt = time.Time{} }},
		{name: "inventory TTL", mutateInventory: func(input *registry.Inventory) { input.LeaseTTLMillis = 0 }},
		{name: "model", mutateRequirement: func(input *StageRequirement) { input.Model.ID = "" }},
		{name: "adapter", mutateRequirement: func(input *StageRequirement) { input.Adapter = " " }},
		{name: "range", mutateRequirement: func(input *StageRequirement) { input.LayerEnd = input.LayerStart }},
		{name: "input ownership", mutateRequirement: func(input *StageRequirement) { input.OwnsInput = false }},
		{name: "output ownership", mutateRequirement: func(input *StageRequirement) { input.OwnsOutput = true }},
		{name: "load memory", mutateRequirement: func(input *StageRequirement) { input.LoadMemoryBytes = 0 }},
		{name: "sequence memory", mutateRequirement: func(input *StageRequirement) { input.SequenceMemoryBytes = 0 }},
		{name: "memory overflow", mutateRequirement: func(input *StageRequirement) { input.LoadMemoryBytes = ^uint64(0) }},
		{name: "transport", mutateRequirement: func(input *StageRequirement) { input.Transport.Protocol = "" }},
		{name: "status age", mutateRequirement: func(input *StageRequirement) { input.StatusMaxAgeMillis = -1 }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			inventory := validInventory
			requirement := validRequirement
			if test.mutateInventory != nil {
				test.mutateInventory(&inventory)
			}
			if test.mutateRequirement != nil {
				test.mutateRequirement(&requirement)
			}
			if _, err := EvaluateCandidates(inventory, requirement); err == nil {
				t.Fatal("invalid placement input was accepted")
			}
		})
	}
}

func testRequirement() StageRequirement {
	return StageRequirement{
		Model: generation.ExecutionModel{
			ID: "model-a", CheckpointFingerprint: "sha256:model-a", LayerCount: 12,
		},
		Adapter: "adapter-a", LayerStart: 0, LayerEnd: 6, OwnsInput: true,
		LoadMemoryBytes: 1000, SequenceMemoryBytes: 100,
		Transport: TransportRequirement{
			Protocol: "http-json-v1", TensorEncoding: "base64-json",
		},
	}
}

func testInventory(now time.Time, workers ...registry.Worker) registry.Inventory {
	return registry.Inventory{
		SchemaVersion: registry.SchemaVersion, Revision: 7, GeneratedAt: now,
		LeaseTTLMillis: (30 * time.Second).Milliseconds(), Workers: workers,
	}
}

func testWorker(id string, statusObservedAt time.Time) registry.Worker {
	return registry.Worker{
		Registration: registry.Registration{
			SchemaVersion: registry.SchemaVersion, ID: id, InstanceID: id + "-instance",
			Endpoint: "http://" + id + ":8080",
			Capabilities: registry.Capabilities{
				Backend: "test", Runtime: "fixture", OS: "linux", Architecture: "arm64",
				Device: "cpu", PhysicalMemoryBytes: 10_000,
				Adapters: []string{"adapter-a"}, Operations: []string{
					"closeSequence", "decode", "detokenize", "loadShard", "modelInfo",
					"openSequence", "prefill", "state", "tokenize",
				},
				CheckpointFingerprints: []string{"sha256:model-a"},
				Admission: registry.AdmissionLimits{
					MaxConcurrentRequests: 1, MaxOpenSequencesPerShard: 2,
					RetainedByteBudget: 1000,
				},
				Transports: []registry.Transport{{
					Protocol: "http-json-v1", TensorEncodings: []string{"base64-json"},
					MaxRequestBytes: 4096,
				}},
			},
			Status: registry.Status{Health: registry.HealthHealthy, AvailableMemoryBytes: 5000},
		},
		RegisteredAt: time.Date(2026, time.August, 16, 11, 0, 0, 0, time.UTC),
		LastSeen:     statusObservedAt, ExpiresAt: statusObservedAt.Add(30 * time.Second),
		StatusObservedAt: statusObservedAt,
	}
}

func testRetainedShard(requirement StageRequirement, openSequences int) registry.RetainedShard {
	return registry.RetainedShard{
		ID: "retained-stage", ModelID: requirement.Model.ID,
		CheckpointFingerprint: requirement.Model.CheckpointFingerprint,
		LayerStart:            requirement.LayerStart, LayerEnd: requirement.LayerEnd,
		OwnsInput: requirement.OwnsInput, OwnsOutput: requirement.OwnsOutput,
		OpenSequenceCount: openSequences,
	}
}

func rejectionCodes(candidate Candidate) []RejectionCode {
	codes := make([]RejectionCode, len(candidate.Rejections))
	for index, rejection := range candidate.Rejections {
		codes[index] = rejection.Code
	}
	return codes
}

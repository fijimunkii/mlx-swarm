package mesh

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/placement"
	"github.com/fijimunkii/mlx-swarm/internal/registry"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

func TestHTTPResolverRequiresInstanceBoundTransport(t *testing.T) {
	resolver := HTTPResolver{}
	for _, requirement := range []placement.TransportRequirement{
		{Protocol: "http-json-v1", TensorEncoding: workerproc.Base64JSONTensorEncoding},
		{Protocol: workerproc.InstanceBoundHTTPProtocol, TensorEncoding: "other"},
	} {
		if err := resolver.ValidateTransport(requirement); err == nil {
			t.Fatalf("transport %+v was accepted", requirement)
		}
	}
	if err := resolver.ValidateTransport(placement.TransportRequirement{
		Protocol:       workerproc.InstanceBoundHTTPProtocol,
		TensorEncoding: workerproc.Base64JSONTensorEncoding,
	}); err != nil {
		t.Fatalf("instance-bound transport was rejected: %v", err)
	}
}

func TestSequenceSchedulerFreezesOneSequenceAndReplansTheNext(t *testing.T) {
	now := time.Date(2026, time.August, 17, 2, 0, 0, 0, time.UTC)
	membership := registry.New(time.Minute, registry.WithClock(func() time.Time { return now }))
	profiles, err := placement.NewProfileStore(placement.ProfileConfig{
		MaxAge: time.Minute, MaxSeries: 32,
	})
	if err != nil {
		t.Fatal(err)
	}
	callers := make(map[string]*schedulerMetadataWorker)
	for _, id := range []string{"worker-a", "worker-b", "worker-c"} {
		registerSchedulerWorker(t, membership, id)
		callers[id] = newSchedulerMetadataWorker()
	}
	var resolved []TargetBinding
	scheduler, err := NewSequenceScheduler(
		membership,
		profiles,
		TargetResolverFunc(func(binding TargetBinding) (workerproc.PersistentCaller, error) {
			resolved = append(resolved, binding)
			caller := callers[binding.WorkerID]
			if caller == nil {
				return nil, fmt.Errorf("no caller for %s", binding.WorkerID)
			}
			return caller, nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	request := schedulerPlanRequest()

	active, initial, err := scheduler.Prepare(
		context.Background(), request, nil, generation.PlannedSessionConfig{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if active == nil || initial.Construction.SelectedPlan == nil || len(initial.Targets) != 2 {
		t.Fatalf("initial selection = %+v", initial)
	}
	initialPlan := initial.Construction.SelectedPlan.Plan
	initialTargets := planTargetIDs(initialPlan)
	if !slices.Equal(initialTargets, []string{"worker-a", "worker-b"}) {
		t.Fatalf("initial targets = %v", initialTargets)
	}
	if len(resolved) != 2 || resolved[0].InstanceID != "worker-a-instance" ||
		resolved[0].Endpoint != "http://worker-a.example:8080" {
		t.Fatalf("resolved bindings = %+v", resolved)
	}

	// Returned evidence and Info are detached from the active runtime contract.
	initial.Targets[0].WorkerID = "mutated"
	initial.Construction.SelectedPlan.Plan.Stages[0].TargetID = "mutated"
	info := active.Info()
	if !slices.Equal(planTargetIDs(info.ExecutionPlan), initialTargets) {
		t.Fatalf("active plan changed through returned evidence: %+v", info)
	}

	now = now.Add(time.Second)
	removedID := initialTargets[0]
	if err := membership.Remove(removedID, removedID+"-instance"); err != nil {
		t.Fatal(err)
	}
	next, afterRemoval, err := scheduler.Prepare(
		context.Background(), request, nil, generation.PlannedSessionConfig{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if next == nil || afterRemoval.Construction.SelectedPlan == nil ||
		slices.Contains(planTargetIDs(afterRemoval.Construction.SelectedPlan.Plan), removedID) {
		t.Fatalf("removed worker survived next-sequence planning: %+v", afterRemoval)
	}
	if active.Info().ExecutionPlan.Revision != initialPlan.Revision {
		t.Fatal("membership removal changed the already prepared active plan")
	}
	if err := next.Close(); err != nil {
		t.Fatal(err)
	}

	// A materially better worker affects only a later sequence.
	now = now.Add(time.Second)
	registerSchedulerWorker(t, membership, "worker-d")
	callers["worker-d"] = newSchedulerMetadataWorker()
	if err := profiles.ObserveLink(now, placement.LinkObservation{
		SourceID: "coordinator", SourceInstanceID: "coordinator-instance",
		TargetID: "worker-d", TargetInstanceID: "worker-d-instance",
		Protocol: "http-json-v1", TensorEncoding: "base64-json",
		ObservedAt: now, RTTMicros: 1,
	}); err != nil {
		t.Fatal(err)
	}
	joined, afterJoin, err := scheduler.Prepare(
		context.Background(), request, nil, generation.PlannedSessionConfig{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if afterJoin.Construction.SelectedPlan == nil ||
		!slices.Contains(planTargetIDs(afterJoin.Construction.SelectedPlan.Plan), "worker-d") {
		t.Fatalf("better worker did not affect later placement: %+v", afterJoin)
	}
	if err := joined.Close(); err != nil {
		t.Fatal(err)
	}

	// A fresh failure signal is considered only when admitting another
	// sequence, and an alternative worker is selected when capacity remains.
	failedID := ""
	for _, targetID := range planTargetIDs(afterJoin.Construction.SelectedPlan.Plan) {
		if targetID != "worker-d" {
			failedID = targetID
			break
		}
	}
	if failedID == "" {
		t.Fatal("joined plan did not contain a non-worker-d target")
	}
	now = now.Add(time.Second)
	updateSchedulerStatus(t, membership, failedID, func(status *registry.Status) {
		status.RecentFailureCount = 20
	})
	afterFailureSequence, afterFailure, err := scheduler.Prepare(
		context.Background(), request, nil, generation.PlannedSessionConfig{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if slices.Contains(planTargetIDs(afterFailure.Construction.SelectedPlan.Plan), failedID) {
		t.Fatalf("failed worker remained in the next plan: %+v", afterFailure)
	}
	if err := afterFailureSequence.Close(); err != nil {
		t.Fatal(err)
	}

	// Fresh compute evidence can similarly move a later sequence away from a
	// worker whose measured range execution has slowed down.
	now = now.Add(time.Second)
	for _, layerRange := range [][2]int{{0, 2}, {2, 4}} {
		for _, observation := range []placement.ComputeObservation{
			{
				WorkerID: "worker-d", WorkerInstanceID: "worker-d-instance", Backend: "test",
				Model: request.Model, Operation: "prefill",
				LayerStart: layerRange[0], LayerEnd: layerRange[1],
				InputTokenCount: request.Scoring.PrefillInputTokens,
				ComputeMicros:   1_000_000, ObservedAt: now,
			},
			{
				WorkerID: "worker-d", WorkerInstanceID: "worker-d-instance", Backend: "test",
				Model: request.Model, Operation: "decode",
				LayerStart: layerRange[0], LayerEnd: layerRange[1],
				InputTokenCount: 1, ComputeMicros: 1_000_000, ObservedAt: now,
			},
		} {
			if err := profiles.ObserveCompute(now, observation); err != nil {
				t.Fatal(err)
			}
		}
	}
	afterSlowdownSequence, afterSlowdown, err := scheduler.Prepare(
		context.Background(), request, nil, generation.PlannedSessionConfig{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if slices.Contains(planTargetIDs(afterSlowdown.Construction.SelectedPlan.Plan), "worker-d") {
		t.Fatalf("slowed worker remained in the next plan: %+v", afterSlowdown)
	}
	if err := afterSlowdownSequence.Close(); err != nil {
		t.Fatal(err)
	}

	// A prepared admission is single-use even when the first attempt is
	// canceled before opening any worker sequence.
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := active.Generate(canceled, generation.Request{
		Prompt: "hello", MaxTokens: 1,
	}); !errors.Is(err, context.Canceled) {
		t.Fatalf("first canceled run error = %v", err)
	}
	if _, err := active.Generate(context.Background(), generation.Request{
		Prompt: "hello", MaxTokens: 1,
	}); !errors.Is(err, ErrSequenceAlreadyRun) {
		t.Fatalf("second run error = %v", err)
	}

	// Expiry is applied at the next admission boundary and returns the full
	// declined construction evidence rather than a stale selection.
	now = now.Add(2 * time.Minute)
	expired, declined, err := scheduler.Prepare(
		context.Background(), request, nil, generation.PlannedSessionConfig{},
	)
	if expired != nil || !errors.Is(err, ErrNoEligiblePlan) ||
		declined.Construction.SelectedPlan != nil || len(declined.Targets) != 0 {
		t.Fatalf("expired inventory admission: session=%v selection=%+v err=%v", expired, declined, err)
	}
}

func TestSequenceSchedulerReservesAndReleasesShardAdmission(t *testing.T) {
	now := time.Date(2026, time.August, 17, 2, 30, 0, 0, time.UTC)
	membership := registry.New(time.Minute, registry.WithClock(func() time.Time { return now }))
	profiles, err := placement.NewProfileStore(placement.ProfileConfig{
		MaxAge: time.Minute, MaxSeries: 16,
	})
	if err != nil {
		t.Fatal(err)
	}
	callers := map[string]*schedulerMetadataWorker{}
	for _, id := range []string{"worker-a", "worker-b"} {
		registration := schedulerRegistration(id)
		registration.Capabilities.Admission.MaxOpenSequencesPerShard = 1
		if _, err := membership.Register(registration); err != nil {
			t.Fatal(err)
		}
		callers[id] = newSchedulerMetadataWorker()
	}
	scheduler, err := NewSequenceScheduler(
		membership,
		profiles,
		TargetResolverFunc(func(binding TargetBinding) (workerproc.PersistentCaller, error) {
			return callers[binding.WorkerID], nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	first, _, err := scheduler.Prepare(
		context.Background(), schedulerPlanRequest(), nil, generation.PlannedSessionConfig{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if second, _, err := scheduler.Prepare(
		context.Background(), schedulerPlanRequest(), nil, generation.PlannedSessionConfig{},
	); second != nil || !errors.Is(err, ErrSequenceCapacityReserved) {
		t.Fatalf("concurrent admission: sequence=%v err=%v", second, err)
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := first.Generate(canceled, generation.Request{
		Prompt: "hello", MaxTokens: 1,
	}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled generation error = %v", err)
	}
	third, _, err := scheduler.Prepare(
		context.Background(), schedulerPlanRequest(), nil, generation.PlannedSessionConfig{},
	)
	if err != nil {
		t.Fatalf("reservation was not released after Generate: %v", err)
	}
	if err := third.Close(); err != nil {
		t.Fatal(err)
	}
	fourth, _, err := scheduler.Prepare(
		context.Background(), schedulerPlanRequest(), nil, generation.PlannedSessionConfig{},
	)
	if err != nil {
		t.Fatalf("reservation was not released after Close: %v", err)
	}
	if err := fourth.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestSequenceSchedulerQuarantinesUnconfirmedCleanupUntilFreshStatus(t *testing.T) {
	now := time.Date(2026, time.August, 17, 2, 45, 0, 0, time.UTC)
	membership := registry.New(time.Minute, registry.WithClock(func() time.Time { return now }))
	profiles, err := placement.NewProfileStore(placement.ProfileConfig{
		MaxAge: time.Minute, MaxSeries: 16,
	})
	if err != nil {
		t.Fatal(err)
	}
	callers := map[string]*schedulerMetadataWorker{}
	for _, id := range []string{"worker-a", "worker-b"} {
		registration := schedulerRegistration(id)
		registration.Capabilities.Admission.MaxOpenSequencesPerShard = 1
		if _, err := membership.Register(registration); err != nil {
			t.Fatal(err)
		}
		callers[id] = newSchedulerMetadataWorker()
	}
	scheduler, err := NewSequenceScheduler(
		membership,
		profiles,
		TargetResolverFunc(func(binding TargetBinding) (workerproc.PersistentCaller, error) {
			return callers[binding.WorkerID], nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	first, _, err := scheduler.Prepare(
		context.Background(), schedulerPlanRequest(), nil, generation.PlannedSessionConfig{},
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, caller := range callers {
		caller.mu.Lock()
		caller.loaded[0].OpenSequenceCount = 1
		caller.mu.Unlock()
	}
	first.finish(false)
	if second, _, err := scheduler.Prepare(
		context.Background(), schedulerPlanRequest(), nil, generation.PlannedSessionConfig{},
	); second != nil || !errors.Is(err, ErrSequenceCapacityReserved) {
		t.Fatalf("ambiguous cleanup was not quarantined: sequence=%v err=%v", second, err)
	}

	now = now.Add(time.Second)
	minimumObservation := uint64(0)
	scheduler.mu.Lock()
	for _, state := range scheduler.reservations {
		if len(state.poisonedAfter) > 0 && state.poisonedAfter[0] > minimumObservation {
			minimumObservation = state.poisonedAfter[0]
		}
	}
	scheduler.mu.Unlock()
	if minimumObservation <= 1 || minimumObservation == unreconciledObservationSequence {
		t.Fatalf("cleanup observation did not establish a worker sequence boundary: %d", minimumObservation)
	}
	for _, id := range []string{"worker-a", "worker-b"} {
		updateSchedulerStatus(t, membership, id, func(status *registry.Status) {
			status.WorkerObservationSequence = minimumObservation - 1
		})
	}
	if stale, _, err := scheduler.Prepare(
		context.Background(), schedulerPlanRequest(), nil, generation.PlannedSessionConfig{},
	); stale != nil || !errors.Is(err, ErrSequenceCapacityReserved) {
		t.Fatalf("late delivery of pre-cleanup state cleared quarantine: sequence=%v err=%v", stale, err)
	}

	now = now.Add(time.Second)
	for _, id := range []string{"worker-a", "worker-b"} {
		updateSchedulerStatus(t, membership, id, func(status *registry.Status) {
			status.WorkerObservationSequence = minimumObservation + 1
		})
	}
	afterFreshStatus, _, err := scheduler.Prepare(
		context.Background(), schedulerPlanRequest(), nil, generation.PlannedSessionConfig{},
	)
	if err != nil {
		t.Fatalf("fresh empty status did not clear quarantine: %v", err)
	}
	if err := afterFreshStatus.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestSequenceSchedulerBoundsPreparationAndReleasesReservation(t *testing.T) {
	now := time.Date(2026, time.August, 17, 2, 50, 0, 0, time.UTC)
	membership := registry.New(time.Minute, registry.WithClock(func() time.Time { return now }))
	profiles, err := placement.NewProfileStore(placement.ProfileConfig{
		MaxAge: time.Minute, MaxSeries: 16,
	})
	if err != nil {
		t.Fatal(err)
	}
	callers := map[string]workerproc.PersistentCaller{}
	for _, id := range []string{"worker-a", "worker-b"} {
		registration := schedulerRegistration(id)
		registration.Capabilities.Admission.MaxOpenSequencesPerShard = 1
		if _, err := membership.Register(registration); err != nil {
			t.Fatal(err)
		}
		callers[id] = contextBlockingWorker{}
	}
	scheduler, err := NewSequenceScheduler(
		membership,
		profiles,
		TargetResolverFunc(func(binding TargetBinding) (workerproc.PersistentCaller, error) {
			return callers[binding.WorkerID], nil
		}),
		WithPrepareTimeout(20*time.Millisecond),
	)
	if err != nil {
		t.Fatal(err)
	}
	started := time.Now()
	if sequence, _, err := scheduler.Prepare(
		context.Background(), schedulerPlanRequest(), nil, generation.PlannedSessionConfig{},
	); sequence != nil || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("bounded prepare: sequence=%v err=%v", sequence, err)
	}
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("bounded prepare took %s", elapsed)
	}
	for _, id := range []string{"worker-a", "worker-b"} {
		callers[id] = newSchedulerMetadataWorker()
	}
	recovered, _, err := scheduler.Prepare(
		context.Background(), schedulerPlanRequest(), nil, generation.PlannedSessionConfig{},
	)
	if err != nil {
		t.Fatalf("failed setup retained reservation: %v", err)
	}
	if err := recovered.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestSequenceSchedulerReservesWorkerRequestCapacity(t *testing.T) {
	now := time.Date(2026, time.August, 17, 2, 55, 0, 0, time.UTC)
	registration := schedulerRegistration("worker-a")
	registration.Capabilities.Admission.MaxConcurrentRequests = 1
	inventory := schedulerInventory(now, registration)
	scheduler := reservationTestScheduler(inventory)

	first, err := scheduler.reserveAdmission(
		inventory, reservationEvaluation("worker-a", "shard-a", 100, 10, false),
	)
	if err != nil {
		t.Fatal(err)
	}
	if second, err := scheduler.reserveAdmission(
		inventory, reservationEvaluation("worker-a", "shard-b", 100, 10, false),
	); second != nil || !errors.Is(err, ErrWorkerCapacityReserved) {
		t.Fatalf("worker-wide admission: reservation=%v err=%v", second, err)
	}
	first.finish(reservationOutcome{cleanupConfirmed: true})
	second, err := scheduler.reserveAdmission(
		inventory, reservationEvaluation("worker-a", "shard-b", 100, 10, false),
	)
	if err != nil {
		t.Fatalf("released worker request slot remained reserved: %v", err)
	}
	second.finish(reservationOutcome{cleanupConfirmed: true})
}

func TestSequenceSchedulerIncludesReportedRequestsInWorkerCapacity(t *testing.T) {
	now := time.Date(2026, time.August, 17, 2, 55, 15, 0, time.UTC)
	registration := schedulerRegistration("worker-a")
	registration.Capabilities.Admission.MaxConcurrentRequests = 1
	registration.Status.OpenSequenceCount = 1
	registration.Status.RetainedShards = []registry.RetainedShard{{
		ID: "shard-a", ModelID: "test-model", CheckpointFingerprint: "test-checkpoint",
		LayerStart: 0, LayerEnd: 1, MemoryBytes: 100, OpenSequenceCount: 1,
	}}
	inventory := schedulerInventory(now, registration)
	scheduler := reservationTestScheduler(inventory)

	reservation, err := scheduler.reserveAdmission(
		inventory, reservationEvaluation("worker-a", "shard-b", 100, 10, false),
	)
	if reservation != nil || !errors.Is(err, ErrWorkerCapacityReserved) {
		t.Fatalf("reported worker request was ignored: reservation=%v err=%v", reservation, err)
	}
}

func TestSequenceSchedulerRequiresOrderedWorkerObservations(t *testing.T) {
	now := time.Date(2026, time.August, 17, 2, 55, 30, 0, time.UTC)
	registration := schedulerRegistration("worker-a")
	registration.Status.WorkerObservationSequence = 0
	inventory := schedulerInventory(now, registration)
	scheduler := reservationTestScheduler(inventory)
	reservation, err := scheduler.reserveAdmission(
		inventory, reservationEvaluation("worker-a", "shard-a", 100, 10, false),
	)
	if reservation != nil || !errors.Is(err, ErrWorkerObservationUnavailable) {
		t.Fatalf("unordered worker admission: reservation=%v err=%v", reservation, err)
	}
}

func TestSequenceSchedulerBridgesStaleMemoryStatus(t *testing.T) {
	now := time.Date(2026, time.August, 17, 2, 56, 0, 0, time.UTC)
	registration := schedulerRegistration("worker-a")
	registration.Capabilities.Admission.MaxConcurrentRequests = 2
	registration.Capabilities.Admission.RetainedByteBudget = 1_000
	registration.Status.AvailableMemoryBytes = 1_000
	inventory := schedulerInventory(now, registration)
	scheduler := reservationTestScheduler(inventory)

	loaded, err := scheduler.reserveAdmission(
		inventory, reservationEvaluation("worker-a", "shard-a", 600, 100, false),
	)
	if err != nil {
		t.Fatal(err)
	}
	loaded.stages[0].setupObserved = true
	loaded.stages[0].setupShardPresent = true
	loaded.stages[0].setupObservationSequence = 10
	loaded.finish(reservationOutcome{mayHaveLoaded: true, cleanupConfirmed: true})
	if next, err := scheduler.reserveAdmission(
		inventory, reservationEvaluation("worker-a", "shard-b", 400, 100, false),
	); next != nil || !errors.Is(err, ErrWorkerMemoryReserved) {
		t.Fatalf("stale memory admission: reservation=%v err=%v", next, err)
	}

	lateStale := inventory
	lateStale.Workers = append([]registry.Worker(nil), inventory.Workers...)
	lateStale.Workers[0].StatusObservedAt = now.Add(time.Second)
	lateStale.Workers[0].Status.WorkerObservationSequence = 9
	if next, err := scheduler.reserveAdmission(
		lateStale, reservationEvaluation("worker-a", "shard-b", 400, 100, false),
	); next != nil || !errors.Is(err, ErrWorkerMemoryReserved) {
		t.Fatalf("late delivery of pre-load memory cleared reservation: reservation=%v err=%v", next, err)
	}

	fresh := inventory
	fresh.Workers = append([]registry.Worker(nil), inventory.Workers...)
	fresh.Workers[0].StatusObservedAt = now.Add(time.Second)
	fresh.Workers[0].Status.WorkerObservationSequence = 12
	next, err := scheduler.reserveAdmission(
		fresh, reservationEvaluation("worker-a", "shard-b", 400, 100, false),
	)
	if err != nil {
		t.Fatalf("fresh status did not reconcile pending memory: %v", err)
	}
	next.finish(reservationOutcome{cleanupConfirmed: true})
}

func TestSequenceSchedulerReservesRetainedMemoryBudget(t *testing.T) {
	now := time.Date(2026, time.August, 17, 2, 57, 0, 0, time.UTC)
	registration := schedulerRegistration("worker-a")
	registration.Capabilities.Admission.MaxConcurrentRequests = 2
	registration.Capabilities.Admission.RetainedByteBudget = 150
	inventory := schedulerInventory(now, registration)
	scheduler := reservationTestScheduler(inventory)

	first, err := scheduler.reserveAdmission(
		inventory, reservationEvaluation("worker-a", "shard-a", 0, 100, true),
	)
	if err != nil {
		t.Fatal(err)
	}
	if second, err := scheduler.reserveAdmission(
		inventory, reservationEvaluation("worker-a", "shard-b", 0, 100, true),
	); second != nil || !errors.Is(err, ErrWorkerMemoryReserved) {
		t.Fatalf("retained-budget admission: reservation=%v err=%v", second, err)
	}
	first.finish(reservationOutcome{cleanupConfirmed: true})
}

func TestSequenceSchedulerRetriesProfileSnapshotRace(t *testing.T) {
	now := time.Date(2026, time.August, 17, 3, 0, 0, 0, time.UTC)
	inventory := schedulerInventory(now, schedulerRegistration("worker-a"), schedulerRegistration("worker-b"))
	profiles := &racingProfileSource{now: now}
	scheduler, err := NewSequenceScheduler(
		staticInventorySource{inventory: inventory},
		profiles,
		TargetResolverFunc(func(TargetBinding) (workerproc.PersistentCaller, error) {
			return newSchedulerMetadataWorker(), nil
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	sequence, _, err := scheduler.Prepare(
		context.Background(), schedulerPlanRequest(), nil, generation.PlannedSessionConfig{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := sequence.Close(); err != nil {
		t.Fatal(err)
	}
	if profiles.calls != 2 {
		t.Fatalf("profile snapshot calls = %d, want 2", profiles.calls)
	}
}

type staticInventorySource struct {
	inventory registry.Inventory
}

func (source staticInventorySource) Snapshot() registry.Inventory { return source.inventory }

func reservationTestScheduler(inventory registry.Inventory) *SequenceScheduler {
	return &SequenceScheduler{
		inventory:    staticInventorySource{inventory: inventory},
		reservations: make(map[sequenceReservationKey]sequenceReservationState),
		workers:      make(map[workerReservationKey]workerReservationState),
	}
}

func reservationEvaluation(
	workerID string,
	shardID string,
	loadMemoryBytes uint64,
	sequenceMemoryBytes uint64,
	reusesRetainedShard bool,
) placement.PlanEvaluation {
	required := sequenceMemoryBytes
	if !reusesRetainedShard {
		required += loadMemoryBytes
	}
	return placement.PlanEvaluation{
		Plan: generation.ExecutionPlan{Stages: []generation.ExecutionStage{{
			TargetID: workerID, ShardID: shardID,
		}}},
		Request: placement.PlanScoringRequest{Stages: []placement.StageCostEstimate{{
			LoadMemoryBytes: loadMemoryBytes, SequenceMemoryBytes: sequenceMemoryBytes,
		}}},
		Stages: []placement.StagePlanEvaluation{{
			SelectedCandidate: placement.Candidate{
				WorkerID: workerID, ReusesRetainedShard: reusesRetainedShard,
				RequiredAdditionalMemoryBytes: required,
			},
		}},
	}
}

type racingProfileSource struct {
	now   time.Time
	calls int
}

func (source *racingProfileSource) Snapshot(at time.Time) (placement.ProfileSnapshot, error) {
	source.calls++
	if source.calls == 1 {
		return placement.ProfileSnapshot{}, placement.ErrProfileSnapshotRewind
	}
	store, err := placement.NewProfileStore(placement.ProfileConfig{
		MaxAge: time.Minute, MaxSeries: 16,
	})
	if err != nil {
		return placement.ProfileSnapshot{}, err
	}
	return store.Snapshot(at)
}

func schedulerPlanRequest() placement.PlanConstructionRequest {
	return placement.PlanConstructionRequest{
		Model: generation.ExecutionModel{
			ID: "test/model", CheckpointFingerprint: "test-checkpoint", LayerCount: 4,
		},
		Scoring: placement.PlanScoringRequest{
			Adapter: "test-adapter",
			Transport: placement.TransportRequirement{
				Protocol: "http-json-v1", TensorEncoding: "base64-json",
			},
			CoordinatorID: "coordinator", CoordinatorInstanceID: "coordinator-instance",
			PrefillInputTokens: 4, DecodeSteps: 2,
			FallbackRTTMicros: 100, FallbackBytesPerSecond: 1_000_000,
		},
		TerminalResponseMode: generation.StageResponseSampledToken,
		MaxStages:            2, MaxSearchOperations: 100,
		Ranges: []placement.RangeCostEstimate{
			schedulerRange(0, 2, 600, 100),
			schedulerRange(2, 4, 600, 100),
			schedulerRange(0, 4, 12_000, 200),
		},
	}
}

func schedulerRange(start, end int, load, compute uint64) placement.RangeCostEstimate {
	return placement.RangeCostEstimate{
		LayerStart: start, LayerEnd: end,
		Estimate: placement.StageCostEstimate{
			LoadMemoryBytes: load, SequenceMemoryBytes: 100,
			PrefillWireBytes: 100, DecodeWireBytesPerStep: 10,
			FallbackPrefillComputeMicros:       compute,
			FallbackDecodeComputeMicrosPerStep: compute / 10,
		},
	}
}

func registerSchedulerWorker(t *testing.T, membership *registry.Registry, id string) {
	t.Helper()
	if _, err := membership.Register(schedulerRegistration(id)); err != nil {
		t.Fatal(err)
	}
}

func updateSchedulerStatus(
	t *testing.T,
	membership *registry.Registry,
	id string,
	update func(*registry.Status),
) {
	t.Helper()
	for _, worker := range membership.Snapshot().Workers {
		if worker.ID != id {
			continue
		}
		status := worker.Status
		update(&status)
		if _, err := membership.Heartbeat(id, registry.Heartbeat{
			SchemaVersion: registry.SchemaVersion,
			InstanceID:    worker.InstanceID, Status: &status,
		}); err != nil {
			t.Fatal(err)
		}
		return
	}
	t.Fatalf("worker %q not found", id)
}

func schedulerRegistration(id string) registry.Registration {
	return registry.Registration{
		SchemaVersion: registry.SchemaVersion,
		ID:            id, InstanceID: id + "-instance", Endpoint: "http://" + id + ".example:8080",
		Capabilities: registry.Capabilities{
			Backend: "test", Runtime: "fixture", OS: "darwin", Architecture: "arm64",
			Device: "test-device", PhysicalMemoryBytes: 20_000,
			Adapters: []string{"test-adapter"},
			Operations: []string{
				"closeSequence", "decode", "detokenize", "loadShard", "modelInfo",
				"openSequence", "prefill", "state", "tokenize",
			},
			CheckpointFingerprints: []string{"test-checkpoint"},
			Admission: registry.AdmissionLimits{
				MaxConcurrentRequests: 4, MaxOpenSequencesPerShard: 2,
				RetainedByteBudget: 10_000,
			},
			Transports: []registry.Transport{{
				Protocol: "http-json-v1", TensorEncodings: []string{"base64-json"},
				MaxRequestBytes: 1 << 20,
			}},
		},
		Status: registry.Status{
			Health: registry.HealthHealthy, AvailableMemoryBytes: 10_000,
			WorkerObservationSequence: 1,
		},
	}
}

func schedulerInventory(at time.Time, registrations ...registry.Registration) registry.Inventory {
	workers := make([]registry.Worker, len(registrations))
	for index, registration := range registrations {
		workers[index] = registry.Worker{
			Registration: registration, RegisteredAt: at, LastSeen: at,
			ExpiresAt: at.Add(time.Minute), StatusObservedAt: at,
		}
	}
	return registry.Inventory{
		SchemaVersion: registry.SchemaVersion, Revision: 1, GeneratedAt: at,
		LeaseTTLMillis: time.Minute.Milliseconds(), Workers: workers,
	}
}

func planTargetIDs(plan generation.ExecutionPlan) []string {
	targets := make([]string, len(plan.Stages))
	for index, stage := range plan.Stages {
		targets[index] = stage.TargetID
	}
	return targets
}

type schedulerMetadataWorker struct {
	mu                  sync.Mutex
	loaded              []workerproc.PersistentShardSnapshot
	observationSequence uint64
}

type contextBlockingWorker struct{}

func (contextBlockingWorker) Call(
	ctx context.Context,
	_ workerproc.PersistentRequest,
) (workerproc.PersistentResponse, error) {
	<-ctx.Done()
	return workerproc.PersistentResponse{}, ctx.Err()
}

func newSchedulerMetadataWorker() *schedulerMetadataWorker { return &schedulerMetadataWorker{} }

func (worker *schedulerMetadataWorker) Call(
	ctx context.Context,
	request workerproc.PersistentRequest,
) (workerproc.PersistentResponse, error) {
	if err := ctx.Err(); err != nil {
		return workerproc.PersistentResponse{}, err
	}
	worker.mu.Lock()
	defer worker.mu.Unlock()
	worker.observationSequence++
	result := &workerproc.PersistentWorkerResult{}
	switch request.Command {
	case "modelInfo":
		result.Model = &workerproc.PersistentModelResult{
			ModelID: request.Model.ModelID, ModelType: "test", LayerCount: 4,
			CheckpointFingerprint: "test-checkpoint", CheckpointBytes: 1_600,
		}
	case "state":
		result.State = &workerproc.PersistentWorkerState{
			LoadedShards: append([]workerproc.PersistentShardSnapshot(nil), worker.loaded...),
		}
	case "loadShard":
		load := request.LoadShard
		if load == nil {
			return workerproc.PersistentResponse{}, errors.New("missing load request")
		}
		snapshot := workerproc.PersistentShardSnapshot{
			ShardID: load.ShardID, ModelID: load.ModelID, ModelType: "test",
			CheckpointFingerprint: load.CheckpointFingerprint,
			LayerStart:            load.LayerStart, LayerEnd: load.LayerEnd,
			OwnsInput: load.OwnsInput, OwnsOutput: load.OwnsOutput,
		}
		worker.loaded = append(worker.loaded, snapshot)
		result.Shard = &snapshot
	default:
		return workerproc.PersistentResponse{}, fmt.Errorf("unexpected command %q", request.Command)
	}
	return workerproc.PersistentResponse{
		OK: true, WorkerObservationSequence: worker.observationSequence, Result: result,
	}, nil
}

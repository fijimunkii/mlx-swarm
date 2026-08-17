package placement

import (
	"math"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/generation"
)

func TestProfileStoreLinkSnapshotIsBoundedAndDeterministic(t *testing.T) {
	now := time.Date(2026, time.August, 17, 0, 0, 0, 0, time.UTC)
	observations := []LinkObservation{
		testLinkObservation("worker-a", now.Add(-3*time.Second), 100, 1000),
		testLinkObservation("worker-a", now.Add(-time.Second), 300, 3000),
		testLinkObservation("worker-a", now.Add(-2*time.Second), 200, 2000),
		testLinkObservation("worker-b", now.Add(-time.Second), 50, 0),
	}
	build := func(input []LinkObservation) ProfileSnapshot {
		store := mustProfileStore(t, ProfileConfig{
			MaxAge: 10 * time.Second, MaxSamplesPerSeries: 2, MaxSeries: 4,
		})
		for _, observation := range input {
			if err := store.ObserveLink(now, observation); err != nil {
				t.Fatal(err)
			}
		}
		return mustProfileSnapshot(t, store, now)
	}
	first := build(observations)
	reversed := append([]LinkObservation(nil), observations...)
	slices.Reverse(reversed)
	second := build(reversed)
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("profile depends on observation order:\nfirst=%+v\nsecond=%+v", first, second)
	}
	if first.SchemaVersion != SchemaVersion || first.Revision != 4 ||
		first.MaxAgeMillis != 10_000 || first.MaxSamplesPerSeries != 2 || first.MaxSeries != 4 ||
		len(first.Links) != 2 || len(first.Compute) != 0 {
		t.Fatalf("unexpected snapshot: %+v", first)
	}
	workerA := first.Links[0]
	if workerA.TargetID != "worker-a" || workerA.LatestObservedAt != now.Add(-time.Second) ||
		workerA.RTTMicros != (ValueDistribution{Count: 2, Min: 200, P50: 200, P95: 300, Max: 300}) ||
		workerA.EffectiveBytesPerSecond != (ValueDistribution{
			Count: 2, Min: 20_000_000, P50: 20_000_000, P95: 30_000_000, Max: 30_000_000,
		}) {
		t.Fatalf("unexpected worker-a profile: %+v", workerA)
	}
	workerB := first.Links[1]
	if workerB.TargetID != "worker-b" || workerB.RTTMicros.Count != 1 ||
		workerB.EffectiveBytesPerSecond.Count != 0 {
		t.Fatalf("unexpected worker-b profile: %+v", workerB)
	}
}

func TestProfileStoreExpiresAndRejectsInvalidObservationTimes(t *testing.T) {
	now := time.Date(2026, time.August, 17, 0, 0, 0, 0, time.UTC)
	store := mustProfileStore(t, ProfileConfig{
		MaxAge: 5 * time.Second, MaxSamplesPerSeries: 8, MaxSeries: 2,
	})
	for _, observation := range []LinkObservation{
		testLinkObservation("worker-a", now.Add(-5*time.Second), 20, 0),
		testLinkObservation("worker-a", now, 30, 0),
	} {
		if err := store.ObserveLink(now, observation); err != nil {
			t.Fatal(err)
		}
	}
	current := mustProfileSnapshot(t, store, now)
	if len(current.Links) != 1 || current.Links[0].RTTMicros.Count != 2 ||
		current.Links[0].RTTMicros.P50 != 20 {
		t.Fatalf("boundary observation was not retained: %+v", current)
	}
	if err := store.ObserveLink(now, testLinkObservation(
		"worker-a", now.Add(time.Second), 40, 0,
	)); err == nil {
		t.Fatal("future-dated observation was accepted")
	}
	if err := store.ObserveLink(now, testLinkObservation(
		"worker-a", now.Add(-6*time.Second), 10, 0,
	)); err == nil {
		t.Fatal("already-stale observation was accepted")
	}
	if _, err := store.Snapshot(now.Add(-time.Nanosecond)); err == nil {
		t.Fatal("snapshot rewound before the latest accepted update")
	}
	expired := mustProfileSnapshot(t, store, now.Add(6*time.Second))
	if len(expired.Links) != 0 {
		t.Fatalf("expired observations remain visible: %+v", expired)
	}
}

func TestProfileStoreIngestsPlannedComputeAtomically(t *testing.T) {
	now := time.Date(2026, time.August, 17, 0, 0, 0, 0, time.UTC)
	plan := testProfilePlan(t)
	workerA := testWorker("worker-a", now)
	workerA.Capabilities.Backend = "cuda"
	workerB := testWorker("worker-b", now)
	workerB.Capabilities.Backend = "mlx"
	inventory := testInventory(now, workerB, workerA)
	sample := testPlannedSample(plan)
	store := mustProfileStore(t, ProfileConfig{MaxAge: time.Minute, MaxSeries: 4})
	if err := store.ObservePlannedSample(now, inventory, plan, sample); err != nil {
		t.Fatal(err)
	}
	snapshot := mustProfileSnapshot(t, store, now)
	if snapshot.Revision != 1 || len(snapshot.Compute) != 2 {
		t.Fatalf("unexpected planned profile: %+v", snapshot)
	}
	if got := []string{snapshot.Compute[0].WorkerID, snapshot.Compute[1].WorkerID}; !slices.Equal(got, []string{"worker-a", "worker-b"}) {
		t.Fatalf("compute profile order = %v", got)
	}
	if snapshot.Compute[0].Backend != "cuda" || snapshot.Compute[0].LayerStart != 6 ||
		snapshot.Compute[0].WorkerInstanceID != "worker-a-instance" ||
		snapshot.Compute[0].ComputeMicros.P50 != 200 ||
		snapshot.Compute[1].Backend != "mlx" || snapshot.Compute[1].LayerEnd != 6 ||
		snapshot.Compute[1].ComputeMicros.P50 != 100 {
		t.Fatalf("planned sample was not mapped to workers: %+v", snapshot.Compute)
	}

	invalid := sample
	invalid.Stages = append([]generation.StageExecution(nil), sample.Stages...)
	invalid.Stages[1].ComputeMicros = 0
	if err := store.ObservePlannedSample(now, inventory, plan, invalid); err == nil {
		t.Fatal("inconsistent planned sample was accepted")
	}
	mismatchedInventory := inventory
	mismatchedInventory.Revision++
	if err := store.ObservePlannedSample(now, mismatchedInventory, plan, sample); err == nil {
		t.Fatal("sample from a different inventory revision was accepted")
	}
	unpinnedPlan, err := generation.BuildExecutionPlan(plan.Model, "", []generation.ExecutionStage{
		{Name: "stage-0", TargetID: "worker-b", LayerStart: 0, LayerEnd: 6, OwnsInput: true, ResponseMode: generation.StageResponseTensor},
		{Name: "stage-1", TargetID: "worker-a", LayerStart: 6, LayerEnd: 12, OwnsOutput: true, ResponseMode: generation.StageResponseSampledToken},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.ObservePlannedSample(
		now, inventory, unpinnedPlan, testPlannedSample(unpinnedPlan),
	); err == nil {
		t.Fatal("sample from an unpinned plan was accepted")
	}
	afterInvalid := mustProfileSnapshot(t, store, now)
	if !reflect.DeepEqual(snapshot, afterInvalid) {
		t.Fatalf("invalid batch partially mutated profile:\nbefore=%+v\nafter=%+v", snapshot, afterInvalid)
	}

	limited := mustProfileStore(t, ProfileConfig{MaxAge: time.Minute, MaxSeries: 1})
	if err := limited.ObservePlannedSample(now, inventory, plan, sample); err == nil {
		t.Fatal("multi-stage batch exceeded the series limit")
	}
	limitedSnapshot := mustProfileSnapshot(t, limited, now)
	if limitedSnapshot.Revision != 0 || len(limitedSnapshot.Compute) != 0 {
		t.Fatalf("rejected batch partially mutated limited store: %+v", limitedSnapshot)
	}
}

func TestProfileStoreIsConcurrencySafeAndKeepsNewestSamples(t *testing.T) {
	base := time.Date(2026, time.August, 17, 0, 0, 0, 0, time.UTC)
	store := mustProfileStore(t, ProfileConfig{
		MaxAge: time.Minute, MaxSamplesPerSeries: 8, MaxSeries: 1,
	})
	var wait sync.WaitGroup
	errors := make(chan error, 100)
	for index := range 100 {
		wait.Add(1)
		go func() {
			defer wait.Done()
			errors <- store.ObserveCompute(base.Add(99*time.Millisecond), ComputeObservation{
				WorkerID: "worker-a", WorkerInstanceID: "worker-a-instance",
				Backend: "mlx", Model: testProfileModel(),
				Operation: "decode", LayerStart: 0, LayerEnd: 6,
				InputTokenCount: 1, ComputeMicros: uint64(index + 1),
				ObservedAt: base.Add(time.Duration(index) * time.Millisecond),
			})
		}()
	}
	wait.Wait()
	close(errors)
	for err := range errors {
		if err != nil {
			t.Fatal(err)
		}
	}
	snapshot := mustProfileSnapshot(t, store, base.Add(99*time.Millisecond))
	if snapshot.Revision != 100 || len(snapshot.Compute) != 1 ||
		snapshot.Compute[0].ComputeMicros != (ValueDistribution{
			Count: 8, Min: 93, P50: 96, P95: 100, Max: 100,
		}) {
		t.Fatalf("concurrent rolling profile is wrong: %+v", snapshot)
	}
}

func TestProfileStoreReclaimsExpiredSeriesDuringIngestion(t *testing.T) {
	now := time.Date(2026, time.August, 17, 0, 0, 0, 0, time.UTC)
	store := mustProfileStore(t, ProfileConfig{
		MaxAge: 5 * time.Second, MaxSamplesPerSeries: 2, MaxSeries: 1,
	})
	if err := store.ObserveLink(now, testLinkObservation("worker-a", now, 10, 0)); err != nil {
		t.Fatal(err)
	}
	computeTime := now.Add(6 * time.Second)
	if err := store.ObserveCompute(computeTime, ComputeObservation{
		WorkerID: "worker-b", WorkerInstanceID: "worker-b-instance",
		Backend: "cuda", Model: testProfileModel(), Operation: "decode",
		LayerStart: 6, LayerEnd: 12, InputTokenCount: 1,
		ComputeMicros: 20, ObservedAt: computeTime,
	}); err != nil {
		t.Fatalf("fresh series was rejected behind expired capacity: %v", err)
	}
	snapshot := mustProfileSnapshot(t, store, computeTime)
	if snapshot.Revision != 2 || len(snapshot.Links) != 0 || len(snapshot.Compute) != 1 ||
		snapshot.Compute[0].WorkerID != "worker-b" {
		t.Fatalf("expired series was not reclaimed: %+v", snapshot)
	}
	if err := store.ObserveLink(now, testLinkObservation("worker-c", now, 30, 0)); err == nil {
		t.Fatal("out-of-order acceptance revived evidence stale at the newest store time")
	}
}

func TestProfileStoreRejectsInvalidInputsAndSeriesOverflow(t *testing.T) {
	now := time.Date(2026, time.August, 17, 0, 0, 0, 0, time.UTC)
	for _, config := range []ProfileConfig{
		{MaxAge: -time.Second},
		{MaxAge: time.Microsecond},
		{MaxSamplesPerSeries: -1},
		{MaxSeries: -1},
	} {
		if _, err := NewProfileStore(config); err == nil {
			t.Fatalf("invalid config was accepted: %+v", config)
		}
	}
	store := mustProfileStore(t, ProfileConfig{MaxAge: time.Minute, MaxSeries: 1})
	validLink := testLinkObservation("worker-a", now, 10, 0)
	invalidLinks := []LinkObservation{
		{},
		func() LinkObservation { value := validLink; value.TargetInstanceID = ""; return value }(),
		func() LinkObservation { value := validLink; value.TargetID = value.SourceID; return value }(),
		func() LinkObservation { value := validLink; value.RTTMicros = 0; return value }(),
		func() LinkObservation { value := validLink; value.ElapsedMicros = 1; return value }(),
		func() LinkObservation {
			value := validLink
			value.Protocol = strings.Repeat("x", maxProfileLabelBytes+1)
			return value
		}(),
	}
	for _, observation := range invalidLinks {
		if err := store.ObserveLink(now, observation); err == nil {
			t.Fatalf("invalid link was accepted: %+v", observation)
		}
	}
	validCompute := ComputeObservation{
		WorkerID: "worker-a", WorkerInstanceID: "worker-a-instance",
		Backend: "mlx", Model: testProfileModel(),
		Operation: "decode", LayerStart: 0, LayerEnd: 6,
		InputTokenCount: 1, ComputeMicros: 10, ObservedAt: now,
	}
	invalidComputes := []ComputeObservation{
		{},
		func() ComputeObservation { value := validCompute; value.WorkerInstanceID = ""; return value }(),
		func() ComputeObservation { value := validCompute; value.LayerEnd = 13; return value }(),
		func() ComputeObservation { value := validCompute; value.InputTokenCount = 0; return value }(),
		func() ComputeObservation { value := validCompute; value.ComputeMicros = 0; return value }(),
	}
	for _, observation := range invalidComputes {
		if err := store.ObserveCompute(now, observation); err == nil {
			t.Fatalf("invalid compute observation was accepted: %+v", observation)
		}
	}
	if err := store.ObserveLink(time.Time{}, validLink); err == nil {
		t.Fatal("zero acceptance time was accepted")
	}
	if _, err := store.Snapshot(time.Time{}); err == nil {
		t.Fatal("zero snapshot time was accepted")
	}
	if err := store.ObserveLink(now, validLink); err != nil {
		t.Fatal(err)
	}
	if err := store.ObserveCompute(now, validCompute); err == nil {
		t.Fatal("new compute series exceeded combined series limit")
	}
	snapshot := mustProfileSnapshot(t, store, now)
	if snapshot.Revision != 1 || len(snapshot.Links) != 1 || len(snapshot.Compute) != 0 {
		t.Fatalf("series overflow mutated store: %+v", snapshot)
	}
	if got := bytesPerSecond(math.MaxUint64, 1); got != math.MaxUint64 {
		t.Fatalf("overflowing throughput = %d", got)
	}
}

func testLinkObservation(
	target string,
	observedAt time.Time,
	rttMicros uint64,
	payloadBytes uint64,
) LinkObservation {
	elapsed := uint64(0)
	if payloadBytes > 0 {
		elapsed = 100
	}
	return LinkObservation{
		SourceID: "coordinator", SourceInstanceID: "coordinator-run",
		TargetID: target, TargetInstanceID: target + "-instance",
		Protocol: "http-json-v1", TensorEncoding: "base64-json",
		ObservedAt: observedAt, RTTMicros: rttMicros,
		PayloadBytes: payloadBytes, ElapsedMicros: elapsed,
	}
}

func testProfileModel() generation.ExecutionModel {
	return generation.ExecutionModel{
		ID: "model-a", CheckpointFingerprint: "sha256:model-a", LayerCount: 12,
	}
}

func testProfilePlan(t *testing.T) generation.ExecutionPlan {
	t.Helper()
	plan, err := generation.BuildExecutionPlan(testProfileModel(), "7", []generation.ExecutionStage{
		{Name: "stage-0", TargetID: "worker-b", LayerStart: 0, LayerEnd: 6, OwnsInput: true, ResponseMode: generation.StageResponseTensor},
		{Name: "stage-1", TargetID: "worker-a", LayerStart: 6, LayerEnd: 12, OwnsOutput: true, ResponseMode: generation.StageResponseSampledToken},
	})
	if err != nil {
		t.Fatal(err)
	}
	return plan
}

func testPlannedSample(plan generation.ExecutionPlan) generation.PlannedStageSample {
	return generation.PlannedStageSample{
		Operation: "prefill", InputTokenCount: 4,
		Stages: []generation.StageExecution{
			{Index: 0, Stage: plan.Stages[0], Operation: "prefill", ComputeMicros: 100},
			{Index: 1, Stage: plan.Stages[1], Operation: "prefill", ComputeMicros: 200},
		},
	}
}

func mustProfileStore(t *testing.T, config ProfileConfig) *ProfileStore {
	t.Helper()
	store, err := NewProfileStore(config)
	if err != nil {
		t.Fatal(err)
	}
	return store
}

func mustProfileSnapshot(t *testing.T, store *ProfileStore, at time.Time) ProfileSnapshot {
	t.Helper()
	snapshot, err := store.Snapshot(at)
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

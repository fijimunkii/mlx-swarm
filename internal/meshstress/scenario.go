package meshstress

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/placement"
	"github.com/fijimunkii/mlx-swarm/internal/registry"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

const (
	proofAdapter             = "synthetic-transformer"
	proofCheckpoint          = "synthetic-checkpoint-v1"
	proofProtocol            = workerproc.InstanceBoundHTTPProtocol
	proofEncoding            = workerproc.Base64JSONTensorEncoding
	proofCoordinator         = "linux-coordinator"
	proofCoordinatorInstance = "synthetic-proof"
)

var proofModel = generation.ExecutionModel{
	ID: "synthetic/control-plane-model", CheckpointFingerprint: proofCheckpoint,
	LayerCount: 12,
}

type lockedClock struct {
	mu  sync.RWMutex
	now time.Time
}

func (clock *lockedClock) Time() time.Time {
	clock.mu.RLock()
	defer clock.mu.RUnlock()
	return clock.now
}

func (clock *lockedClock) Advance(delta time.Duration) time.Time {
	clock.mu.Lock()
	defer clock.mu.Unlock()
	clock.now = clock.now.Add(delta)
	return clock.now
}

type countingTransport struct {
	base  http.RoundTripper
	calls atomic.Uint64
}

func (transport *countingTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	transport.calls.Add(1)
	return transport.base.RoundTrip(request)
}

// Run executes the deterministic scale/churn scenario through the real
// membership HTTP protocol and production placement engine.
func Run(ctx context.Context, config Config) (Result, error) {
	result := Result{SchemaVersion: SchemaVersion, Model: proofModel}
	if err := normalizeConfig(&config); err != nil {
		return result, err
	}
	clock := &lockedClock{now: time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)}
	membership := registry.New(2*time.Second, registry.WithClock(clock.Time))
	server := httptest.NewServer(registry.NewHTTPHandler(membership))
	defer server.Close()
	transport := &countingTransport{base: http.DefaultTransport}
	client, err := registry.NewClient(server.URL, &http.Client{Transport: transport})
	if err != nil {
		return result, err
	}
	ranges := proofRanges()
	profiles, err := placement.NewProfileStore(placement.ProfileConfig{
		MaxAge: time.Minute, MaxSamplesPerSeries: 8,
		MaxSeries: (config.WorkerCount+1)*(1+2*len(ranges)) + 16,
	})
	if err != nil {
		return result, err
	}
	specs := defaultWorkerSpecs(config.WorkerCount)

	if err := registerConcurrently(ctx, client, specs[:3]); err != nil {
		return result, fmt.Errorf("join baseline synthetic workers: %w", err)
	}
	if err := observeSpecs(profiles, clock.Time(), specs[:3], ranges); err != nil {
		return result, err
	}
	if err := appendTransition(
		ctx, client, &result, "baseline_join", "three eligible workers joined concurrently",
	); err != nil {
		return result, err
	}
	baseline, err := decide(ctx, client, profiles, config, "baseline", ranges)
	if err != nil {
		return result, err
	}
	result.Decisions = append(result.Decisions, baseline)

	if err := registerConcurrently(ctx, client, specs[3:]); err != nil {
		return result, fmt.Errorf("join scaled synthetic workers: %w", err)
	}
	if err := observeSpecs(profiles, clock.Time(), specs[3:], ranges); err != nil {
		return result, err
	}
	scaledTransition, err := transition(
		ctx, client, "scaled_join", fmt.Sprintf("%d workers visible", config.WorkerCount),
	)
	if err != nil {
		return result, err
	}
	result.Transitions = append(result.Transitions, scaledTransition)
	scaled, err := decide(ctx, client, profiles, config, "scaled", ranges)
	if err != nil {
		return result, err
	}
	deterministic, err := decide(ctx, client, profiles, config, "deterministic_repeat", ranges)
	if err != nil {
		return result, err
	}
	result.Decisions = append(result.Decisions, scaled, deterministic)

	duplicate := specs[0]
	duplicate.InstanceID = "conflicting-instance"
	_, duplicateErr := client.Register(ctx, duplicate.Registration(), true)
	duplicateRejected := remoteCode(duplicateErr) == "duplicate_worker"
	if err := appendTransition(
		ctx, client, &result, "duplicate_identity",
		"active stable identity rejected a second instance",
	); err != nil {
		return result, err
	}

	expiring := &specs[len(specs)-1]
	clock.Advance(time.Second)
	for index := range specs[:len(specs)-1] {
		status := specs[index].Registration().Status
		if _, err := client.Heartbeat(ctx, specs[index].ID, registry.Heartbeat{
			SchemaVersion: registry.SchemaVersion, InstanceID: specs[index].InstanceID,
			Status: &status,
		}); err != nil {
			return result, fmt.Errorf("refresh synthetic worker %q: %w", specs[index].ID, err)
		}
	}
	clock.Advance(1100 * time.Millisecond)
	expired, err := transition(ctx, client, "expiry", "one unrefreshed lease expired")
	if err != nil {
		return result, err
	}
	result.Transitions = append(result.Transitions, expired)
	expiryRemoved := !slices.Contains(expired.WorkerIDs, expiring.ID)
	expiring.InstanceID += "-rejoin"
	expiring.Endpoint += "/rejoin"
	if _, err := client.Register(ctx, expiring.Registration(), true); err != nil {
		return result, fmt.Errorf("rejoin expired synthetic worker: %w", err)
	}
	if err := observeSpecs(profiles, clock.Time(), []WorkerSpec{*expiring}, ranges); err != nil {
		return result, err
	}
	rejoined, err := transition(
		ctx, client, "rejoin", "expired stable identity rejoined with a new instance",
	)
	if err != nil {
		return result, err
	}
	result.Transitions = append(result.Transitions, rejoined)

	incompatible := specs[2]
	incompatible.CheckpointFingerprints = []string{"other-checkpoint"}
	if _, err := client.Register(ctx, incompatible.Registration(), true); err != nil {
		return result, fmt.Errorf("publish incompatible capability change: %w", err)
	}
	incompatibleDecision, err := decide(ctx, client, profiles, config, "incompatible_capability", ranges)
	if err != nil {
		return result, err
	}
	result.Decisions = append(result.Decisions, incompatibleDecision)
	if err := appendTransition(
		ctx, client, &result, "capability_change",
		"eligible worker advertised an incompatible checkpoint",
	); err != nil {
		return result, err
	}
	if _, err := client.Register(ctx, specs[2].Registration(), true); err != nil {
		return result, fmt.Errorf("restore compatible capability: %w", err)
	}

	better := betterWorkerSpec()
	if _, err := client.Register(ctx, better.Registration(), true); err != nil {
		return result, fmt.Errorf("join better worker: %w", err)
	}
	if err := observeSpecs(profiles, clock.Time(), []WorkerSpec{better}, ranges); err != nil {
		return result, err
	}
	if err := appendTransition(
		ctx, client, &result, "better_join", "materially better eligible worker joined",
	); err != nil {
		return result, err
	}
	betterDecision, err := decide(ctx, client, profiles, config, "better_join", ranges)
	if err != nil {
		return result, err
	}
	result.Decisions = append(result.Decisions, betterDecision)
	if err := client.Remove(ctx, better.ID, better.InstanceID); err != nil {
		return result, fmt.Errorf("remove better worker: %w", err)
	}
	if err := appendTransition(
		ctx, client, &result, "selected_remove", "selected better worker removed",
	); err != nil {
		return result, err
	}
	afterRemoval, err := decide(ctx, client, profiles, config, "after_removal", ranges)
	if err != nil {
		return result, err
	}
	result.Decisions = append(result.Decisions, afterRemoval)

	slowedID := firstTarget(afterRemoval)
	slowedSpec, found := findSpec(specs, slowedID)
	if !found {
		return result, fmt.Errorf("selected worker %q has no synthetic spec", slowedID)
	}
	slowedSpec.PrefillMicrosPerLayer *= 100
	slowedSpec.DecodeMicrosPerLayer *= 100
	for range 2 {
		if err := observeSpecs(profiles, clock.Time(), []WorkerSpec{slowedSpec}, ranges); err != nil {
			return result, err
		}
	}
	if err := appendTransition(
		ctx, client, &result, "slowdown",
		"fresh compute profiles made one selected worker materially slower",
	); err != nil {
		return result, err
	}
	afterSlowdown, err := decide(ctx, client, profiles, config, "after_slowdown", ranges)
	if err != nil {
		return result, err
	}
	result.Decisions = append(result.Decisions, afterSlowdown)

	profileSnapshot, err := profiles.Snapshot(clock.Time())
	if err != nil {
		return result, err
	}
	result.Bounds = BoundsEvidence{
		WorkerCount: config.WorkerCount, MaxSearchOperations: config.MaxSearchOperations,
		MaxDecisionMicros: config.MaxDecisionDuration.Microseconds(),
		NetworkProbeLimit: 0, NetworkProbeCount: 0,
		MembershipRequestCount:    transport.calls.Load(),
		ProfileLinkSeriesCount:    len(profileSnapshot.Links),
		ProfileComputeSeriesCount: len(profileSnapshot.Compute),
	}
	result.Checks = Checks{
		ThirtyTwoWorkersVisible:   scaledTransition.WorkerCount >= DefaultWorkerCount,
		ConcurrentJoinSucceeded:   scaledTransition.WorkerCount == config.WorkerCount,
		DuplicateIdentityRejected: duplicateRejected,
		ExpiryRemovedWorker:       expiryRemoved && expired.WorkerCount == config.WorkerCount-1,
		RejoinSucceeded: rejoined.WorkerCount == config.WorkerCount &&
			slices.Contains(rejoined.WorkerIDs, expiring.ID),
		IncompatibleChangeRejected: decisionRejects(
			incompatibleDecision, incompatible.ID, placement.RejectionIncompatibleCheckpoint,
		),
		UnsuitableWorkersPlanStable: baseline.PlanSignature == scaled.PlanSignature,
		BetterWorkerChangedPlan: betterDecision.PlanSignature != scaled.PlanSignature &&
			decisionUsesWorker(betterDecision, better.ID),
		RemovedWorkerNotReused: !decisionUsesWorker(afterRemoval, better.ID),
		SlowWorkerReplanned: afterSlowdown.PlanSignature != afterRemoval.PlanSignature &&
			!decisionUsesWorker(afterSlowdown, slowedID),
		DeterministicPlacement: scaled.PlanSignature == deterministic.PlanSignature &&
			reflect.DeepEqual(scaled.SelectedPlan, deterministic.SelectedPlan),
		RejectionReasonsRecorded: len(scaled.RejectedWorkers) == config.WorkerCount-3 &&
			recordsRequiredRejections(scaled),
		SearchWorkWithinBound: decisionsWithinSearchBound(result.Decisions, config.MaxSearchOperations),
		DecisionLatencyWithinBound: decisionsWithinTimeBound(
			result.Decisions, config.MaxDecisionDuration,
		),
		NetworkProbesWithinBound: result.Bounds.NetworkProbeCount <= result.Bounds.NetworkProbeLimit,
	}
	result.Checks.AllPassed = allChecksPassed(result.Checks)
	if !result.Checks.AllPassed {
		return result, errors.New("synthetic mesh scale/churn proof did not satisfy every check")
	}
	return result, nil
}

func normalizeConfig(config *Config) error {
	if config.WorkerCount < DefaultWorkerCount {
		return fmt.Errorf("synthetic mesh proof requires at least %d workers", DefaultWorkerCount)
	}
	if config.MaxSearchOperations == 0 {
		config.MaxSearchOperations = DefaultConfig().MaxSearchOperations
	}
	if config.MaxDecisionDuration < time.Millisecond {
		return errors.New("decision duration bound must be at least 1ms")
	}
	return nil
}

func defaultWorkerSpecs(count int) []WorkerSpec {
	specs := make([]WorkerSpec, count)
	for index := range specs {
		id := fmt.Sprintf("synthetic-%02d", index)
		specs[index] = baseWorkerSpec(id)
		specs[index].PrefillMicrosPerLayer += uint64(index * 10)
		specs[index].DecodeMicrosPerLayer += uint64(index)
		if index < 3 {
			continue
		}
		switch index % 5 {
		case 0:
			specs[index].AvailableMemoryBytes = 100
		case 1:
			specs[index].Health = registry.HealthDegraded
		case 2:
			specs[index].Adapters = []string{"incompatible-adapter"}
		case 3:
			specs[index].CheckpointFingerprints = []string{"incompatible-checkpoint"}
		case 4:
			specs[index].Transports[0].Protocol = "synthetic-control-v1"
		}
	}
	retainedIndex := len(specs) - 2
	if retainedIndex >= 3 {
		specs[retainedIndex].RetainedBytes = 200
		specs[retainedIndex].RetainedShards = []registry.RetainedShard{{
			ID: "synthetic-retained", ModelID: proofModel.ID,
			CheckpointFingerprint: "incompatible-checkpoint", LayerStart: 0, LayerEnd: 6,
			OwnsInput: true, MemoryBytes: 600,
		}}
	}
	return specs
}

func baseWorkerSpec(id string) WorkerSpec {
	return WorkerSpec{
		ID: id, InstanceID: id + "-instance", Endpoint: "http://" + id + ".invalid:8080",
		Backend: "synthetic", Runtime: "meshstress-v1", OS: "linux", Architecture: "amd64",
		Device: "synthetic-cpu", PhysicalMemoryBytes: 8_000, AvailableMemoryBytes: 2_000,
		Adapters: []string{proofAdapter}, Operations: []string{
			"closeSequence", "decode", "detokenize", "loadShard", "modelInfo",
			"openSequence", "prefill", "state", "tokenize",
		},
		CheckpointFingerprints: []string{proofCheckpoint},
		Admission: registry.AdmissionLimits{
			MaxConcurrentRequests: 4, MaxOpenSequencesPerShard: 2, RetainedByteBudget: 2_000,
		},
		Transports: []registry.Transport{{
			Protocol: proofProtocol, TensorEncodings: []string{proofEncoding},
			MaxRequestBytes: 1 << 20,
		}},
		Health: registry.HealthHealthy, RTTMicros: 100,
		BandwidthBytesPerSecond: 10_000_000,
		PrefillMicrosPerLayer:   100, DecodeMicrosPerLayer: 10,
	}
}

func betterWorkerSpec() WorkerSpec {
	spec := baseWorkerSpec("synthetic-better")
	spec.PhysicalMemoryBytes = 16_000
	spec.AvailableMemoryBytes = 8_000
	spec.RTTMicros = 20
	spec.BandwidthBytesPerSecond = 100_000_000
	spec.PrefillMicrosPerLayer = 10
	spec.DecodeMicrosPerLayer = 1
	return spec
}

func proofRanges() []placement.RangeCostEstimate {
	return []placement.RangeCostEstimate{
		proofRange(0, 4, 800), proofRange(4, 8, 800), proofRange(8, 12, 800),
		proofRange(0, 6, 1200), proofRange(6, 12, 1200), proofRange(0, 12, 5_000),
	}
}

func proofRange(start, end int, load uint64) placement.RangeCostEstimate {
	return placement.RangeCostEstimate{
		LayerStart: start, LayerEnd: end,
		Estimate: placement.StageCostEstimate{
			LoadMemoryBytes: load, SequenceMemoryBytes: 100,
			PrefillWireBytes: 4096, DecodeWireBytesPerStep: 512,
			FallbackPrefillComputeMicros:       uint64((end - start) * 1000),
			FallbackDecodeComputeMicrosPerStep: uint64((end - start) * 100),
		},
	}
}

func proofRequest(config Config, ranges []placement.RangeCostEstimate) placement.PlanConstructionRequest {
	return placement.PlanConstructionRequest{
		Model: proofModel,
		Scoring: placement.PlanScoringRequest{
			Adapter: proofAdapter,
			Transport: placement.TransportRequirement{
				Protocol: proofProtocol, TensorEncoding: proofEncoding,
			},
			CoordinatorID: proofCoordinator, CoordinatorInstanceID: proofCoordinatorInstance,
			PrefillInputTokens: 8, DecodeSteps: 4,
			FallbackRTTMicros: 500, FallbackBytesPerSecond: 1_000_000,
		},
		TerminalResponseMode: generation.StageResponseSampledToken,
		MaxStages:            3, MaxSearchOperations: config.MaxSearchOperations,
		Ranges: append([]placement.RangeCostEstimate(nil), ranges...),
	}
}

func registerConcurrently(ctx context.Context, client *registry.Client, specs []WorkerSpec) error {
	var wait sync.WaitGroup
	errorsByIndex := make([]error, len(specs))
	for index := range specs {
		wait.Add(1)
		go func(index int) {
			defer wait.Done()
			_, errorsByIndex[index] = client.Register(ctx, specs[index].Registration(), true)
		}(index)
	}
	wait.Wait()
	return errors.Join(errorsByIndex...)
}

func observeSpecs(
	profiles *placement.ProfileStore,
	at time.Time,
	specs []WorkerSpec,
	ranges []placement.RangeCostEstimate,
) error {
	for _, spec := range specs {
		payloadBytes := spec.BandwidthBytesPerSecond / 1000
		if payloadBytes == 0 {
			payloadBytes = 1
		}
		if err := profiles.ObserveLink(at, placement.LinkObservation{
			SourceID: proofCoordinator, SourceInstanceID: proofCoordinatorInstance,
			TargetID: spec.ID, TargetInstanceID: spec.InstanceID,
			Protocol: proofProtocol, TensorEncoding: proofEncoding, ObservedAt: at,
			RTTMicros: spec.RTTMicros, PayloadBytes: payloadBytes, ElapsedMicros: 1000,
		}); err != nil {
			return fmt.Errorf("observe synthetic link %q: %w", spec.ID, err)
		}
		for _, candidateRange := range ranges {
			layerCount := uint64(candidateRange.LayerEnd - candidateRange.LayerStart)
			for _, operation := range []struct {
				name   string
				tokens uint64
				micros uint64
			}{
				{name: "prefill", tokens: 8, micros: spec.PrefillMicrosPerLayer * layerCount},
				{name: "decode", tokens: 1, micros: spec.DecodeMicrosPerLayer * layerCount},
			} {
				if err := profiles.ObserveCompute(at, placement.ComputeObservation{
					WorkerID: spec.ID, WorkerInstanceID: spec.InstanceID, Backend: spec.Backend,
					Model: proofModel, Operation: operation.name,
					LayerStart: candidateRange.LayerStart, LayerEnd: candidateRange.LayerEnd,
					InputTokenCount: operation.tokens, ComputeMicros: operation.micros, ObservedAt: at,
				}); err != nil {
					return fmt.Errorf("observe synthetic compute %q: %w", spec.ID, err)
				}
			}
		}
	}
	return nil
}

func decide(
	ctx context.Context,
	client *registry.Client,
	profiles *placement.ProfileStore,
	config Config,
	name string,
	ranges []placement.RangeCostEstimate,
) (DecisionEvidence, error) {
	started := time.Now()
	inventory, err := client.Inventory(ctx)
	if err != nil {
		return DecisionEvidence{}, fmt.Errorf("%s inventory: %w", name, err)
	}
	profile, err := profiles.Snapshot(inventory.GeneratedAt)
	if err != nil {
		return DecisionEvidence{}, fmt.Errorf("%s profile: %w", name, err)
	}
	construction, err := placement.ConstructPlan(inventory, profile, proofRequest(config, ranges))
	duration := time.Since(started)
	if err != nil {
		return DecisionEvidence{}, fmt.Errorf("%s placement: %w", name, err)
	}
	if construction.SelectedPlan == nil {
		return DecisionEvidence{}, fmt.Errorf("%s placement selected no plan", name)
	}
	return DecisionEvidence{
		Name: name, InventoryRevision: inventory.Revision, ProfileRevision: profile.Revision,
		VisibleWorkerCount:       len(inventory.Workers),
		EligibleWorkerCount:      construction.CandidateWorkerCount,
		SearchOperationCount:     construction.SearchOperationCount,
		RetainedSearchStateCount: construction.RetainedSearchStateCount,
		CompletePlanCount:        construction.CompletePlanCount,
		DecisionMicros:           duration.Microseconds(),
		PlanSignature:            planSignature(construction.SelectedPlan.Plan),
		SelectedPlan:             construction.SelectedPlan,
		RejectedWorkers:          rejectedWorkers(construction),
	}, nil
}

func rejectedWorkers(construction placement.PlanConstructionResult) []WorkerRejectionEvidence {
	codes := make(map[string]map[placement.RejectionCode]struct{})
	eligible := make(map[string]bool)
	for _, candidateRange := range construction.Ranges {
		for _, candidate := range candidateRange.Eligibility.Candidates {
			if candidate.Eligible {
				eligible[candidate.WorkerID] = true
			}
			if codes[candidate.WorkerID] == nil {
				codes[candidate.WorkerID] = make(map[placement.RejectionCode]struct{})
			}
			for _, rejection := range candidate.Rejections {
				codes[candidate.WorkerID][rejection.Code] = struct{}{}
			}
		}
	}
	workerIDs := make([]string, 0, len(codes))
	for workerID := range codes {
		if !eligible[workerID] {
			workerIDs = append(workerIDs, workerID)
		}
	}
	slices.Sort(workerIDs)
	result := make([]WorkerRejectionEvidence, len(workerIDs))
	for index, workerID := range workerIDs {
		workerCodes := make([]placement.RejectionCode, 0, len(codes[workerID]))
		for code := range codes[workerID] {
			workerCodes = append(workerCodes, code)
		}
		slices.Sort(workerCodes)
		result[index] = WorkerRejectionEvidence{WorkerID: workerID, Codes: workerCodes}
	}
	return result
}

func transition(
	ctx context.Context,
	client *registry.Client,
	name, outcome string,
) (TransitionEvidence, error) {
	inventory, err := client.Inventory(ctx)
	if err != nil {
		return TransitionEvidence{}, fmt.Errorf("record %s transition: %w", name, err)
	}
	ids := make([]string, len(inventory.Workers))
	for index, worker := range inventory.Workers {
		ids[index] = worker.ID
	}
	return TransitionEvidence{
		Name: name, ObservedAt: inventory.GeneratedAt,
		InventoryRevision: inventory.Revision, WorkerCount: len(ids), WorkerIDs: ids,
		Outcome: outcome,
	}, nil
}

func appendTransition(
	ctx context.Context,
	client *registry.Client,
	result *Result,
	name, outcome string,
) error {
	evidence, err := transition(ctx, client, name, outcome)
	if err != nil {
		return err
	}
	result.Transitions = append(result.Transitions, evidence)
	return nil
}

func planSignature(plan generation.ExecutionPlan) string {
	parts := make([]string, len(plan.Stages))
	for index, stage := range plan.Stages {
		parts[index] = fmt.Sprintf("%s:[%d,%d)", stage.TargetID, stage.LayerStart, stage.LayerEnd)
	}
	return strings.Join(parts, ",")
}

func firstTarget(decision DecisionEvidence) string {
	if decision.SelectedPlan == nil || len(decision.SelectedPlan.Plan.Stages) == 0 {
		return ""
	}
	return decision.SelectedPlan.Plan.Stages[0].TargetID
}

func decisionUsesWorker(decision DecisionEvidence, workerID string) bool {
	if decision.SelectedPlan == nil {
		return false
	}
	for _, stage := range decision.SelectedPlan.Plan.Stages {
		if stage.TargetID == workerID {
			return true
		}
	}
	return false
}

func findSpec(specs []WorkerSpec, id string) (WorkerSpec, bool) {
	for _, spec := range specs {
		if spec.ID == id {
			return spec, true
		}
	}
	return WorkerSpec{}, false
}

func remoteCode(err error) string {
	var remote *registry.RemoteError
	if errors.As(err, &remote) {
		return remote.Code
	}
	return ""
}

func decisionRejects(
	decision DecisionEvidence,
	workerID string,
	code placement.RejectionCode,
) bool {
	for _, worker := range decision.RejectedWorkers {
		if worker.WorkerID == workerID && slices.Contains(worker.Codes, code) {
			return true
		}
	}
	return false
}

func recordsRequiredRejections(decision DecisionEvidence) bool {
	recorded := make(map[placement.RejectionCode]bool)
	for _, worker := range decision.RejectedWorkers {
		for _, code := range worker.Codes {
			recorded[code] = true
		}
	}
	for _, required := range []placement.RejectionCode{
		placement.RejectionUnhealthy,
		placement.RejectionUnsupportedAdapter,
		placement.RejectionIncompatibleCheckpoint,
		placement.RejectionUnsupportedTransport,
		placement.RejectionInsufficientMemory,
	} {
		if !recorded[required] {
			return false
		}
	}
	return true
}

func decisionsWithinSearchBound(decisions []DecisionEvidence, limit uint64) bool {
	for _, decision := range decisions {
		if decision.SearchOperationCount > limit {
			return false
		}
	}
	return true
}

func decisionsWithinTimeBound(decisions []DecisionEvidence, limit time.Duration) bool {
	for _, decision := range decisions {
		if time.Duration(decision.DecisionMicros)*time.Microsecond > limit {
			return false
		}
	}
	return true
}

func allChecksPassed(checks Checks) bool {
	return checks.ThirtyTwoWorkersVisible &&
		checks.ConcurrentJoinSucceeded &&
		checks.DuplicateIdentityRejected &&
		checks.ExpiryRemovedWorker &&
		checks.RejoinSucceeded &&
		checks.IncompatibleChangeRejected &&
		checks.UnsuitableWorkersPlanStable &&
		checks.BetterWorkerChangedPlan &&
		checks.RemovedWorkerNotReused &&
		checks.SlowWorkerReplanned &&
		checks.DeterministicPlacement &&
		checks.RejectionReasonsRecorded &&
		checks.SearchWorkWithinBound &&
		checks.DecisionLatencyWithinBound &&
		checks.NetworkProbesWithinBound
}

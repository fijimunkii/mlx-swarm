// Package mesh binds live placement decisions to the canonical generation
// runtime without moving placement policy or model execution into the control
// plane.
package mesh

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/placement"
	"github.com/fijimunkii/mlx-swarm/internal/registry"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

const (
	SchemaVersion                   = 1
	defaultSnapshotAttempts         = 4
	DefaultPrepareTimeout           = 10 * time.Minute
	DefaultObservationTimeout       = 10 * time.Second
	unreconciledObservationSequence = ^uint64(0)
)

var (
	// ErrNoEligiblePlan reports a clean scheduling decline. The accompanying
	// SequenceSelection retains every candidate rejection reason.
	ErrNoEligiblePlan = errors.New("current mesh has no eligible execution plan")
	// ErrSequenceAlreadyRun prevents a selected plan from being reused for a
	// later sequence after membership or performance evidence may have changed.
	ErrSequenceAlreadyRun = errors.New("scheduled sequence has already run")
	// ErrSequenceCapacityReserved reports that another prepared or running
	// sequence owns the final locally visible slot for a selected shard.
	ErrSequenceCapacityReserved = errors.New("selected shard sequence capacity is reserved")
	// ErrWorkerCapacityReserved reports that another prepared or running
	// sequence owns the final worker-wide request slot for a selected target.
	ErrWorkerCapacityReserved = errors.New("selected worker request capacity is reserved")
	// ErrWorkerMemoryReserved reports that in-flight work or a recently loaded
	// shard consumes memory not yet reflected in the membership snapshot.
	ErrWorkerMemoryReserved = errors.New("selected worker memory is reserved")
	// ErrWorkerObservationUnavailable reports that membership cannot order the
	// selected worker's state against scheduler-bound mutation observations.
	ErrWorkerObservationUnavailable = errors.New("selected worker observation sequence is unavailable")
	// ErrSequenceRunning reports an invalid attempt to cancel a sequence while
	// its single Generate call is still active.
	ErrSequenceRunning = errors.New("scheduled sequence is running")
)

type inventorySource interface {
	Snapshot() registry.Inventory
}

type profileSource interface {
	Snapshot(time.Time) (placement.ProfileSnapshot, error)
}

// TargetBinding pins the process incarnation and endpoint selected for one
// stage. WorkerID remains the stable plan identity; InstanceID prevents the
// evidence from silently describing a restarted process as the old target.
type TargetBinding struct {
	Index      int    `json:"index"`
	WorkerID   string `json:"workerID"`
	InstanceID string `json:"instanceID"`
	Endpoint   string `json:"endpoint"`
}

// SequenceSelection is the complete machine-readable admission decision for
// one sequence boundary.
type SequenceSelection struct {
	SchemaVersion int                              `json:"schemaVersion"`
	Construction  placement.PlanConstructionResult `json:"construction"`
	Targets       []TargetBinding                  `json:"targets"`
}

// TargetResolver turns one frozen membership binding into a worker caller.
// Production uses HTTPResolver; tests and embedded coordinators can provide a
// resolver without coupling placement policy to a transport implementation.
// State responses must carry the worker daemon's monotonically increasing
// WorkerObservationSequence so admission can prove post-mutation ordering.
type TargetResolver interface {
	Resolve(TargetBinding) (workerproc.PersistentCaller, error)
}

type transportValidatingResolver interface {
	ValidateTransport(placement.TransportRequirement) error
}

// TargetResolverFunc adapts a function to TargetResolver.
type TargetResolverFunc func(TargetBinding) (workerproc.PersistentCaller, error)

func (resolve TargetResolverFunc) Resolve(binding TargetBinding) (workerproc.PersistentCaller, error) {
	return resolve(binding)
}

// HTTPResolver binds selected trusted-network worker endpoints to the current
// JSON compatibility transport.
type HTTPResolver struct {
	Client *http.Client
}

// ValidateTransport prevents a scheduler from using the legacy unbound
// protocol, which an older daemon can accept while ignoring identity headers.
func (HTTPResolver) ValidateTransport(requirement placement.TransportRequirement) error {
	if requirement.Protocol != workerproc.InstanceBoundHTTPProtocol {
		return fmt.Errorf(
			"HTTP mesh scheduler requires transport %q, got %q",
			workerproc.InstanceBoundHTTPProtocol, requirement.Protocol,
		)
	}
	if requirement.TensorEncoding != workerproc.Base64JSONTensorEncoding {
		return fmt.Errorf(
			"HTTP mesh scheduler requires tensor encoding %q, got %q",
			workerproc.Base64JSONTensorEncoding, requirement.TensorEncoding,
		)
	}
	return nil
}

func (resolver HTTPResolver) Resolve(binding TargetBinding) (workerproc.PersistentCaller, error) {
	caller, err := workerproc.NewBoundHTTPPersistentClient(
		binding.Endpoint, resolver.Client, binding.WorkerID, binding.InstanceID,
	)
	if err != nil {
		return nil, fmt.Errorf("worker %q endpoint: %w", binding.WorkerID, err)
	}
	return caller, nil
}

// SequenceScheduler creates one single-use generation session from current
// membership and profile snapshots. Preparing another sequence always takes
// new snapshots and constructs a new plan.
type SequenceScheduler struct {
	mu             sync.Mutex
	inventory      inventorySource
	profiles       profileSource
	resolver       TargetResolver
	prepareTimeout time.Duration
	reservations   map[sequenceReservationKey]sequenceReservationState
	workers        map[workerReservationKey]workerReservationState
}

type sequenceReservationKey struct {
	workerID   string
	instanceID string
	shardID    string
}

type sequenceReservationState struct {
	active        int
	poisonedAfter []uint64
}

type workerReservationKey struct {
	workerID   string
	instanceID string
}

type pendingMemoryReservation struct {
	availableBytes             uint64
	retainedBytes              uint64
	minimumObservationSequence uint64
}

type workerReservationState struct {
	activeRequests       int
	activeAvailableBytes uint64
	activeRetainedBytes  uint64
	pendingMemory        []pendingMemoryReservation
}

type stageReservation struct {
	sequenceKey                sequenceReservationKey
	workerKey                  workerReservationKey
	loadMemoryBytes            uint64
	sequenceMemoryBytes        uint64
	caller                     workerproc.PersistentCaller
	setupObserved              bool
	setupShardPresent          bool
	setupObservationSequence   uint64
	cleanupObserved            bool
	cleanupShardOpen           bool
	cleanupObservationSequence uint64
}

type reservationOutcome struct {
	mayHaveLoaded    bool
	cleanupConfirmed bool
}

// SchedulerOption configures bounded runtime behavior.
type SchedulerOption func(*SequenceScheduler) error

// WithPrepareTimeout bounds selected-worker metadata checks and shard loads.
// The parent context may impose a shorter deadline.
func WithPrepareTimeout(timeout time.Duration) SchedulerOption {
	return func(scheduler *SequenceScheduler) error {
		if timeout < time.Millisecond {
			return errors.New("mesh preparation timeout must be at least 1ms")
		}
		scheduler.prepareTimeout = timeout
		return nil
	}
}

func NewSequenceScheduler(
	inventory inventorySource,
	profiles profileSource,
	resolver TargetResolver,
	options ...SchedulerOption,
) (*SequenceScheduler, error) {
	if inventory == nil || profiles == nil || resolver == nil {
		return nil, errors.New("mesh scheduler requires inventory, profiles, and target resolution")
	}
	scheduler := &SequenceScheduler{
		inventory: inventory, profiles: profiles, resolver: resolver,
		prepareTimeout: DefaultPrepareTimeout,
		reservations:   make(map[sequenceReservationKey]sequenceReservationState),
		workers:        make(map[workerReservationKey]workerReservationState),
	}
	for _, option := range options {
		if option == nil {
			continue
		}
		if err := option(scheduler); err != nil {
			return nil, err
		}
	}
	return scheduler, nil
}

// Prepare selects and materializes the current plan. A nil session with a
// non-nil selection and ErrNoEligiblePlan is a normal bounded decline.
func (scheduler *SequenceScheduler) Prepare(
	ctx context.Context,
	request placement.PlanConstructionRequest,
	reference workerproc.PersistentCaller,
	config generation.PlannedSessionConfig,
) (*ScheduledSequence, SequenceSelection, error) {
	if err := ctx.Err(); err != nil {
		return nil, SequenceSelection{}, err
	}
	if validator, ok := scheduler.resolver.(transportValidatingResolver); ok {
		if err := validator.ValidateTransport(request.Scoring.Transport); err != nil {
			return nil, SequenceSelection{}, fmt.Errorf("validate scheduler transport: %w", err)
		}
	}
	inventory, profile, err := scheduler.snapshots()
	if err != nil {
		return nil, SequenceSelection{}, err
	}
	construction, err := placement.ConstructPlan(inventory, profile, request)
	selection := SequenceSelection{SchemaVersion: SchemaVersion, Construction: construction}
	if err != nil {
		return nil, selection, fmt.Errorf("construct current mesh plan: %w", err)
	}
	if construction.SelectedPlan == nil {
		return nil, selection, ErrNoEligiblePlan
	}

	bindings, err := selectedTargetBindings(inventory, construction.SelectedPlan.Plan)
	if err != nil {
		return nil, selection, err
	}
	selection.Targets = bindings
	reservation, err := scheduler.reserveAdmission(inventory, *construction.SelectedPlan)
	if err != nil {
		return nil, selection, err
	}
	targets := make([]generation.ExecutionTarget, len(bindings))
	for index, binding := range bindings {
		caller, resolveErr := scheduler.resolver.Resolve(binding)
		if resolveErr != nil {
			reservation.finish(reservationOutcome{cleanupConfirmed: true})
			return nil, selection, fmt.Errorf("resolve selected stage %d: %w", index, resolveErr)
		}
		if caller == nil {
			reservation.finish(reservationOutcome{cleanupConfirmed: true})
			return nil, selection, fmt.Errorf("resolve selected stage %d returned no caller", index)
		}
		targets[index] = generation.ExecutionTarget{TargetID: binding.WorkerID, Caller: caller}
	}
	reservation.bindCallers(targets)
	prepareContext, cancelPrepare := context.WithTimeout(ctx, scheduler.prepareTimeout)
	defer cancelPrepare()
	session, err := generation.NewPlannedSession(
		prepareContext, construction.SelectedPlan.Plan, targets, reference, config,
	)
	if err != nil {
		_ = reservation.observeSetup(prepareContext, false)
		reservation.finish(reservationOutcome{mayHaveLoaded: true, cleanupConfirmed: true})
		return nil, selection, fmt.Errorf("prepare selected mesh plan: %w", err)
	}
	if err := reservation.observeSetup(prepareContext, true); err != nil {
		reservation.finish(reservationOutcome{mayHaveLoaded: true, cleanupConfirmed: true})
		return nil, selection, fmt.Errorf("observe prepared mesh plan: %w", err)
	}
	return &ScheduledSequence{session: session, reservation: reservation}, selection, nil
}

func (scheduler *SequenceScheduler) reserveAdmission(
	inventory registry.Inventory,
	evaluation placement.PlanEvaluation,
) (*sequenceReservation, error) {
	workers := make(map[string]registry.Worker, len(inventory.Workers))
	for _, worker := range inventory.Workers {
		workers[worker.ID] = worker
	}
	if len(evaluation.Plan.Stages) != len(evaluation.Stages) ||
		len(evaluation.Plan.Stages) != len(evaluation.Request.Stages) {
		return nil, errors.New("selected plan reservation evidence is incomplete")
	}
	stages := make([]stageReservation, len(evaluation.Plan.Stages))
	scheduler.mu.Lock()
	defer scheduler.mu.Unlock()
	scheduler.reconcileReservationsLocked(workers)
	for index, stage := range evaluation.Plan.Stages {
		worker, found := workers[stage.TargetID]
		if !found {
			return nil, fmt.Errorf("reserve target %q is not in the planning snapshot", stage.TargetID)
		}
		if worker.Status.WorkerObservationSequence == 0 {
			return nil, fmt.Errorf(
				"%w: worker %q instance %q",
				ErrWorkerObservationUnavailable, worker.ID, worker.InstanceID,
			)
		}
		sequenceKey := sequenceReservationKey{
			workerID: worker.ID, instanceID: worker.InstanceID, shardID: stage.ShardID,
		}
		workerKey := workerReservationKey{workerID: worker.ID, instanceID: worker.InstanceID}
		reportedOpen := 0
		for _, shard := range worker.Status.RetainedShards {
			if shard.ID == stage.ShardID {
				reportedOpen = shard.OpenSequenceCount
				break
			}
		}
		limit := worker.Capabilities.Admission.MaxOpenSequencesPerShard
		sequenceState := scheduler.reservations[sequenceKey]
		localOpen := sequenceState.active + len(sequenceState.poisonedAfter)
		if reportedOpen+localOpen >= limit {
			return nil, fmt.Errorf(
				"%w: worker %q instance %q shard %q has %d reported and %d local opens; limit is %d",
				ErrSequenceCapacityReserved, worker.ID, worker.InstanceID, stage.ShardID,
				reportedOpen, localOpen, limit,
			)
		}

		workerState := scheduler.workers[workerKey]
		requestLimit := worker.Capabilities.Admission.MaxConcurrentRequests
		if workerState.activeRequests >= requestLimit {
			return nil, fmt.Errorf(
				"%w: worker %q instance %q has %d local requests; limit is %d",
				ErrWorkerCapacityReserved, worker.ID, worker.InstanceID,
				workerState.activeRequests, requestLimit,
			)
		}
		candidate := evaluation.Stages[index].SelectedCandidate
		if candidate.WorkerID != worker.ID {
			return nil, fmt.Errorf(
				"selected stage %d memory evidence is for worker %q, want %q",
				index, candidate.WorkerID, worker.ID,
			)
		}
		cost := evaluation.Request.Stages[index]
		loadBytes := uint64(0)
		if !candidate.ReusesRetainedShard {
			loadBytes = cost.LoadMemoryBytes
		}
		requiredBytes, overflow := addReservationBytes(loadBytes, cost.SequenceMemoryBytes)
		if overflow || requiredBytes != candidate.RequiredAdditionalMemoryBytes {
			return nil, fmt.Errorf("selected stage %d has inconsistent memory evidence", index)
		}
		reservedAvailable, reservedRetained, overflow := reservedWorkerMemory(workerState)
		if overflow || reservedAvailable > worker.Status.AvailableMemoryBytes ||
			requiredBytes > subtractAvailable(worker.Status.AvailableMemoryBytes, reservedAvailable) {
			return nil, fmt.Errorf(
				"%w: worker %q instance %q requires %d bytes with %d locally reserved; %d reported available",
				ErrWorkerMemoryReserved, worker.ID, worker.InstanceID, requiredBytes,
				reservedAvailable, worker.Status.AvailableMemoryBytes,
			)
		}
		projectedRetained, overflow := addReservationBytes(worker.Status.RetainedBytes, reservedRetained)
		if !overflow {
			projectedRetained, overflow = addReservationBytes(projectedRetained, cost.SequenceMemoryBytes)
		}
		if overflow || projectedRetained > worker.Capabilities.Admission.RetainedByteBudget {
			return nil, fmt.Errorf(
				"%w: worker %q instance %q projects %d retained bytes; budget is %d",
				ErrWorkerMemoryReserved, worker.ID, worker.InstanceID, projectedRetained,
				worker.Capabilities.Admission.RetainedByteBudget,
			)
		}
		stages[index] = stageReservation{
			sequenceKey: sequenceKey, workerKey: workerKey,
			loadMemoryBytes: loadBytes, sequenceMemoryBytes: cost.SequenceMemoryBytes,
		}
	}
	for _, stage := range stages {
		sequenceState := scheduler.reservations[stage.sequenceKey]
		sequenceState.active++
		scheduler.reservations[stage.sequenceKey] = sequenceState
		workerState := scheduler.workers[stage.workerKey]
		workerState.activeRequests++
		workerState.activeAvailableBytes, _ = addReservationBytes(
			workerState.activeAvailableBytes,
			stage.loadMemoryBytes+stage.sequenceMemoryBytes,
		)
		workerState.activeRetainedBytes, _ = addReservationBytes(
			workerState.activeRetainedBytes, stage.sequenceMemoryBytes,
		)
		scheduler.workers[stage.workerKey] = workerState
	}
	return &sequenceReservation{scheduler: scheduler, stages: stages}, nil
}

func (scheduler *SequenceScheduler) reconcileReservationsLocked(
	workers map[string]registry.Worker,
) {
	for key, state := range scheduler.reservations {
		if len(state.poisonedAfter) == 0 {
			continue
		}
		worker, found := workers[key.workerID]
		if !found || worker.InstanceID != key.instanceID {
			state.poisonedAfter = nil
		} else {
			open := 0
			for _, shard := range worker.Status.RetainedShards {
				if shard.ID == key.shardID {
					open = shard.OpenSequenceCount
					break
				}
			}
			if open == 0 {
				retained := state.poisonedAfter[:0]
				for _, minimumObservation := range state.poisonedAfter {
					if worker.Status.WorkerObservationSequence < minimumObservation {
						retained = append(retained, minimumObservation)
					}
				}
				state.poisonedAfter = retained
			}
		}
		if state.active == 0 && len(state.poisonedAfter) == 0 {
			delete(scheduler.reservations, key)
			continue
		}
		scheduler.reservations[key] = state
	}
	for key, state := range scheduler.workers {
		worker, found := workers[key.workerID]
		if !found || worker.InstanceID != key.instanceID {
			state.pendingMemory = nil
		} else if len(state.pendingMemory) > 0 {
			retained := state.pendingMemory[:0]
			for _, pending := range state.pendingMemory {
				if worker.Status.WorkerObservationSequence < pending.minimumObservationSequence {
					retained = append(retained, pending)
				}
			}
			state.pendingMemory = retained
		}
		if state.activeRequests == 0 && state.activeAvailableBytes == 0 &&
			state.activeRetainedBytes == 0 && len(state.pendingMemory) == 0 {
			delete(scheduler.workers, key)
			continue
		}
		scheduler.workers[key] = state
	}
}

type sequenceReservation struct {
	scheduler *SequenceScheduler
	stages    []stageReservation
	once      sync.Once
}

func (reservation *sequenceReservation) bindCallers(targets []generation.ExecutionTarget) {
	for index := range reservation.stages {
		if index < len(targets) {
			reservation.stages[index].caller = targets[index].Caller
		}
	}
}

func (reservation *sequenceReservation) observeSetup(
	ctx context.Context,
	requireShards bool,
) error {
	var observationErr error
	for index := range reservation.stages {
		stage := &reservation.stages[index]
		observation, err := workerproc.ObserveState(ctx, stage.caller)
		if err != nil {
			observationErr = errors.Join(
				observationErr,
				fmt.Errorf("stage %d worker state: %w", index, err),
			)
			continue
		}
		if observation.ObservationSequence == 0 {
			observationErr = errors.Join(
				observationErr,
				fmt.Errorf("stage %d worker state has no observation sequence", index),
			)
			continue
		}
		stage.setupObserved = true
		stage.setupObservationSequence = observation.ObservationSequence
		stage.setupShardPresent = stateHasShard(observation.State, stage.sequenceKey.shardID)
		if requireShards && !stage.setupShardPresent {
			observationErr = errors.Join(
				observationErr,
				fmt.Errorf("stage %d shard %q is absent after setup", index, stage.sequenceKey.shardID),
			)
		}
	}
	return observationErr
}

func (reservation *sequenceReservation) observeCleanup(ctx context.Context) {
	for index := range reservation.stages {
		stage := &reservation.stages[index]
		observation, err := workerproc.ObserveState(ctx, stage.caller)
		if err != nil {
			continue
		}
		if observation.ObservationSequence == 0 {
			continue
		}
		stage.cleanupObserved = true
		stage.cleanupObservationSequence = observation.ObservationSequence
		stage.cleanupShardOpen = stateShardOpen(
			observation.State, stage.sequenceKey.shardID,
		)
	}
}

func stateHasShard(state *workerproc.PersistentWorkerState, shardID string) bool {
	for _, shard := range state.LoadedShards {
		if shard.ShardID == shardID {
			return true
		}
	}
	return false
}

func stateShardOpen(state *workerproc.PersistentWorkerState, shardID string) bool {
	for _, shard := range state.LoadedShards {
		if shard.ShardID == shardID {
			return shard.OpenSequenceCount > 0
		}
	}
	return false
}

func (reservation *sequenceReservation) finish(outcome reservationOutcome) {
	reservation.once.Do(func() {
		if !outcome.cleanupConfirmed {
			ctx, cancel := context.WithTimeout(context.Background(), DefaultObservationTimeout)
			reservation.observeCleanup(ctx)
			cancel()
		}
		reservation.scheduler.finishReservation(reservation.stages, outcome)
	})
}

func (scheduler *SequenceScheduler) finishReservation(
	stages []stageReservation,
	outcome reservationOutcome,
) {
	inventory := scheduler.inventory.Snapshot()
	workers := make(map[string]registry.Worker, len(inventory.Workers))
	for _, worker := range inventory.Workers {
		workers[worker.ID] = worker
	}
	scheduler.mu.Lock()
	defer scheduler.mu.Unlock()
	for _, stage := range stages {
		stageCleanupConfirmed := outcome.cleanupConfirmed ||
			(stage.cleanupObserved && !stage.cleanupShardOpen)
		sequenceState := scheduler.reservations[stage.sequenceKey]
		if sequenceState.active > 0 {
			sequenceState.active--
		}
		worker, found := workers[stage.workerKey.workerID]
		if !stageCleanupConfirmed && found && worker.InstanceID == stage.workerKey.instanceID {
			minimumObservation := unreconciledObservationSequence
			if stage.cleanupObserved {
				minimumObservation = observationAfter(stage.cleanupObservationSequence)
			}
			sequenceState.poisonedAfter = append(
				sequenceState.poisonedAfter, minimumObservation,
			)
		}
		if sequenceState.active == 0 && len(sequenceState.poisonedAfter) == 0 {
			delete(scheduler.reservations, stage.sequenceKey)
		} else {
			scheduler.reservations[stage.sequenceKey] = sequenceState
		}

		workerState := scheduler.workers[stage.workerKey]
		if workerState.activeRequests > 0 {
			workerState.activeRequests--
		}
		activeBytes := stage.loadMemoryBytes + stage.sequenceMemoryBytes
		workerState.activeAvailableBytes = subtractAvailable(
			workerState.activeAvailableBytes, activeBytes,
		)
		workerState.activeRetainedBytes = subtractAvailable(
			workerState.activeRetainedBytes, stage.sequenceMemoryBytes,
		)
		pendingAvailable := uint64(0)
		pendingRetained := uint64(0)
		minimumObservation := uint64(0)
		if outcome.mayHaveLoaded && (!stage.setupObserved || stage.setupShardPresent) {
			pendingAvailable = stage.loadMemoryBytes
			if stage.setupObserved {
				minimumObservation = observationAfter(stage.setupObservationSequence)
			} else {
				minimumObservation = unreconciledObservationSequence
			}
		}
		if !stageCleanupConfirmed {
			pendingAvailable, _ = addReservationBytes(
				pendingAvailable, stage.sequenceMemoryBytes,
			)
			pendingRetained = stage.sequenceMemoryBytes
			cleanupMinimum := unreconciledObservationSequence
			if stage.cleanupObserved {
				cleanupMinimum = observationAfter(stage.cleanupObservationSequence)
			}
			if cleanupMinimum > minimumObservation {
				minimumObservation = cleanupMinimum
			}
		}
		if pendingAvailable > 0 && found && worker.InstanceID == stage.workerKey.instanceID {
			workerState.pendingMemory = append(workerState.pendingMemory, pendingMemoryReservation{
				availableBytes:             pendingAvailable,
				retainedBytes:              pendingRetained,
				minimumObservationSequence: minimumObservation,
			})
		}
		if workerState.activeRequests == 0 && workerState.activeAvailableBytes == 0 &&
			workerState.activeRetainedBytes == 0 && len(workerState.pendingMemory) == 0 {
			delete(scheduler.workers, stage.workerKey)
		} else {
			scheduler.workers[stage.workerKey] = workerState
		}
	}
}

func reservedWorkerMemory(state workerReservationState) (uint64, uint64, bool) {
	available := state.activeAvailableBytes
	retained := state.activeRetainedBytes
	for _, pending := range state.pendingMemory {
		var overflow bool
		available, overflow = addReservationBytes(available, pending.availableBytes)
		if overflow {
			return 0, 0, true
		}
		retained, overflow = addReservationBytes(retained, pending.retainedBytes)
		if overflow {
			return 0, 0, true
		}
	}
	return available, retained, false
}

func addReservationBytes(left, right uint64) (uint64, bool) {
	if left > ^uint64(0)-right {
		return ^uint64(0), true
	}
	return left + right, false
}

func subtractAvailable(available, reserved uint64) uint64 {
	if reserved >= available {
		return 0
	}
	return available - reserved
}

func observationAfter(sequence uint64) uint64 {
	if sequence == 0 || sequence == unreconciledObservationSequence {
		return unreconciledObservationSequence
	}
	return sequence + 1
}

func (scheduler *SequenceScheduler) snapshots() (
	registry.Inventory,
	placement.ProfileSnapshot,
	error,
) {
	var inventory registry.Inventory
	var profile placement.ProfileSnapshot
	var err error
	for attempt := 0; attempt < defaultSnapshotAttempts; attempt++ {
		inventory = scheduler.inventory.Snapshot()
		profile, err = scheduler.profiles.Snapshot(inventory.GeneratedAt)
		if err == nil {
			return inventory, profile, nil
		}
		if !errors.Is(err, placement.ErrProfileSnapshotRewind) {
			return registry.Inventory{}, placement.ProfileSnapshot{},
				fmt.Errorf("snapshot mesh profile: %w", err)
		}
	}
	return registry.Inventory{}, placement.ProfileSnapshot{}, fmt.Errorf(
		"snapshot mesh profile after %d attempts: %w",
		defaultSnapshotAttempts, err,
	)
}

func selectedTargetBindings(
	inventory registry.Inventory,
	plan generation.ExecutionPlan,
) ([]TargetBinding, error) {
	if plan.InventoryRevision != strconv.FormatUint(inventory.Revision, 10) {
		return nil, fmt.Errorf(
			"selected plan inventory revision is %q; current snapshot is %d",
			plan.InventoryRevision, inventory.Revision,
		)
	}
	workers := make(map[string]registry.Worker, len(inventory.Workers))
	for _, worker := range inventory.Workers {
		workers[worker.ID] = worker
	}
	bindings := make([]TargetBinding, len(plan.Stages))
	for index, stage := range plan.Stages {
		worker, found := workers[stage.TargetID]
		if !found {
			return nil, fmt.Errorf("selected target %q is not in the planning snapshot", stage.TargetID)
		}
		if worker.InstanceID == "" || worker.Endpoint == "" {
			return nil, fmt.Errorf("selected target %q has incomplete binding", stage.TargetID)
		}
		bindings[index] = TargetBinding{
			Index: index, WorkerID: worker.ID,
			InstanceID: worker.InstanceID, Endpoint: worker.Endpoint,
		}
	}
	return bindings, nil
}

// ScheduledSequence owns exactly one Generate attempt. Its selected plan is
// never consulted against newer inventory mid-sequence; a later request must
// be admitted through SequenceScheduler.Prepare again.
type ScheduledSequence struct {
	mu          sync.Mutex
	session     *generation.PlannedSession
	reservation *sequenceReservation
	state       scheduledSequenceState
}

type scheduledSequenceState uint8

const (
	sequencePrepared scheduledSequenceState = iota
	sequenceRunning
	sequenceFinished
	sequenceClosed
)

func (sequence *ScheduledSequence) Info() generation.PlannedSessionInfo {
	return sequence.session.Info()
}

func (sequence *ScheduledSequence) Generate(
	ctx context.Context,
	request generation.Request,
) (generation.PlannedResult, error) {
	sequence.mu.Lock()
	if sequence.state != sequencePrepared {
		sequence.mu.Unlock()
		return generation.PlannedResult{}, ErrSequenceAlreadyRun
	}
	sequence.state = sequenceRunning
	sequence.mu.Unlock()
	result, err := sequence.session.Generate(ctx, request)
	sequence.finish(result.SequenceCleanupConfirmed)
	return result, err
}

// Close releases admission reserved by Prepare when the caller decides not to
// run the sequence. Generate releases it automatically after success or
// failure. Close must not race an active Generate call.
func (sequence *ScheduledSequence) Close() error {
	sequence.mu.Lock()
	switch sequence.state {
	case sequenceRunning:
		sequence.mu.Unlock()
		return ErrSequenceRunning
	case sequencePrepared:
		sequence.state = sequenceClosed
		reservation := sequence.reservation
		sequence.reservation = nil
		sequence.mu.Unlock()
		reservation.finish(reservationOutcome{mayHaveLoaded: true, cleanupConfirmed: true})
		return nil
	case sequenceFinished, sequenceClosed:
		sequence.mu.Unlock()
		return nil
	default:
		sequence.mu.Unlock()
		return errors.New("scheduled sequence has invalid state")
	}
}

func (sequence *ScheduledSequence) finish(cleanupConfirmed bool) {
	sequence.mu.Lock()
	sequence.state = sequenceFinished
	reservation := sequence.reservation
	sequence.reservation = nil
	sequence.mu.Unlock()
	reservation.finish(reservationOutcome{
		mayHaveLoaded: true, cleanupConfirmed: cleanupConfirmed,
	})
}

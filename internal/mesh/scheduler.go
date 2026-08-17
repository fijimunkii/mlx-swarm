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
	SchemaVersion           = 1
	defaultSnapshotAttempts = 4
	DefaultPrepareTimeout   = 10 * time.Minute
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
type TargetResolver interface {
	Resolve(TargetBinding) (workerproc.PersistentCaller, error)
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
}

type sequenceReservationKey struct {
	workerID   string
	instanceID string
	shardID    string
}

type sequenceReservationState struct {
	active        int
	poisonedAfter []time.Time
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
	reservation, err := scheduler.reserveSequenceCapacity(inventory, construction.SelectedPlan.Plan)
	if err != nil {
		return nil, selection, err
	}
	targets := make([]generation.ExecutionTarget, len(bindings))
	for index, binding := range bindings {
		caller, resolveErr := scheduler.resolver.Resolve(binding)
		if resolveErr != nil {
			reservation.finish(true)
			return nil, selection, fmt.Errorf("resolve selected stage %d: %w", index, resolveErr)
		}
		if caller == nil {
			reservation.finish(true)
			return nil, selection, fmt.Errorf("resolve selected stage %d returned no caller", index)
		}
		targets[index] = generation.ExecutionTarget{TargetID: binding.WorkerID, Caller: caller}
	}
	prepareContext, cancelPrepare := context.WithTimeout(ctx, scheduler.prepareTimeout)
	defer cancelPrepare()
	session, err := generation.NewPlannedSession(
		prepareContext, construction.SelectedPlan.Plan, targets, reference, config,
	)
	if err != nil {
		reservation.finish(true)
		return nil, selection, fmt.Errorf("prepare selected mesh plan: %w", err)
	}
	return &ScheduledSequence{session: session, reservation: reservation}, selection, nil
}

func (scheduler *SequenceScheduler) reserveSequenceCapacity(
	inventory registry.Inventory,
	plan generation.ExecutionPlan,
) (*sequenceReservation, error) {
	workers := make(map[string]registry.Worker, len(inventory.Workers))
	for _, worker := range inventory.Workers {
		workers[worker.ID] = worker
	}
	keys := make([]sequenceReservationKey, len(plan.Stages))
	scheduler.mu.Lock()
	defer scheduler.mu.Unlock()
	scheduler.reconcilePoisonedLocked(workers)
	for index, stage := range plan.Stages {
		worker, found := workers[stage.TargetID]
		if !found {
			return nil, fmt.Errorf("reserve target %q is not in the planning snapshot", stage.TargetID)
		}
		key := sequenceReservationKey{
			workerID: worker.ID, instanceID: worker.InstanceID, shardID: stage.ShardID,
		}
		reportedOpen := 0
		for _, shard := range worker.Status.RetainedShards {
			if shard.ID == stage.ShardID {
				reportedOpen = shard.OpenSequenceCount
				break
			}
		}
		limit := worker.Capabilities.Admission.MaxOpenSequencesPerShard
		state := scheduler.reservations[key]
		localOpen := state.active + len(state.poisonedAfter)
		if reportedOpen+localOpen >= limit {
			return nil, fmt.Errorf(
				"%w: worker %q instance %q shard %q has %d reported and %d local opens; limit is %d",
				ErrSequenceCapacityReserved, worker.ID, worker.InstanceID, stage.ShardID,
				reportedOpen, localOpen, limit,
			)
		}
		keys[index] = key
	}
	for _, key := range keys {
		state := scheduler.reservations[key]
		state.active++
		scheduler.reservations[key] = state
	}
	return &sequenceReservation{scheduler: scheduler, keys: keys}, nil
}

func (scheduler *SequenceScheduler) reconcilePoisonedLocked(
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
				for _, baseline := range state.poisonedAfter {
					if !worker.StatusObservedAt.After(baseline) {
						retained = append(retained, baseline)
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
}

type sequenceReservation struct {
	scheduler *SequenceScheduler
	keys      []sequenceReservationKey
	once      sync.Once
}

func (reservation *sequenceReservation) finish(cleanupConfirmed bool) {
	reservation.once.Do(func() {
		if cleanupConfirmed {
			reservation.scheduler.releaseReservation(reservation.keys)
			return
		}
		reservation.scheduler.poisonReservation(reservation.keys)
	})
}

func (scheduler *SequenceScheduler) releaseReservation(keys []sequenceReservationKey) {
	scheduler.mu.Lock()
	defer scheduler.mu.Unlock()
	for _, key := range keys {
		state := scheduler.reservations[key]
		if state.active > 0 {
			state.active--
		}
		if state.active == 0 && len(state.poisonedAfter) == 0 {
			delete(scheduler.reservations, key)
			continue
		}
		scheduler.reservations[key] = state
	}
}

func (scheduler *SequenceScheduler) poisonReservation(keys []sequenceReservationKey) {
	inventory := scheduler.inventory.Snapshot()
	workers := make(map[string]registry.Worker, len(inventory.Workers))
	for _, worker := range inventory.Workers {
		workers[worker.ID] = worker
	}
	scheduler.mu.Lock()
	defer scheduler.mu.Unlock()
	for _, key := range keys {
		state := scheduler.reservations[key]
		if state.active > 0 {
			state.active--
		}
		worker, found := workers[key.workerID]
		if found && worker.InstanceID == key.instanceID {
			state.poisonedAfter = append(state.poisonedAfter, worker.StatusObservedAt)
		}
		if state.active == 0 && len(state.poisonedAfter) == 0 {
			delete(scheduler.reservations, key)
			continue
		}
		scheduler.reservations[key] = state
	}
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
		reservation.finish(true)
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
	reservation.finish(cleanupConfirmed)
}

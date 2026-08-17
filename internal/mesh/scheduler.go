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
)

var (
	// ErrNoEligiblePlan reports a clean scheduling decline. The accompanying
	// SequenceSelection retains every candidate rejection reason.
	ErrNoEligiblePlan = errors.New("current mesh has no eligible execution plan")
	// ErrSequenceAlreadyRun prevents a selected plan from being reused for a
	// later sequence after membership or performance evidence may have changed.
	ErrSequenceAlreadyRun = errors.New("scheduled sequence has already run")
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
	caller, err := workerproc.NewHTTPPersistentClient(binding.Endpoint, resolver.Client)
	if err != nil {
		return nil, fmt.Errorf("worker %q endpoint: %w", binding.WorkerID, err)
	}
	return caller, nil
}

// SequenceScheduler creates one single-use generation session from current
// membership and profile snapshots. Preparing another sequence always takes
// new snapshots and constructs a new plan.
type SequenceScheduler struct {
	inventory inventorySource
	profiles  profileSource
	resolver  TargetResolver
}

func NewSequenceScheduler(
	inventory inventorySource,
	profiles profileSource,
	resolver TargetResolver,
) (*SequenceScheduler, error) {
	if inventory == nil || profiles == nil || resolver == nil {
		return nil, errors.New("mesh scheduler requires inventory, profiles, and target resolution")
	}
	return &SequenceScheduler{inventory: inventory, profiles: profiles, resolver: resolver}, nil
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
	targets := make([]generation.ExecutionTarget, len(bindings))
	for index, binding := range bindings {
		caller, resolveErr := scheduler.resolver.Resolve(binding)
		if resolveErr != nil {
			return nil, selection, fmt.Errorf("resolve selected stage %d: %w", index, resolveErr)
		}
		if caller == nil {
			return nil, selection, fmt.Errorf("resolve selected stage %d returned no caller", index)
		}
		targets[index] = generation.ExecutionTarget{TargetID: binding.WorkerID, Caller: caller}
	}
	session, err := generation.NewPlannedSession(
		ctx, construction.SelectedPlan.Plan, targets, reference, config,
	)
	if err != nil {
		return nil, selection, fmt.Errorf("prepare selected mesh plan: %w", err)
	}
	return &ScheduledSequence{session: session}, selection, nil
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
	mu      sync.Mutex
	session *generation.PlannedSession
	run     bool
}

func (sequence *ScheduledSequence) Info() generation.PlannedSessionInfo {
	return sequence.session.Info()
}

func (sequence *ScheduledSequence) Generate(
	ctx context.Context,
	request generation.Request,
) (generation.PlannedResult, error) {
	sequence.mu.Lock()
	if sequence.run {
		sequence.mu.Unlock()
		return generation.PlannedResult{}, ErrSequenceAlreadyRun
	}
	sequence.run = true
	sequence.mu.Unlock()
	return sequence.session.Generate(ctx, request)
}

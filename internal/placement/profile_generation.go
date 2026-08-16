package placement

import (
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/registry"
)

// ObservePlannedSample ingests one complete successful N-stage traversal as an
// atomic compute-profile update. Transport observations remain explicit
// because StageExecution overhead mixes HTTP, queueing, and serialization and
// is not an honest standalone bandwidth measurement.
func (store *ProfileStore) ObservePlannedSample(
	at time.Time,
	inventory registry.Inventory,
	plan generation.ExecutionPlan,
	sample generation.PlannedStageSample,
) error {
	if at.IsZero() {
		return errors.New("profile observation time is required")
	}
	if inventory.SchemaVersion != registry.SchemaVersion {
		return fmt.Errorf(
			"inventory schema version is %d, want %d",
			inventory.SchemaVersion, registry.SchemaVersion,
		)
	}
	if err := generation.ValidateExecutionPlan(plan); err != nil {
		return fmt.Errorf("profile execution plan: %w", err)
	}
	if plan.InventoryRevision != "" &&
		plan.InventoryRevision != strconv.FormatUint(inventory.Revision, 10) {
		return fmt.Errorf(
			"profile plan inventory revision is %q; current inventory revision is %d",
			plan.InventoryRevision, inventory.Revision,
		)
	}
	if sample.Operation == "" || sample.InputTokenCount <= 0 {
		return errors.New("profile planned sample requires an operation and input tokens")
	}
	if len(sample.Stages) != len(plan.Stages) {
		return fmt.Errorf(
			"profile planned sample has %d stages; want %d",
			len(sample.Stages), len(plan.Stages),
		)
	}
	workers := make(map[string]registry.Worker, len(inventory.Workers))
	for _, worker := range inventory.Workers {
		if _, exists := workers[worker.ID]; exists {
			return fmt.Errorf("profile inventory worker %q is duplicated", worker.ID)
		}
		workers[worker.ID] = worker
	}
	observations := make([]ComputeObservation, len(sample.Stages))
	for index, execution := range sample.Stages {
		stage := plan.Stages[index]
		if execution.Index != index || execution.Stage != stage ||
			execution.Operation != sample.Operation || execution.ComputeMicros == 0 {
			return fmt.Errorf("profile planned sample stage %d is inconsistent", index)
		}
		worker, exists := workers[stage.TargetID]
		if !exists {
			return fmt.Errorf("profile plan target %q is not in inventory", stage.TargetID)
		}
		observations[index] = ComputeObservation{
			WorkerID: stage.TargetID, WorkerInstanceID: worker.InstanceID,
			Backend: worker.Capabilities.Backend,
			Model:   plan.Model, Operation: sample.Operation,
			LayerStart: stage.LayerStart, LayerEnd: stage.LayerEnd,
			InputTokenCount: uint64(sample.InputTokenCount),
			ComputeMicros:   execution.ComputeMicros, ObservedAt: at,
		}
	}
	return store.observeComputeBatch(observations)
}

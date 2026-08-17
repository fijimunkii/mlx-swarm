package meshstress

import (
	"context"
	"testing"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/registry"
)

func TestRunProvesSyntheticScaleAndChurn(t *testing.T) {
	result, err := Run(context.Background(), DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if !result.Checks.AllPassed {
		t.Fatalf("checks = %+v", result.Checks)
	}
	if result.Bounds.WorkerCount != DefaultWorkerCount || result.Bounds.NetworkProbeCount != 0 {
		t.Fatalf("bounds = %+v", result.Bounds)
	}
	if len(result.Decisions) < 7 || len(result.Transitions) < 8 {
		t.Fatalf("evidence is incomplete: %d decisions, %d transitions", len(result.Decisions), len(result.Transitions))
	}
}

func TestWorkerSpecMapsConfigurableRegistryState(t *testing.T) {
	spec := baseWorkerSpec("synthetic-test")
	spec.Backend = "cuda"
	spec.AvailableMemoryBytes = 1234
	spec.Health = registry.HealthDegraded
	spec.RecentFailureCount = 7
	spec.RetainedBytes = 20
	spec.RetainedShards = []registry.RetainedShard{{
		ID: "resident", ModelID: proofModel.ID, CheckpointFingerprint: proofCheckpoint,
		LayerStart: 0, LayerEnd: 6, OwnsInput: true, OpenSequenceCount: 1,
	}}
	registration := spec.Registration()
	if registration.Capabilities.Backend != "cuda" ||
		registration.Status.AvailableMemoryBytes != 1234 ||
		registration.Status.Health != registry.HealthDegraded ||
		registration.Status.RecentFailureCount != 7 ||
		registration.Status.OpenSequenceCount != 1 ||
		registration.Status.RetainedBytes != 20 {
		t.Fatalf("registration = %+v", registration)
	}
}

func TestRunRejectsAnUnboundedSmallScenario(t *testing.T) {
	config := DefaultConfig()
	config.WorkerCount = DefaultWorkerCount - 1
	if _, err := Run(context.Background(), config); err == nil {
		t.Fatal("expected worker-count rejection")
	}
	config = DefaultConfig()
	config.MaxDecisionDuration = time.Microsecond
	if _, err := Run(context.Background(), config); err == nil {
		t.Fatal("expected decision-bound rejection")
	}
}

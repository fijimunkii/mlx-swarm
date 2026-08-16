package main

import (
	"context"
	"encoding/json"
	"errors"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/registry"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

type fixtureCapabilityRunner struct {
	capabilities localCapabilities
}

func (runner fixtureCapabilityRunner) Run(
	context.Context,
	[]string,
	[]byte,
) (workerproc.Result, error) {
	encoded, err := json.Marshal(runner.capabilities)
	return workerproc.Result{Output: encoded}, err
}

type fixtureMembershipWorker struct {
	state    workerproc.PersistentWorkerState
	restarts int
	err      error
}

func (worker *fixtureMembershipWorker) Call(
	context.Context,
	workerproc.PersistentRequest,
) (workerproc.PersistentResponse, error) {
	if worker.err != nil {
		return workerproc.PersistentResponse{}, worker.err
	}
	state := worker.state
	return workerproc.PersistentResponse{
		OK: true, Result: &workerproc.PersistentWorkerResult{State: &state},
	}, nil
}

func (worker *fixtureMembershipWorker) RestartCount() int { return worker.restarts }

type blockingMembershipWorker struct {
	state   workerproc.PersistentWorkerState
	calls   atomic.Int32
	blocked chan struct{}
	release chan struct{}
}

func (worker *blockingMembershipWorker) Call(
	ctx context.Context,
	_ workerproc.PersistentRequest,
) (workerproc.PersistentResponse, error) {
	if worker.calls.Add(1) == 1 {
		state := worker.state
		return workerproc.PersistentResponse{
			OK: true, Result: &workerproc.PersistentWorkerResult{State: &state},
		}, nil
	}
	select {
	case worker.blocked <- struct{}{}:
	default:
	}
	select {
	case <-worker.release:
		state := worker.state
		return workerproc.PersistentResponse{
			OK: true, Result: &workerproc.PersistentWorkerResult{State: &state},
		}, nil
	case <-ctx.Done():
		return workerproc.PersistentResponse{}, ctx.Err()
	}
}

func (*blockingMembershipWorker) RestartCount() int { return 0 }

func TestMembershipAgentRegistersRefreshesAndRejoins(t *testing.T) {
	membership := registry.New(time.Minute)
	server := httptest.NewServer(registry.NewHTTPHandler(membership))
	defer server.Close()
	worker := &fixtureMembershipWorker{state: testMembershipState()}
	agent, err := newMembershipAgent(
		context.Background(),
		membershipConfig{
			ControlURL: server.URL, WorkerID: "mac-a", InstanceID: "process-a",
			PublicURL: "http://mac-a:8080", Backend: "mlx", HeartbeatInterval: time.Second,
		},
		fixtureCapabilityRunner{capabilities: localCapabilities{
			Runtime: "mlx-swift", Device: "gpu",
			CheckpointShardModelTypes: []string{"adapter-a", "adapter-b"}, PhysicalMemoryBytes: 8192,
		}},
		worker,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := agent.Register(context.Background()); err != nil {
		t.Fatal(err)
	}
	inventory := membership.Snapshot()
	if len(inventory.Workers) != 1 {
		t.Fatalf("registered workers = %d, want 1", len(inventory.Workers))
	}
	record := inventory.Workers[0]
	if record.ID != "mac-a" || record.Capabilities.Backend != "mlx" ||
		len(record.Capabilities.Adapters) != 2 || record.Capabilities.Admission.MaxOpenSequencesPerShard != 16 ||
		len(record.Status.RetainedShards) != 1 {
		t.Fatalf("incomplete worker record: %+v", record)
	}

	worker.restarts = 2
	worker.state.LoadedShards[0].OpenSequenceCount = 1
	agent.recordStatus(agent.probeStatus(context.Background(), agent.registration.Status))
	if err := agent.heartbeat(context.Background()); err != nil {
		t.Fatal(err)
	}
	refreshed := membership.Snapshot().Workers[0]
	if refreshed.Status.RestartCount != 2 || refreshed.Status.OpenSequenceCount != 1 {
		t.Fatalf("heartbeat did not refresh status: %+v", refreshed.Status)
	}

	agent.recordStatus(agent.registration.Status)
	agent.statusSampledAt = time.Now().Add(-2 * time.Minute)
	if err := membership.Remove("mac-a", "process-a"); err != nil {
		t.Fatal(err)
	}
	if err := agent.heartbeat(context.Background()); err != nil {
		t.Fatalf("agent did not re-register after removal: %v", err)
	}
	if workers := membership.Snapshot().Workers; len(workers) != 1 ||
		workers[0].InstanceID != "process-a" || !workers[0].StatusObservedAt.IsZero() {
		t.Fatalf("unexpected rejoined inventory: %+v", workers)
	}
}

func TestMembershipAgentRefreshesLeaseWhileStateProbeIsBlocked(t *testing.T) {
	membership := registry.New(250 * time.Millisecond)
	server := httptest.NewServer(registry.NewHTTPHandler(membership))
	defer server.Close()
	worker := &blockingMembershipWorker{
		state: testMembershipState(), blocked: make(chan struct{}, 1), release: make(chan struct{}),
	}
	agent, err := newMembershipAgent(
		context.Background(),
		membershipConfig{
			ControlURL: server.URL, WorkerID: "mac-a", InstanceID: "process-a",
			PublicURL: "http://mac-a:8080", Backend: "mlx", HeartbeatInterval: 20 * time.Millisecond,
		},
		fixtureCapabilityRunner{capabilities: localCapabilities{
			Runtime: "mlx-swift", Device: "gpu",
			CheckpointShardModelTypes: []string{"adapter-a"}, PhysicalMemoryBytes: 8192,
		}},
		worker,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := agent.Register(context.Background()); err != nil {
		t.Fatal(err)
	}
	registered := membership.Snapshot()
	registeredAt := registered.Workers[0].LastSeen
	statusObservedAt := registered.Workers[0].StatusObservedAt

	ctx, cancel := context.WithCancel(context.Background())
	runErr := make(chan error, 1)
	go func() { runErr <- agent.Run(ctx) }()
	select {
	case <-worker.blocked:
	case <-time.After(time.Second):
		cancel()
		t.Fatal("worker state probe did not block")
	}
	time.Sleep(500 * time.Millisecond)
	inventory := membership.Snapshot()
	if len(inventory.Workers) != 1 || !inventory.Workers[0].LastSeen.After(registeredAt) ||
		!inventory.Workers[0].StatusObservedAt.Equal(statusObservedAt) {
		cancel()
		t.Fatalf("blocked state probe changed lease or status freshness incorrectly: %+v", inventory)
	}
	close(worker.release)
	deadline := time.Now().Add(time.Second)
	for !time.Now().After(deadline) {
		inventory = membership.Snapshot()
		if len(inventory.Workers) == 1 && inventory.Workers[0].StatusObservedAt.After(statusObservedAt) {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if len(inventory.Workers) != 1 || !inventory.Workers[0].StatusObservedAt.After(statusObservedAt) {
		cancel()
		t.Fatalf("fresh worker state did not advance status observation time: %+v", inventory)
	}

	cancel()
	select {
	case err := <-runErr:
		if err != nil {
			t.Fatalf("membership agent stopped with error: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("membership agent did not stop")
	}
}

func TestMembershipAgentStopsAfterLeaseOwnershipChanges(t *testing.T) {
	membership := registry.New(time.Minute)
	server := httptest.NewServer(registry.NewHTTPHandler(membership))
	defer server.Close()
	worker := &fixtureMembershipWorker{state: testMembershipState()}
	agent, err := newMembershipAgent(
		context.Background(),
		membershipConfig{
			ControlURL: server.URL, WorkerID: "mac-a", InstanceID: "process-a",
			PublicURL: "http://mac-a:8080", Backend: "mlx", HeartbeatInterval: time.Second,
		},
		fixtureCapabilityRunner{capabilities: localCapabilities{
			Runtime: "mlx-swift", Device: "gpu",
			CheckpointShardModelTypes: []string{"adapter-a"}, PhysicalMemoryBytes: 8192,
		}},
		worker,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := agent.Register(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := membership.Remove("mac-a", "process-a"); err != nil {
		t.Fatal(err)
	}
	replacement := agent.registration
	replacement.InstanceID = "process-b"
	if _, err := membership.Register(replacement); err != nil {
		t.Fatal(err)
	}
	var remote *registry.RemoteError
	if err := agent.heartbeat(context.Background()); err == nil ||
		!errors.As(err, &remote) || remote.Code != "stale_instance" {
		t.Fatalf("lost lease heartbeat error = %v", err)
	}
}

func TestMembershipEnvironmentIsOptIn(t *testing.T) {
	t.Setenv("SWARMD_CONTROL_URL", "")
	if _, enabled, err := membershipConfigFromEnvironment(); err != nil || enabled {
		t.Fatalf("disabled config = enabled %t, err %v", enabled, err)
	}
	t.Setenv("SWARMD_CONTROL_URL", "http://control:9090")
	if _, _, err := membershipConfigFromEnvironment(); err == nil {
		t.Fatal("incomplete opt-in config was accepted")
	}
	t.Setenv("SWARMD_WORKER_ID", "mac-a")
	t.Setenv("SWARMD_PUBLIC_URL", "http://mac-a:8080")
	config, enabled, err := membershipConfigFromEnvironment()
	if err != nil || !enabled || len(config.InstanceID) != 32 {
		t.Fatalf("configured membership = %+v, enabled %t, err %v", config, enabled, err)
	}
}

func testMembershipState() workerproc.PersistentWorkerState {
	return workerproc.PersistentWorkerState{
		RetainedByteBudget: 1024, MaxOpenSequencesPerShard: 16, PhysicalMemoryBytes: 8192,
		Memory: workerproc.StageMemory{ProcessPhysicalBytes: 2048},
		LoadedShards: []workerproc.PersistentShardSnapshot{{
			ShardID: "shard-a", ModelID: "model-a", ModelType: "adapter-a",
			CheckpointFingerprint: "sha256:a", LayerStart: 0, LayerEnd: 4,
			OwnsInput: true, LoadedMemory: workerproc.StageMemory{ActiveBytes: 512},
		}},
	}
}

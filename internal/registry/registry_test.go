package registry

import (
	"errors"
	"testing"
	"time"
)

func TestRegistryLifecycleIsLeasedAndDeterministic(t *testing.T) {
	now := time.Date(2026, time.August, 16, 12, 0, 0, 0, time.UTC)
	registry := New(10*time.Second, WithClock(func() time.Time { return now }))

	second := testRegistration("worker-b", "instance-b")
	first := testRegistration("worker-a", "instance-a")
	first.Capabilities.Operations = []string{"prefill", "decode", "decode"}
	mutation, err := registry.Register(second)
	if err != nil {
		t.Fatal(err)
	}
	if mutation.InventoryRevision != 1 {
		t.Fatalf("first revision = %d, want 1", mutation.InventoryRevision)
	}
	if _, err := registry.Register(first); err != nil {
		t.Fatal(err)
	}

	inventory := registry.Snapshot()
	if inventory.Revision != 2 || len(inventory.Workers) != 2 ||
		inventory.Workers[0].ID != "worker-a" || inventory.Workers[1].ID != "worker-b" {
		t.Fatalf("unexpected inventory: %+v", inventory)
	}
	operations := inventory.Workers[0].Capabilities.Operations
	if len(operations) != 2 || operations[0] != "decode" || operations[1] != "prefill" {
		t.Fatalf("operations were not canonicalized: %v", operations)
	}
	if renewed, err := registry.Register(first); err != nil || renewed.InventoryRevision != 2 {
		t.Fatalf("unchanged registration changed revision: mutation=%+v err=%v", renewed, err)
	}

	// Returned records cannot mutate the registry's owned slices.
	inventory.Workers[0].Capabilities.Operations[0] = "corrupted"
	if got := registry.Snapshot().Workers[0].Capabilities.Operations[0]; got != "decode" {
		t.Fatalf("snapshot mutated registry state: %q", got)
	}

	duplicate := testRegistration("worker-a", "new-instance")
	if _, err := registry.Register(duplicate); !errors.Is(err, ErrDuplicateWorker) {
		t.Fatalf("duplicate registration error = %v", err)
	}
	duplicateInstance := testRegistration("worker-c", "instance-a")
	if _, err := registry.Register(duplicateInstance); !errors.Is(err, ErrDuplicateInstance) {
		t.Fatalf("duplicate instance error = %v", err)
	}
	duplicateEndpoint := testRegistration("worker-c", "instance-c")
	duplicateEndpoint.Endpoint = first.Endpoint
	if _, err := registry.Register(duplicateEndpoint); !errors.Is(err, ErrDuplicateEndpoint) {
		t.Fatalf("duplicate endpoint error = %v", err)
	}
	if _, err := registry.Heartbeat("worker-a", Heartbeat{
		SchemaVersion: SchemaVersion, InstanceID: "new-instance", Status: first.Status,
	}); !errors.Is(err, ErrStaleInstance) {
		t.Fatalf("stale heartbeat error = %v", err)
	}

	now = now.Add(4 * time.Second)
	status := first.Status
	status.AvailableMemoryBytes--
	heartbeat, err := registry.Heartbeat("worker-a", Heartbeat{
		SchemaVersion: SchemaVersion, InstanceID: "instance-a", Status: status,
	})
	if err != nil {
		t.Fatal(err)
	}
	if heartbeat.InventoryRevision != 3 || !heartbeat.Worker.LastSeen.Equal(now) ||
		!heartbeat.Worker.ExpiresAt.Equal(now.Add(10*time.Second)) {
		t.Fatalf("unexpected heartbeat mutation: %+v", heartbeat)
	}
	if renewed, err := registry.Heartbeat("worker-a", Heartbeat{
		SchemaVersion: SchemaVersion, InstanceID: "instance-a", Status: status,
	}); err != nil || renewed.InventoryRevision != 3 {
		t.Fatalf("unchanged heartbeat changed revision: mutation=%+v err=%v", renewed, err)
	}

	if err := registry.Remove("worker-a", "new-instance"); !errors.Is(err, ErrStaleInstance) {
		t.Fatalf("stale removal error = %v", err)
	}
	if err := registry.Remove("worker-a", "instance-a"); err != nil {
		t.Fatal(err)
	}
	remaining := registry.Snapshot()
	if remaining.Revision != 4 || len(remaining.Workers) != 1 || remaining.Workers[0].ID != "worker-b" {
		t.Fatalf("unexpected inventory after removal: %+v", remaining)
	}
}

func TestRegistryExpiresAndAllowsRejoin(t *testing.T) {
	now := time.Date(2026, time.August, 16, 12, 0, 0, 0, time.UTC)
	registry := New(5*time.Second, WithClock(func() time.Time { return now }))
	if _, err := registry.Register(testRegistration("worker-a", "instance-a")); err != nil {
		t.Fatal(err)
	}

	now = now.Add(5 * time.Second)
	if _, err := registry.Heartbeat("worker-a", Heartbeat{
		SchemaVersion: SchemaVersion,
		InstanceID:    "instance-a",
		Status:        testRegistration("worker-a", "instance-a").Status,
	}); !errors.Is(err, ErrLeaseExpired) {
		t.Fatalf("expired heartbeat error = %v", err)
	}
	if inventory := registry.Snapshot(); len(inventory.Workers) != 0 || inventory.Revision != 2 {
		t.Fatalf("expired worker remains visible: %+v", inventory)
	}

	mutation, err := registry.Register(testRegistration("worker-a", "instance-b"))
	if err != nil {
		t.Fatal(err)
	}
	if mutation.InventoryRevision != 3 || mutation.Worker.InstanceID != "instance-b" {
		t.Fatalf("unexpected rejoin mutation: %+v", mutation)
	}
}

func TestRegistryRejectsIncompleteRecords(t *testing.T) {
	registry := New(time.Minute)
	tests := []struct {
		name   string
		mutate func(*Registration)
	}{
		{"schema", func(input *Registration) { input.SchemaVersion++ }},
		{"endpoint", func(input *Registration) { input.Endpoint = "public.example.com" }},
		{"backend", func(input *Registration) { input.Capabilities.Backend = "" }},
		{"memory", func(input *Registration) { input.Capabilities.PhysicalMemoryBytes = 0 }},
		{"adapter", func(input *Registration) { input.Capabilities.Adapters = nil }},
		{"operation", func(input *Registration) { input.Capabilities.Operations = nil }},
		{"transport", func(input *Registration) { input.Capabilities.Transports = nil }},
		{"health", func(input *Registration) { input.Status.Health = "unknown" }},
		{"available memory", func(input *Registration) { input.Status.AvailableMemoryBytes = 2049 }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			input := testRegistration("worker-a", "instance-a")
			test.mutate(&input)
			if _, err := registry.Register(input); err == nil {
				t.Fatal("invalid registration was accepted")
			}
		})
	}
}

func testRegistration(id, instanceID string) Registration {
	return Registration{
		SchemaVersion: SchemaVersion,
		ID:            id, InstanceID: instanceID, Endpoint: "http://" + id + ":8080",
		Capabilities: Capabilities{
			Backend: "test", Runtime: "fixture", OS: "linux", Architecture: "arm64",
			Device: "cpu", PhysicalMemoryBytes: 2048,
			Adapters: []string{"fixture"}, Operations: []string{"decode", "prefill"},
			Admission: AdmissionLimits{
				MaxConcurrentRequests: 1, MaxOpenSequencesPerShard: 2, RetainedByteBudget: 1024,
			},
			Transports: []Transport{{
				Protocol: "http-json-v1", TensorEncodings: []string{"base64"}, MaxRequestBytes: 4096,
			}},
		},
		Status: Status{Health: HealthHealthy, AvailableMemoryBytes: 2048},
	}
}

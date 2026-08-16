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
	statusObservedAt := inventory.Workers[0].StatusObservedAt
	if !statusObservedAt.Equal(now) || !inventory.Workers[0].StatusFresh(now, time.Second) {
		t.Fatalf("registration status freshness was not server-stamped: %+v", inventory.Workers[0])
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
	staleStatus := first.Status
	if _, err := registry.Heartbeat("worker-a", Heartbeat{
		SchemaVersion: SchemaVersion, InstanceID: "new-instance", Status: &staleStatus,
	}); !errors.Is(err, ErrStaleInstance) {
		t.Fatalf("stale heartbeat error = %v", err)
	}

	now = now.Add(4 * time.Second)
	leaseOnly, err := registry.Heartbeat("worker-a", Heartbeat{
		SchemaVersion: SchemaVersion, InstanceID: "instance-a",
	})
	if err != nil {
		t.Fatal(err)
	}
	if leaseOnly.InventoryRevision != 2 || !leaseOnly.Worker.LastSeen.Equal(now) ||
		!leaseOnly.Worker.ExpiresAt.Equal(now.Add(10*time.Second)) ||
		!leaseOnly.Worker.StatusObservedAt.Equal(statusObservedAt) ||
		leaseOnly.Worker.Status.RestartCount != first.Status.RestartCount {
		t.Fatalf("lease-only heartbeat changed dynamic status: %+v", leaseOnly)
	}
	if !leaseOnly.Worker.StatusFresh(now, 4*time.Second) ||
		leaseOnly.Worker.StatusFresh(now, 3*time.Second) ||
		leaseOnly.Worker.StatusFresh(statusObservedAt.Add(-time.Second), time.Minute) ||
		leaseOnly.Worker.StatusFresh(now, 0) {
		t.Fatalf("unexpected conservative freshness result: %+v", leaseOnly.Worker)
	}
	freshInventory := Inventory{
		GeneratedAt: now, LeaseTTLMillis: (10 * time.Second).Milliseconds(),
	}
	staleInventory := freshInventory
	staleInventory.GeneratedAt = now.Add(7 * time.Second)
	if !freshInventory.WorkerStatusFresh(leaseOnly.Worker) ||
		staleInventory.WorkerStatusFresh(leaseOnly.Worker) {
		t.Fatalf("inventory lease TTL did not provide a conservative freshness window")
	}

	now = now.Add(time.Second)
	status := first.Status
	status.AvailableMemoryBytes--
	heartbeat, err := registry.Heartbeat("worker-a", Heartbeat{
		SchemaVersion: SchemaVersion, InstanceID: "instance-a", Status: &status,
	})
	if err != nil {
		t.Fatal(err)
	}
	if heartbeat.InventoryRevision != 3 || !heartbeat.Worker.LastSeen.Equal(now) ||
		!heartbeat.Worker.ExpiresAt.Equal(now.Add(10*time.Second)) ||
		!heartbeat.Worker.StatusObservedAt.Equal(now) {
		t.Fatalf("unexpected heartbeat mutation: %+v", heartbeat)
	}
	now = now.Add(time.Second)
	if renewed, err := registry.Heartbeat("worker-a", Heartbeat{
		SchemaVersion: SchemaVersion, InstanceID: "instance-a", Status: &status,
	}); err != nil || renewed.InventoryRevision != 3 || !renewed.Worker.StatusObservedAt.Equal(now) {
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

func TestRegistrationWithoutFreshStatusRequiresANewObservation(t *testing.T) {
	now := time.Date(2026, time.August, 16, 12, 0, 0, 0, time.UTC)
	registry := New(5*time.Second, WithClock(func() time.Time { return now }))
	input := testRegistration("worker-a", "instance-a")
	registered, err := registry.register(input, false)
	if err != nil {
		t.Fatal(err)
	}
	if !registered.Worker.StatusObservedAt.IsZero() ||
		registered.Worker.StatusFresh(now, time.Minute) {
		t.Fatalf("cached registration was represented as fresh: %+v", registered)
	}

	now = now.Add(time.Second)
	leaseOnly, err := registry.Heartbeat("worker-a", Heartbeat{
		SchemaVersion: SchemaVersion, InstanceID: "instance-a",
	})
	if err != nil {
		t.Fatal(err)
	}
	if leaseOnly.InventoryRevision != 1 || !leaseOnly.Worker.StatusObservedAt.IsZero() {
		t.Fatalf("lease renewal published cached status: %+v", leaseOnly)
	}

	now = now.Add(time.Second)
	status := input.Status
	observed, err := registry.Heartbeat("worker-a", Heartbeat{
		SchemaVersion: SchemaVersion, InstanceID: "instance-a", Status: &status,
	})
	if err != nil {
		t.Fatal(err)
	}
	if observed.InventoryRevision != 2 || !observed.Worker.StatusObservedAt.Equal(now) ||
		!observed.Worker.StatusFresh(now, 5*time.Second) {
		t.Fatalf("fresh status did not restore placement eligibility: %+v", observed)
	}
}

func TestRegistryCanonicalizesEndpointUniqueness(t *testing.T) {
	tests := []struct {
		name          string
		firstEndpoint string
		aliasEndpoint string
		wantEndpoint  string
	}{
		{
			name: "hostname case", firstEndpoint: "http://worker-a:8080",
			aliasEndpoint: "HTTP://WORKER-A:8080/", wantEndpoint: "http://worker-a:8080",
		},
		{
			name: "default HTTP port", firstEndpoint: "http://worker-a",
			aliasEndpoint: "http://WORKER-A:80/", wantEndpoint: "http://worker-a",
		},
		{
			name: "default HTTPS port", firstEndpoint: "https://worker-a",
			aliasEndpoint: "HTTPS://WORKER-A:443/", wantEndpoint: "https://worker-a",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			registry := New(time.Minute)
			first := testRegistration("worker-a", "instance-a")
			first.Endpoint = test.firstEndpoint
			mutation, err := registry.Register(first)
			if err != nil {
				t.Fatal(err)
			}
			if mutation.Worker.Endpoint != test.wantEndpoint {
				t.Fatalf("canonical endpoint = %q, want %q", mutation.Worker.Endpoint, test.wantEndpoint)
			}
			alias := testRegistration("worker-b", "instance-b")
			alias.Endpoint = test.aliasEndpoint
			if _, err := registry.Register(alias); !errors.Is(err, ErrDuplicateEndpoint) {
				t.Fatalf("endpoint alias registration error = %v", err)
			}
		})
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

package registry

import "time"

const SchemaVersion = 1

type Health string

const (
	HealthHealthy  Health = "healthy"
	HealthDegraded Health = "degraded"
	HealthDraining Health = "draining"
)

// Transport describes one trusted-network protocol a worker can accept.
type Transport struct {
	Protocol        string   `json:"protocol"`
	TensorEncodings []string `json:"tensorEncodings"`
	MaxRequestBytes int64    `json:"maxRequestBytes"`
	TLS             bool     `json:"tls"`
}

// AdmissionLimits are the worker-advertised limits placement must respect.
type AdmissionLimits struct {
	MaxConcurrentRequests    int    `json:"maxConcurrentRequests"`
	MaxOpenSequencesPerShard int    `json:"maxOpenSequencesPerShard"`
	RetainedByteBudget       uint64 `json:"retainedByteBudget"`
}

// Capabilities are relatively stable for one worker process incarnation.
// Backend-specific values remain strings so the control plane is not coupled
// to MLX, Metal, CUDA, or a particular model family.
type Capabilities struct {
	Backend                string          `json:"backend"`
	Runtime                string          `json:"runtime"`
	OS                     string          `json:"os"`
	Architecture           string          `json:"architecture"`
	Device                 string          `json:"device"`
	PhysicalMemoryBytes    uint64          `json:"physicalMemoryBytes"`
	Adapters               []string        `json:"adapters"`
	Operations             []string        `json:"operations"`
	CheckpointFingerprints []string        `json:"checkpointFingerprints"`
	Admission              AdmissionLimits `json:"admission"`
	Transports             []Transport     `json:"transports"`
}

// RetainedShard is the scheduler-relevant part of one resident shard.
type RetainedShard struct {
	ID                    string `json:"id"`
	ModelID               string `json:"modelID"`
	CheckpointFingerprint string `json:"checkpointFingerprint"`
	LayerStart            int    `json:"layerStart"`
	LayerEnd              int    `json:"layerEnd"`
	OwnsInput             bool   `json:"ownsInput"`
	OwnsOutput            bool   `json:"ownsOutput"`
	MemoryBytes           uint64 `json:"memoryBytes"`
	OpenSequenceCount     int    `json:"openSequenceCount"`
}

// Status is the dynamic worker state refreshed by heartbeats.
type Status struct {
	Health                    Health          `json:"health"`
	AvailableMemoryBytes      uint64          `json:"availableMemoryBytes"`
	MemoryPressureBytes       uint64          `json:"memoryPressureBytes"`
	OpenSequenceCount         int             `json:"openSequenceCount"`
	RetainedBytes             uint64          `json:"retainedBytes"`
	RestartCount              int             `json:"restartCount"`
	RecentFailureCount        int             `json:"recentFailureCount"`
	WorkerObservationSequence uint64          `json:"workerObservationSequence,omitempty"`
	RetainedShards            []RetainedShard `json:"retainedShards"`
}

// Registration is supplied by a worker when it joins or refreshes its stable
// capabilities. InstanceID distinguishes restarts that reuse the same stable
// worker ID.
type Registration struct {
	SchemaVersion int          `json:"schemaVersion"`
	ID            string       `json:"id"`
	InstanceID    string       `json:"instanceID"`
	Endpoint      string       `json:"endpoint"`
	Capabilities  Capabilities `json:"capabilities"`
	Status        Status       `json:"status"`
}

// RegistrationRequest distinguishes a newly sampled status from cached state
// carried while reclaiming a lease after control-plane loss.
type RegistrationRequest struct {
	Registration
	StatusFresh bool `json:"statusFresh"`
}

type Heartbeat struct {
	SchemaVersion int     `json:"schemaVersion"`
	InstanceID    string  `json:"instanceID"`
	Status        *Status `json:"status,omitempty"`
}

// Worker is the server-authoritative leased membership record.
type Worker struct {
	Registration
	RegisteredAt     time.Time `json:"registeredAt"`
	LastSeen         time.Time `json:"lastSeen"`
	ExpiresAt        time.Time `json:"expiresAt"`
	StatusObservedAt time.Time `json:"statusObservedAt"`
}

// StatusFresh reports whether the server-observed worker state is recent
// enough to be considered by placement. Non-positive windows and timestamps
// from after the caller's reference time fail conservatively.
func (worker Worker) StatusFresh(at time.Time, maxAge time.Duration) bool {
	if maxAge <= 0 || worker.StatusObservedAt.IsZero() || worker.StatusObservedAt.After(at) {
		return false
	}
	return at.Sub(worker.StatusObservedAt) <= maxAge
}

type Mutation struct {
	InventoryRevision uint64 `json:"inventoryRevision"`
	Worker            Worker `json:"worker"`
}

type Inventory struct {
	SchemaVersion  int       `json:"schemaVersion"`
	Revision       uint64    `json:"revision"`
	GeneratedAt    time.Time `json:"generatedAt"`
	LeaseTTLMillis int64     `json:"leaseTTLMillis"`
	Workers        []Worker  `json:"workers"`
}

// WorkerStatusFresh applies the inventory lease TTL as the default maximum
// status age. Placement may impose a stricter window through Worker.StatusFresh.
func (inventory Inventory) WorkerStatusFresh(worker Worker) bool {
	maxAge := time.Duration(inventory.LeaseTTLMillis) * time.Millisecond
	return worker.StatusFresh(inventory.GeneratedAt, maxAge)
}

type APIError struct {
	Code  string `json:"code"`
	Error string `json:"error"`
}

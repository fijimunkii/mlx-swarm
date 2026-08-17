// Package meshstress provides protocol-level synthetic membership and
// placement scenarios without pretending that synthetic peers execute model
// math.
package meshstress

import (
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/placement"
	"github.com/fijimunkii/mlx-swarm/internal/registry"
)

const (
	SchemaVersion      = 1
	DefaultWorkerCount = 32
)

// Config bounds the reproducible synthetic scale scenario.
type Config struct {
	WorkerCount         int
	MaxSearchOperations uint64
	MaxDecisionDuration time.Duration
}

// DefaultConfig exercises the M8 control-plane scale boundary while keeping
// the proof inexpensive enough for a standard Linux CI runner.
func DefaultConfig() Config {
	return Config{
		WorkerCount: DefaultWorkerCount, MaxSearchOperations: 20_000,
		MaxDecisionDuration: 5 * time.Second,
	}
}

// WorkerSpec is a complete synthetic worker advertisement plus the topology
// and compute hints used by placement. Every field maps to an existing
// production registry or profile contract.
type WorkerSpec struct {
	ID                      string
	InstanceID              string
	Endpoint                string
	Backend                 string
	Runtime                 string
	OS                      string
	Architecture            string
	Device                  string
	PhysicalMemoryBytes     uint64
	AvailableMemoryBytes    uint64
	MemoryPressureBytes     uint64
	Adapters                []string
	Operations              []string
	CheckpointFingerprints  []string
	Admission               registry.AdmissionLimits
	Transports              []registry.Transport
	Health                  registry.Health
	RestartCount            int
	RecentFailureCount      int
	RetainedBytes           uint64
	RetainedShards          []registry.RetainedShard
	RTTMicros               uint64
	BandwidthBytesPerSecond uint64
	PrefillMicrosPerLayer   uint64
	DecodeMicrosPerLayer    uint64
}

// Registration returns the exact versioned record sent through the membership
// HTTP protocol by the synthetic peer.
func (spec WorkerSpec) Registration() registry.Registration {
	openSequences := 0
	for _, shard := range spec.RetainedShards {
		openSequences += shard.OpenSequenceCount
	}
	return registry.Registration{
		SchemaVersion: registry.SchemaVersion,
		ID:            spec.ID, InstanceID: spec.InstanceID, Endpoint: spec.Endpoint,
		Capabilities: registry.Capabilities{
			Backend: spec.Backend, Runtime: spec.Runtime, OS: spec.OS,
			Architecture: spec.Architecture, Device: spec.Device,
			PhysicalMemoryBytes: spec.PhysicalMemoryBytes,
			Adapters:            append([]string(nil), spec.Adapters...),
			Operations:          append([]string(nil), spec.Operations...),
			CheckpointFingerprints: append(
				[]string(nil), spec.CheckpointFingerprints...,
			),
			Admission:  spec.Admission,
			Transports: append([]registry.Transport(nil), spec.Transports...),
		},
		Status: registry.Status{
			Health: spec.Health, AvailableMemoryBytes: spec.AvailableMemoryBytes,
			MemoryPressureBytes: spec.MemoryPressureBytes,
			OpenSequenceCount:   openSequences, RetainedBytes: spec.RetainedBytes,
			RestartCount: spec.RestartCount, RecentFailureCount: spec.RecentFailureCount,
			WorkerObservationSequence: 1,
			RetainedShards:            append([]registry.RetainedShard(nil), spec.RetainedShards...),
		},
	}
}

type TransitionEvidence struct {
	Name              string    `json:"name"`
	ObservedAt        time.Time `json:"observedAt"`
	InventoryRevision uint64    `json:"inventoryRevision"`
	WorkerCount       int       `json:"workerCount"`
	WorkerIDs         []string  `json:"workerIDs"`
	Outcome           string    `json:"outcome"`
}

type WorkerRejectionEvidence struct {
	WorkerID string                    `json:"workerID"`
	Codes    []placement.RejectionCode `json:"codes"`
}

type DecisionEvidence struct {
	Name                     string                    `json:"name"`
	InventoryRevision        uint64                    `json:"inventoryRevision"`
	ProfileRevision          uint64                    `json:"profileRevision"`
	VisibleWorkerCount       int                       `json:"visibleWorkerCount"`
	EligibleWorkerCount      int                       `json:"eligibleWorkerCount"`
	SearchOperationCount     uint64                    `json:"searchOperationCount"`
	RetainedSearchStateCount uint64                    `json:"retainedSearchStateCount"`
	CompletePlanCount        uint64                    `json:"completePlanCount"`
	DecisionMicros           int64                     `json:"decisionMicros"`
	PlanSignature            string                    `json:"planSignature"`
	SelectedPlan             *placement.PlanEvaluation `json:"selectedPlan,omitempty"`
	RejectedWorkers          []WorkerRejectionEvidence `json:"rejectedWorkers"`
}

type BoundsEvidence struct {
	WorkerCount               int    `json:"workerCount"`
	MaxSearchOperations       uint64 `json:"maxSearchOperations"`
	MaxDecisionMicros         int64  `json:"maxDecisionMicros"`
	NetworkProbeLimit         int    `json:"networkProbeLimit"`
	NetworkProbeCount         int    `json:"networkProbeCount"`
	MembershipRequestCount    uint64 `json:"membershipRequestCount"`
	ProfileLinkSeriesCount    int    `json:"profileLinkSeriesCount"`
	ProfileComputeSeriesCount int    `json:"profileComputeSeriesCount"`
}

type Checks struct {
	ThirtyTwoWorkersVisible     bool `json:"thirtyTwoWorkersVisible"`
	ConcurrentJoinSucceeded     bool `json:"concurrentJoinSucceeded"`
	DuplicateIdentityRejected   bool `json:"duplicateIdentityRejected"`
	ExpiryRemovedWorker         bool `json:"expiryRemovedWorker"`
	RejoinSucceeded             bool `json:"rejoinSucceeded"`
	IncompatibleChangeRejected  bool `json:"incompatibleChangeRejected"`
	UnsuitableWorkersPlanStable bool `json:"unsuitableWorkersPlanStable"`
	BetterWorkerChangedPlan     bool `json:"betterWorkerChangedPlan"`
	RemovedWorkerNotReused      bool `json:"removedWorkerNotReused"`
	SlowWorkerReplanned         bool `json:"slowWorkerReplanned"`
	DeterministicPlacement      bool `json:"deterministicPlacement"`
	RejectionReasonsRecorded    bool `json:"rejectionReasonsRecorded"`
	SearchWorkWithinBound       bool `json:"searchWorkWithinBound"`
	DecisionLatencyWithinBound  bool `json:"decisionLatencyWithinBound"`
	NetworkProbesWithinBound    bool `json:"networkProbesWithinBound"`
	AllPassed                   bool `json:"allPassed"`
}

type Result struct {
	SchemaVersion int                       `json:"schemaVersion"`
	Model         generation.ExecutionModel `json:"model"`
	Bounds        BoundsEvidence            `json:"bounds"`
	Transitions   []TransitionEvidence      `json:"transitions"`
	Decisions     []DecisionEvidence        `json:"decisions"`
	Checks        Checks                    `json:"checks"`
}

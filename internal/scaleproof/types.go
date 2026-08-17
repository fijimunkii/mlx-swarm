package scaleproof

import (
	"net/http"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/benchmark"
	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/mesh"
	"github.com/fijimunkii/mlx-swarm/internal/placement"
	"github.com/fijimunkii/mlx-swarm/internal/pooledproof"
	"github.com/fijimunkii/mlx-swarm/internal/registry"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

const (
	SchemaVersion             = 1
	RequiredNodeCount         = 5
	DefaultSmallModelID       = "mlx-community/gemma-3-270m-it-4bit"
	DefaultPrompt             = "Write a short story about two computers working together:"
	DefaultTokenCount         = 32
	DefaultSyntheticPeerCount = 27
)

type CoordinatorEvidence struct {
	ID           string `json:"id"`
	RunID        string `json:"runID"`
	OS           string `json:"os"`
	Architecture string `json:"architecture"`
}

type Node struct {
	ID           string
	Endpoint     string
	Capabilities pooledproof.Capabilities
	ProbeMicros  int64
	Caller       workerproc.PersistentCaller
}

type NodeEvidence struct {
	ID           string                   `json:"id"`
	Endpoint     string                   `json:"endpoint"`
	ProbeMicros  int64                    `json:"probeMicros"`
	Capabilities pooledproof.Capabilities `json:"capabilities"`
}

type RunConfig struct {
	Coordinator                  CoordinatorEvidence
	Nodes                        []Node
	SmallModel                   string
	Prompt                       string
	TokenCount                   int
	Reference                    pooledproof.Reference
	InventoryRevision            string
	EdgeReserveLayers            int
	ExpectedMemoryThresholdBytes int
	RTol                         float64
	ATol                         float64
	ForwardTimeout               time.Duration
	ControlURL                   string
	HTTPClient                   *http.Client
	SyntheticPeerCount           int
}

type TeardownEvidence struct {
	NodeID                    string `json:"nodeID"`
	WorkerObservationSequence uint64 `json:"workerObservationSequence"`
	LoadedShardCount          int    `json:"loadedShardCount"`
	OpenSequenceCount         int    `json:"openSequenceCount"`
	KVCacheBytes              int    `json:"kvCacheBytes"`
	RetainedBytes             int    `json:"retainedBytes"`
}

type RunEvidence struct {
	Name                   string                   `json:"name"`
	StageCount             int                      `json:"stageCount"`
	CriticalPathWorkers    int                      `json:"criticalPathWorkers"`
	NetworkBoundaryCount   int                      `json:"networkBoundaryCount"`
	Plan                   generation.ExecutionPlan `json:"plan"`
	StageLoads             []generation.StageLoad   `json:"stageLoads"`
	Generation             generation.PlannedResult `json:"generation"`
	Prefill                benchmark.PlannedSummary `json:"prefill"`
	Decode                 benchmark.PlannedSummary `json:"decode"`
	All                    benchmark.PlannedSummary `json:"all"`
	TokensMatchExpectation bool                     `json:"tokensMatchExpectation"`
	SequenceStateReleased  bool                     `json:"sequenceStateReleased"`
	Teardown               []TeardownEvidence       `json:"teardown"`
}

type CorrectnessEvidence struct {
	Run            RunEvidence             `json:"run"`
	Verification   generation.Verification `json:"verification"`
	RTol           float64                 `json:"rtol"`
	ATol           float64                 `json:"atol"`
	AllLogitsMatch bool                    `json:"allLogitsMatch"`
}

type PooledNodeEvidence struct {
	NodeID                     string                             `json:"nodeID"`
	Stage                      generation.ExecutionStage          `json:"stage"`
	Load                       workerproc.PersistentShardSnapshot `json:"load"`
	MaxProcessPhysicalBytes    uint64                             `json:"maxProcessPhysicalBytes"`
	WithinPhysicalMemory       bool                               `json:"withinPhysicalMemory"`
	UsesConfiguredMLXThreshold bool                               `json:"usesConfiguredMLXThreshold"`
}

type PooledEvidence struct {
	Run                           RunEvidence          `json:"run"`
	ReferenceModel                string               `json:"referenceModel"`
	ReferenceCheckpointBytes      uint64               `json:"referenceCheckpointBytes"`
	ReferenceFullProcessPeak      uint64               `json:"referenceFullProcessPeak"`
	CheckpointMatchesReference    bool                 `json:"checkpointMatchesReference"`
	PromptTokensMatchReference    bool                 `json:"promptTokensMatchReference"`
	GeneratedTokensMatchReference bool                 `json:"generatedTokensMatchReference"`
	NoServingFullModelOracle      bool                 `json:"noServingFullModelOracle"`
	ComplementaryShardsOnly       bool                 `json:"complementaryShardsOnly"`
	Nodes                         []PooledNodeEvidence `json:"nodes"`
}

type HybridSyntheticRejection struct {
	WorkerID string                    `json:"workerID"`
	Codes    []placement.RejectionCode `json:"codes"`
}

type HybridEvidence struct {
	InventoryRevision             uint64                     `json:"inventoryRevision"`
	InventoryGeneratedAt          time.Time                  `json:"inventoryGeneratedAt"`
	InventoryWorkerCount          int                        `json:"inventoryWorkerCount"`
	RealWorkerIDs                 []string                   `json:"realWorkerIDs"`
	SyntheticWorkerIDs            []string                   `json:"syntheticWorkerIDs"`
	Selection                     mesh.SequenceSelection     `json:"selection"`
	SyntheticRejections           []HybridSyntheticRejection `json:"syntheticRejections"`
	Run                           RunEvidence                `json:"run"`
	SelectedRealWorkersOnly       bool                       `json:"selectedRealWorkersOnly"`
	SelectedFiveDistinctWorkers   bool                       `json:"selectedFiveDistinctWorkers"`
	EverySyntheticWorkerRejected  bool                       `json:"everySyntheticWorkerRejected"`
	GeneratedTokensMatchReference bool                       `json:"generatedTokensMatchReference"`
	SequenceStateReleased         bool                       `json:"sequenceStateReleased"`
	PostRunWorkers                []HybridWorkerObservation  `json:"postRunWorkers"`
}

type HybridWorkerObservation struct {
	WorkerID                  string `json:"workerID"`
	WorkerObservationSequence uint64 `json:"workerObservationSequence"`
	LoadedShardCount          int    `json:"loadedShardCount"`
	OpenSequenceCount         int    `json:"openSequenceCount"`
	KVCacheBytes              int    `json:"kvCacheBytes"`
	RetainedBytes             int    `json:"retainedBytes"`
}

type Checks struct {
	FiveDistinctNodes           bool `json:"fiveDistinctNodes"`
	CleanWorkersAtStart         bool `json:"cleanWorkersAtStart"`
	SmallModelLogitsMatch       bool `json:"smallModelLogitsMatch"`
	SmallModelTokensMatch       bool `json:"smallModelTokensMatch"`
	ScalingRunsTwoThroughFive   bool `json:"scalingRunsTwoThroughFive"`
	ScalingTokensMatch          bool `json:"scalingTokensMatch"`
	PooledCheckpointMatches     bool `json:"pooledCheckpointMatches"`
	PooledPromptTokensMatch     bool `json:"pooledPromptTokensMatch"`
	PooledGeneratedTokensMatch  bool `json:"pooledGeneratedTokensMatch"`
	PooledWorkersWithinMemory   bool `json:"pooledWorkersWithinMemory"`
	NoServingFullModelOracle    bool `json:"noServingFullModelOracle"`
	SequenceStateReleased       bool `json:"sequenceStateReleased"`
	WorkersCleanAfterProof      bool `json:"workersCleanAfterProof"`
	HybridInventoryAtLeast32    bool `json:"hybridInventoryAtLeast32"`
	HybridSelectedRealWorkers   bool `json:"hybridSelectedRealWorkers"`
	HybridRejectedSynthetic     bool `json:"hybridRejectedSynthetic"`
	HybridTokensMatch           bool `json:"hybridTokensMatch"`
	HybridSequenceStateReleased bool `json:"hybridSequenceStateReleased"`
	AllPassed                   bool `json:"allPassed"`
}

type Result struct {
	SchemaVersion int                 `json:"schemaVersion"`
	Coordinator   CoordinatorEvidence `json:"coordinator"`
	Nodes         []NodeEvidence      `json:"nodes"`
	Correctness   CorrectnessEvidence `json:"correctness"`
	Scaling       []RunEvidence       `json:"scaling"`
	PooledMemory  PooledEvidence      `json:"pooledMemory"`
	Hybrid        HybridEvidence      `json:"hybrid"`
	Checks        Checks              `json:"checks"`
}

type fixedInventory struct {
	inventory registry.Inventory
}

func (source fixedInventory) Snapshot() registry.Inventory { return source.inventory }

package placement

import (
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/registry"
)

const SchemaVersion = 1

type RejectionCode string

const (
	RejectionStaleStatus               RejectionCode = "stale_status"
	RejectionUnhealthy                 RejectionCode = "unhealthy"
	RejectionUnsupportedAdapter        RejectionCode = "unsupported_adapter"
	RejectionIncompatibleCheckpoint    RejectionCode = "incompatible_checkpoint"
	RejectionUnsupportedOperation      RejectionCode = "unsupported_operation"
	RejectionUnsupportedTransport      RejectionCode = "unsupported_transport"
	RejectionUnsupportedEncoding       RejectionCode = "unsupported_tensor_encoding"
	RejectionTLSRequired               RejectionCode = "tls_required"
	RejectionInsufficientMemory        RejectionCode = "insufficient_memory"
	RejectionRetainedBudgetExceeded    RejectionCode = "retained_budget_exceeded"
	RejectionSequenceCapacityExhausted RejectionCode = "sequence_capacity_exhausted"
)

type TransportRequirement struct {
	Protocol       string `json:"protocol"`
	TensorEncoding string `json:"tensorEncoding"`
	RequireTLS     bool   `json:"requireTLS"`
}

// StageRequirement describes the known constraints for one proposed
// contiguous stage. LoadMemoryBytes is incremental model memory when a
// compatible retained shard cannot be reused. SequenceMemoryBytes is the
// retained KV/output reserve for the new sequence.
type StageRequirement struct {
	Model               generation.ExecutionModel `json:"model"`
	Adapter             string                    `json:"adapter"`
	ShardID             string                    `json:"shardID,omitempty"`
	LayerStart          int                       `json:"layerStart"`
	LayerEnd            int                       `json:"layerEnd"`
	OwnsInput           bool                      `json:"ownsInput"`
	OwnsOutput          bool                      `json:"ownsOutput"`
	LoadMemoryBytes     uint64                    `json:"loadMemoryBytes"`
	SequenceMemoryBytes uint64                    `json:"sequenceMemoryBytes"`
	Transport           TransportRequirement      `json:"transport"`
	StatusMaxAgeMillis  int64                     `json:"statusMaxAgeMillis,omitempty"`
}

type Rejection struct {
	Code   RejectionCode `json:"code"`
	Detail string        `json:"detail"`
}

type Candidate struct {
	WorkerID                      string      `json:"workerID"`
	Endpoint                      string      `json:"endpoint"`
	Eligible                      bool        `json:"eligible"`
	ReusesRetainedShard           bool        `json:"reusesRetainedShard"`
	RetainedShardID               string      `json:"retainedShardID,omitempty"`
	RequiredAdditionalMemoryBytes uint64      `json:"requiredAdditionalMemoryBytes"`
	Rejections                    []Rejection `json:"rejections"`
}

type Evaluation struct {
	SchemaVersion        int              `json:"schemaVersion"`
	InventoryRevision    uint64           `json:"inventoryRevision"`
	InventoryGeneratedAt time.Time        `json:"inventoryGeneratedAt"`
	StatusMaxAgeMillis   int64            `json:"statusMaxAgeMillis"`
	Requirement          StageRequirement `json:"requirement"`
	RequiredOperations   []string         `json:"requiredOperations"`
	Candidates           []Candidate      `json:"candidates"`
}

// EvaluateCandidates applies deterministic hard constraints to every worker
// in an inventory. It does not score eligible workers or choose a layer split.
func EvaluateCandidates(
	inventory registry.Inventory,
	requirement StageRequirement,
) (Evaluation, error) {
	requirement.Adapter = strings.TrimSpace(requirement.Adapter)
	requirement.ShardID = strings.TrimSpace(requirement.ShardID)
	requirement.Transport.Protocol = strings.TrimSpace(requirement.Transport.Protocol)
	requirement.Transport.TensorEncoding = strings.TrimSpace(
		requirement.Transport.TensorEncoding,
	)
	statusMaxAgeMillis, err := validateInputs(inventory, requirement)
	if err != nil {
		return Evaluation{}, err
	}
	statusMaxAge := time.Duration(statusMaxAgeMillis) * time.Millisecond
	requiredOperations := stageOperations(requirement)
	workers := append([]registry.Worker(nil), inventory.Workers...)
	slices.SortFunc(workers, func(left, right registry.Worker) int {
		return strings.Compare(left.ID, right.ID)
	})

	evaluation := Evaluation{
		SchemaVersion: SchemaVersion, InventoryRevision: inventory.Revision,
		InventoryGeneratedAt: inventory.GeneratedAt, StatusMaxAgeMillis: statusMaxAgeMillis,
		Requirement: requirement, RequiredOperations: requiredOperations,
		Candidates: make([]Candidate, len(workers)),
	}
	for index, worker := range workers {
		evaluation.Candidates[index] = evaluateCandidate(
			inventory.GeneratedAt, statusMaxAge, requirement, requiredOperations, worker,
		)
	}
	return evaluation, nil
}

func validateInputs(inventory registry.Inventory, requirement StageRequirement) (int64, error) {
	if inventory.SchemaVersion != registry.SchemaVersion {
		return 0, fmt.Errorf(
			"inventory schema version is %d, want %d",
			inventory.SchemaVersion, registry.SchemaVersion,
		)
	}
	if inventory.GeneratedAt.IsZero() {
		return 0, errors.New("inventory generation time is required")
	}
	if inventory.LeaseTTLMillis <= 0 {
		return 0, errors.New("inventory lease TTL must be positive")
	}
	if requirement.Model.ID == "" || requirement.Model.CheckpointFingerprint == "" ||
		requirement.Model.LayerCount <= 0 {
		return 0, errors.New("placement model identity is incomplete")
	}
	if requirement.Adapter == "" {
		return 0, errors.New("placement adapter is required")
	}
	if requirement.LayerStart < 0 || requirement.LayerEnd <= requirement.LayerStart ||
		requirement.LayerEnd > requirement.Model.LayerCount {
		return 0, fmt.Errorf(
			"placement stage range [%d,%d) is invalid for %d layers",
			requirement.LayerStart, requirement.LayerEnd, requirement.Model.LayerCount,
		)
	}
	if requirement.OwnsInput != (requirement.LayerStart == 0) {
		return 0, errors.New("placement stage input ownership does not match its range")
	}
	if requirement.OwnsOutput != (requirement.LayerEnd == requirement.Model.LayerCount) {
		return 0, errors.New("placement stage output ownership does not match its range")
	}
	if requirement.LoadMemoryBytes == 0 || requirement.SequenceMemoryBytes == 0 {
		return 0, errors.New("placement memory estimates must be positive")
	}
	if requirement.LoadMemoryBytes > ^uint64(0)-requirement.SequenceMemoryBytes {
		return 0, errors.New("placement memory estimate overflows uint64")
	}
	if requirement.Transport.Protocol == "" || requirement.Transport.TensorEncoding == "" {
		return 0, errors.New("placement transport protocol and tensor encoding are required")
	}
	if requirement.StatusMaxAgeMillis < 0 {
		return 0, errors.New("placement status maximum age cannot be negative")
	}
	statusMaxAgeMillis := requirement.StatusMaxAgeMillis
	if statusMaxAgeMillis == 0 {
		statusMaxAgeMillis = inventory.LeaseTTLMillis
	}
	const maxDurationMillis = int64(^uint64(0)>>1) / int64(time.Millisecond)
	if statusMaxAgeMillis > maxDurationMillis {
		return 0, errors.New("placement status maximum age is too large")
	}
	return statusMaxAgeMillis, nil
}

func evaluateCandidate(
	at time.Time,
	statusMaxAge time.Duration,
	requirement StageRequirement,
	requiredOperations []string,
	worker registry.Worker,
) Candidate {
	candidate := Candidate{
		WorkerID: worker.ID, Endpoint: worker.Endpoint, Rejections: make([]Rejection, 0),
	}
	retained := compatibleRetainedShard(worker.Status.RetainedShards, requirement)
	if retained != nil {
		candidate.ReusesRetainedShard = true
		candidate.RetainedShardID = retained.ID
	}
	candidate.RequiredAdditionalMemoryBytes = requirement.SequenceMemoryBytes
	if retained == nil {
		candidate.RequiredAdditionalMemoryBytes += requirement.LoadMemoryBytes
	}

	if !worker.StatusFresh(at, statusMaxAge) {
		candidate.reject(RejectionStaleStatus, fmt.Sprintf(
			"status is not fresh within %dms", statusMaxAge.Milliseconds(),
		))
	}
	if worker.Status.Health != registry.HealthHealthy {
		candidate.reject(RejectionUnhealthy, fmt.Sprintf(
			"worker health is %q", worker.Status.Health,
		))
	}
	if !slices.Contains(worker.Capabilities.Adapters, requirement.Adapter) {
		candidate.reject(RejectionUnsupportedAdapter, fmt.Sprintf(
			"adapter %q is not advertised", requirement.Adapter,
		))
	}
	if len(worker.Capabilities.CheckpointFingerprints) > 0 &&
		!slices.Contains(
			worker.Capabilities.CheckpointFingerprints, requirement.Model.CheckpointFingerprint,
		) {
		candidate.reject(RejectionIncompatibleCheckpoint, fmt.Sprintf(
			"checkpoint %q is not advertised", requirement.Model.CheckpointFingerprint,
		))
	}
	missingOperations := make([]string, 0)
	for _, operation := range requiredOperations {
		if !slices.Contains(worker.Capabilities.Operations, operation) {
			missingOperations = append(missingOperations, operation)
		}
	}
	if len(missingOperations) > 0 {
		candidate.reject(RejectionUnsupportedOperation, fmt.Sprintf(
			"operations are not advertised: %s", strings.Join(missingOperations, ","),
		))
	}

	transport := advertisedTransport(worker.Capabilities.Transports, requirement.Transport.Protocol)
	if transport == nil {
		candidate.reject(RejectionUnsupportedTransport, fmt.Sprintf(
			"transport %q is not advertised", requirement.Transport.Protocol,
		))
	} else {
		if !slices.Contains(transport.TensorEncodings, requirement.Transport.TensorEncoding) {
			candidate.reject(RejectionUnsupportedEncoding, fmt.Sprintf(
				"tensor encoding %q is not advertised", requirement.Transport.TensorEncoding,
			))
		}
		if requirement.Transport.RequireTLS && !transport.TLS {
			candidate.reject(RejectionTLSRequired, "transport does not provide TLS")
		}
	}
	if worker.Status.AvailableMemoryBytes < candidate.RequiredAdditionalMemoryBytes {
		candidate.reject(RejectionInsufficientMemory, fmt.Sprintf(
			"additional memory requires %d bytes; %d available",
			candidate.RequiredAdditionalMemoryBytes, worker.Status.AvailableMemoryBytes,
		))
	}
	projectedRetained, overflow := addUint64(
		worker.Status.RetainedBytes, requirement.SequenceMemoryBytes,
	)
	if overflow || projectedRetained > worker.Capabilities.Admission.RetainedByteBudget {
		candidate.reject(RejectionRetainedBudgetExceeded, fmt.Sprintf(
			"retained memory requires %d bytes; budget is %d",
			projectedRetained, worker.Capabilities.Admission.RetainedByteBudget,
		))
	}
	if retained != nil &&
		retained.OpenSequenceCount >= worker.Capabilities.Admission.MaxOpenSequencesPerShard {
		candidate.reject(RejectionSequenceCapacityExhausted, fmt.Sprintf(
			"retained shard has %d open sequences; limit is %d",
			retained.OpenSequenceCount,
			worker.Capabilities.Admission.MaxOpenSequencesPerShard,
		))
	}
	candidate.Eligible = len(candidate.Rejections) == 0
	return candidate
}

func (candidate *Candidate) reject(code RejectionCode, detail string) {
	candidate.Rejections = append(candidate.Rejections, Rejection{Code: code, Detail: detail})
}

func stageOperations(requirement StageRequirement) []string {
	operations := []string{
		"closeSequence", "decode", "loadShard", "modelInfo", "openSequence", "prefill", "state",
	}
	if requirement.OwnsInput {
		operations = append(operations, "detokenize", "tokenize")
	}
	slices.Sort(operations)
	return operations
}

func compatibleRetainedShard(
	shards []registry.RetainedShard,
	requirement StageRequirement,
) *registry.RetainedShard {
	if requirement.ShardID == "" {
		return nil
	}
	for index := range shards {
		shard := &shards[index]
		if shard.ID == requirement.ShardID && shard.ModelID == requirement.Model.ID &&
			shard.CheckpointFingerprint == requirement.Model.CheckpointFingerprint &&
			shard.LayerStart == requirement.LayerStart && shard.LayerEnd == requirement.LayerEnd &&
			shard.OwnsInput == requirement.OwnsInput && shard.OwnsOutput == requirement.OwnsOutput {
			return shard
		}
	}
	return nil
}

func advertisedTransport(transports []registry.Transport, protocol string) *registry.Transport {
	for index := range transports {
		if transports[index].Protocol == protocol {
			return &transports[index]
		}
	}
	return nil
}

func addUint64(left, right uint64) (uint64, bool) {
	if left > ^uint64(0)-right {
		return ^uint64(0), true
	}
	return left + right, false
}

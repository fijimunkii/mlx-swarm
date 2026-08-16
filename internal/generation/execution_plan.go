package generation

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
)

const executionPlanSchemaVersion = "1"

type StageResponseMode string

const (
	StageResponseTensor       StageResponseMode = "tensor"
	StageResponseSampledToken StageResponseMode = "sampledToken"
)

// ExecutionModel pins the checkpoint identity and complete transformer range
// that an execution plan was constructed for.
type ExecutionModel struct {
	ID                    string `json:"id"`
	CheckpointFingerprint string `json:"checkpointFingerprint"`
	LayerCount            int    `json:"layerCount"`
}

// ExecutionStage describes one ordered contiguous checkpoint stage. TargetID
// is a stable scheduler/inventory identity; the concrete transport remains in
// ExecutionTarget so plans can be inspected and persisted independently.
type ExecutionStage struct {
	Name         string            `json:"name"`
	TargetID     string            `json:"targetID"`
	ShardID      string            `json:"shardID"`
	LayerStart   int               `json:"layerStart"`
	LayerEnd     int               `json:"layerEnd"`
	OwnsInput    bool              `json:"ownsInput"`
	OwnsOutput   bool              `json:"ownsOutput"`
	ResponseMode StageResponseMode `json:"responseMode"`
}

// ExecutionPlan is the self-describing, architecture-neutral ordered pipeline
// used by distributed generation. Revision is a deterministic digest of the
// model, inventory revision, and semantic stage fields; shard identities are
// derived from it so differently shaped plans cannot collide on a worker.
type ExecutionPlan struct {
	SchemaVersion     string           `json:"schemaVersion"`
	Revision          string           `json:"revision"`
	InventoryRevision string           `json:"inventoryRevision,omitempty"`
	Model             ExecutionModel   `json:"model"`
	Stages            []ExecutionStage `json:"stages"`
}

// BuildExecutionPlan constructs an immutable plan from caller-selected stage
// ranges. It supplies the schema version, revision, and collision-safe shard
// IDs so experiments can use explicit non-balanced layouts without duplicating
// plan identity logic. Any caller-supplied ShardID values are replaced.
func BuildExecutionPlan(
	model ExecutionModel,
	inventoryRevision string,
	stages []ExecutionStage,
) (ExecutionPlan, error) {
	plan := ExecutionPlan{
		SchemaVersion:     executionPlanSchemaVersion,
		InventoryRevision: inventoryRevision,
		Model:             model,
		Stages:            append([]ExecutionStage(nil), stages...),
	}
	finalizeExecutionPlan(&plan)
	if err := ValidateExecutionPlan(plan); err != nil {
		return ExecutionPlan{}, err
	}
	return plan, nil
}

// BuildBalancedExecutionPlan produces a deterministic contiguous split for
// experiments that need an explicit plan before the dynamic scheduler exists.
// Remainder layers are assigned to earlier targets. This is a correctness
// helper, not placement policy.
func BuildBalancedExecutionPlan(
	model ExecutionModel,
	targetIDs []string,
	terminalResponseMode StageResponseMode,
) (ExecutionPlan, error) {
	if len(targetIDs) == 0 {
		return ExecutionPlan{}, errors.New("execution targets are required")
	}
	if model.LayerCount > 0 && len(targetIDs) > model.LayerCount {
		return ExecutionPlan{}, fmt.Errorf(
			"stage count %d exceeds layer count %d", len(targetIDs), model.LayerCount,
		)
	}

	stages := make([]ExecutionStage, 0, len(targetIDs))
	base := model.LayerCount / len(targetIDs)
	remainder := model.LayerCount % len(targetIDs)
	start := 0
	for index, targetID := range targetIDs {
		size := base
		if index < remainder {
			size++
		}
		end := start + size
		responseMode := StageResponseTensor
		if index == len(targetIDs)-1 {
			responseMode = terminalResponseMode
		}
		stages = append(stages, ExecutionStage{
			Name:         fmt.Sprintf("stage-%d", index),
			TargetID:     targetID,
			LayerStart:   start,
			LayerEnd:     end,
			OwnsInput:    index == 0,
			OwnsOutput:   index == len(targetIDs)-1,
			ResponseMode: responseMode,
		})
		start = end
	}
	return BuildExecutionPlan(model, "", stages)
}

// ValidateExecutionPlan rejects incomplete, ambiguous, stale, or mutated
// pipelines before model state is loaded.
func ValidateExecutionPlan(plan ExecutionPlan) error {
	if plan.SchemaVersion != executionPlanSchemaVersion {
		return fmt.Errorf(
			"unsupported execution plan schema %q; want %q",
			plan.SchemaVersion, executionPlanSchemaVersion,
		)
	}
	if plan.Model.ID == "" {
		return errors.New("execution plan model ID is required")
	}
	if plan.Model.CheckpointFingerprint == "" {
		return errors.New("execution plan checkpoint fingerprint is required")
	}
	if plan.Model.LayerCount <= 0 {
		return errors.New("execution plan layer count must be positive")
	}
	if len(plan.Stages) == 0 {
		return errors.New("execution plan requires at least one stage")
	}
	if len(plan.Stages) > plan.Model.LayerCount {
		return fmt.Errorf(
			"execution plan has %d stages for %d layers",
			len(plan.Stages), plan.Model.LayerCount,
		)
	}

	names := make(map[string]struct{}, len(plan.Stages))
	targets := make(map[string]struct{}, len(plan.Stages))
	shards := make(map[string]struct{}, len(plan.Stages))
	expectedStart := 0
	for index, stage := range plan.Stages {
		if stage.Name == "" {
			return fmt.Errorf("stage %d has no name", index)
		}
		if _, exists := names[stage.Name]; exists {
			return fmt.Errorf("stage name %q is duplicated", stage.Name)
		}
		names[stage.Name] = struct{}{}
		if stage.TargetID == "" {
			return fmt.Errorf("stage %d has no target ID", index)
		}
		if _, exists := targets[stage.TargetID]; exists {
			return fmt.Errorf("target ID %q is duplicated", stage.TargetID)
		}
		targets[stage.TargetID] = struct{}{}
		if stage.ShardID == "" {
			return fmt.Errorf("stage %d has no shard ID", index)
		}
		if _, exists := shards[stage.ShardID]; exists {
			return fmt.Errorf("shard ID %q is duplicated", stage.ShardID)
		}
		shards[stage.ShardID] = struct{}{}
		if stage.LayerStart != expectedStart {
			return fmt.Errorf(
				"stage %d starts at layer %d; expected %d",
				index, stage.LayerStart, expectedStart,
			)
		}
		if stage.LayerEnd <= stage.LayerStart {
			return fmt.Errorf(
				"stage %d has invalid range [%d,%d)",
				index, stage.LayerStart, stage.LayerEnd,
			)
		}
		if stage.LayerEnd > plan.Model.LayerCount {
			return fmt.Errorf(
				"stage %d ends at layer %d beyond model layer count %d",
				index, stage.LayerEnd, plan.Model.LayerCount,
			)
		}
		wantInput := index == 0
		if stage.OwnsInput != wantInput {
			return fmt.Errorf(
				"stage %d input ownership is %t; expected %t",
				index, stage.OwnsInput, wantInput,
			)
		}
		wantOutput := index == len(plan.Stages)-1
		if stage.OwnsOutput != wantOutput {
			return fmt.Errorf(
				"stage %d output ownership is %t; expected %t",
				index, stage.OwnsOutput, wantOutput,
			)
		}
		wantResponse := StageResponseTensor
		if wantOutput {
			wantResponse = stage.ResponseMode
			if wantResponse != StageResponseTensor && wantResponse != StageResponseSampledToken {
				return fmt.Errorf("stage %d has unsupported response mode %q", index, stage.ResponseMode)
			}
		}
		if !wantOutput && stage.ResponseMode != wantResponse {
			return fmt.Errorf(
				"stage %d response mode is %q; intermediate stages must return tensors",
				index, stage.ResponseMode,
			)
		}
		expectedStart = stage.LayerEnd
	}
	if expectedStart != plan.Model.LayerCount {
		return fmt.Errorf(
			"execution plan ends at layer %d; expected %d",
			expectedStart, plan.Model.LayerCount,
		)
	}

	expectedRevision := executionPlanRevision(plan)
	if plan.Revision != expectedRevision {
		return fmt.Errorf(
			"execution plan revision %q does not match contents %q",
			plan.Revision, expectedRevision,
		)
	}
	for index, stage := range plan.Stages {
		expectedShardID := executionShardID(index, stage, plan.Revision)
		if stage.ShardID != expectedShardID {
			return fmt.Errorf(
				"stage %d shard ID %q does not match plan %q",
				index, stage.ShardID, expectedShardID,
			)
		}
	}
	return nil
}

func finalizeExecutionPlan(plan *ExecutionPlan) {
	plan.Revision = executionPlanRevision(*plan)
	for index := range plan.Stages {
		plan.Stages[index].ShardID = executionShardID(index, plan.Stages[index], plan.Revision)
	}
}

func executionPlanRevision(plan ExecutionPlan) string {
	type revisionStage struct {
		Name         string            `json:"name"`
		TargetID     string            `json:"targetID"`
		LayerStart   int               `json:"layerStart"`
		LayerEnd     int               `json:"layerEnd"`
		OwnsInput    bool              `json:"ownsInput"`
		OwnsOutput   bool              `json:"ownsOutput"`
		ResponseMode StageResponseMode `json:"responseMode"`
	}
	payload := struct {
		SchemaVersion     string          `json:"schemaVersion"`
		InventoryRevision string          `json:"inventoryRevision,omitempty"`
		Model             ExecutionModel  `json:"model"`
		Stages            []revisionStage `json:"stages"`
	}{
		SchemaVersion:     plan.SchemaVersion,
		InventoryRevision: plan.InventoryRevision,
		Model:             plan.Model,
		Stages:            make([]revisionStage, len(plan.Stages)),
	}
	for index, stage := range plan.Stages {
		payload.Stages[index] = revisionStage{
			Name: stage.Name, TargetID: stage.TargetID,
			LayerStart: stage.LayerStart, LayerEnd: stage.LayerEnd,
			OwnsInput: stage.OwnsInput, OwnsOutput: stage.OwnsOutput,
			ResponseMode: stage.ResponseMode,
		}
	}
	encoded, _ := json.Marshal(payload)
	digest := sha256.Sum256(encoded)
	return hex.EncodeToString(digest[:])
}

func executionShardID(index int, stage ExecutionStage, revision string) string {
	suffix := revision
	if len(suffix) > 12 {
		suffix = suffix[:12]
	}
	return fmt.Sprintf(
		"generate-stage-%02d-%d-%d-%s",
		index, stage.LayerStart, stage.LayerEnd, suffix,
	)
}

// LegacyShardPlan exposes the exact two-stage shape used by the Distributed
// Inference Proof. It lets compatibility callers retain their public schema
// while both paths execute through the N-stage runtime.
func (plan ExecutionPlan) LegacyShardPlan() (ShardPlan, bool) {
	if len(plan.Stages) != 2 {
		return ShardPlan{}, false
	}
	return ShardPlan{
		Producer: Shard{
			ID:         plan.Stages[0].ShardID,
			LayerStart: plan.Stages[0].LayerStart,
			LayerEnd:   plan.Stages[0].LayerEnd,
		},
		Consumer: Shard{
			ID:         plan.Stages[1].ShardID,
			LayerStart: plan.Stages[1].LayerStart,
			LayerEnd:   plan.Stages[1].LayerEnd,
		},
	}, true
}

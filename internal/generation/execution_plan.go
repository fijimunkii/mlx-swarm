package generation

import (
	"errors"
	"fmt"
)

// ExecutionStage describes one ordered contiguous checkpoint stage. The
// caller/transport used to execute the stage is intentionally kept outside the
// serializable plan so the same plan can be inspected, persisted, or scheduled
// independently from a concrete worker connection.
type ExecutionStage struct {
	Name       string `json:"name"`
	ShardID    string `json:"shardID"`
	LayerStart int    `json:"layerStart"`
	LayerEnd   int    `json:"layerEnd"`
	OwnsInput  bool   `json:"ownsInput"`
	OwnsOutput bool   `json:"ownsOutput"`
}

// ExecutionPlan is the architecture-neutral ordered pipeline used by
// distributed generation. Stages must cover the complete transformer range
// exactly once, with input ownership on the first stage and output ownership on
// the final stage.
type ExecutionPlan struct {
	Stages []ExecutionStage `json:"stages"`
}

// BuildBalancedExecutionPlan produces a deterministic contiguous split for
// experiments that need an explicit N-stage plan before the dynamic scheduler
// exists. Remainder layers are assigned to the earlier stages. This is a
// correctness-oriented helper, not the future placement policy.
func BuildBalancedExecutionPlan(
	modelID string,
	checkpointFingerprint string,
	layerCount int,
	stageCount int,
) (ExecutionPlan, error) {
	if modelID == "" {
		return ExecutionPlan{}, errors.New("model ID is required")
	}
	if checkpointFingerprint == "" {
		return ExecutionPlan{}, errors.New("checkpoint fingerprint is required")
	}
	if layerCount <= 0 {
		return ExecutionPlan{}, errors.New("layer count must be positive")
	}
	if stageCount <= 0 {
		return ExecutionPlan{}, errors.New("stage count must be positive")
	}
	if stageCount > layerCount {
		return ExecutionPlan{}, fmt.Errorf(
			"stage count %d exceeds layer count %d", stageCount, layerCount,
		)
	}

	base := layerCount / stageCount
	remainder := layerCount % stageCount
	suffix := modelHashSuffix(modelID, checkpointFingerprint)
	plan := ExecutionPlan{Stages: make([]ExecutionStage, 0, stageCount)}
	start := 0
	for index := 0; index < stageCount; index++ {
		size := base
		if index < remainder {
			size++
		}
		end := start + size
		plan.Stages = append(plan.Stages, ExecutionStage{
			Name:       fmt.Sprintf("stage-%d", index),
			ShardID:    fmt.Sprintf("generate-stage-%02d-%s", index, suffix),
			LayerStart: start,
			LayerEnd:   end,
			OwnsInput:  index == 0,
			OwnsOutput: index == stageCount-1,
		})
		start = end
	}
	if err := ValidateExecutionPlan(plan, layerCount); err != nil {
		return ExecutionPlan{}, err
	}
	return plan, nil
}

// ValidateExecutionPlan rejects incomplete or ambiguous pipelines before any
// checkpoint state is loaded. A valid plan covers layers [0, layerCount)
// contiguously, uses unique names and shard IDs, and gives input/output
// ownership only to the first/final stage respectively.
func ValidateExecutionPlan(plan ExecutionPlan, layerCount int) error {
	if layerCount <= 0 {
		return errors.New("layer count must be positive")
	}
	if len(plan.Stages) == 0 {
		return errors.New("execution plan requires at least one stage")
	}
	if len(plan.Stages) > layerCount {
		return fmt.Errorf(
			"execution plan has %d stages for %d layers", len(plan.Stages), layerCount,
		)
	}

	names := make(map[string]struct{}, len(plan.Stages))
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
		if stage.LayerEnd > layerCount {
			return fmt.Errorf(
				"stage %d ends at layer %d beyond model layer count %d",
				index, stage.LayerEnd, layerCount,
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
		expectedStart = stage.LayerEnd
	}
	if expectedStart != layerCount {
		return fmt.Errorf(
			"execution plan ends at layer %d; expected %d", expectedStart, layerCount,
		)
	}
	return nil
}

// LegacyShardPlan exposes the exact two-stage shape used by the Distributed
// Inference Proof. It lets compatibility callers continue to consume the old
// producer/consumer schema while the runtime transitions to ExecutionPlan.
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

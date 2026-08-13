package scaleproof

import (
	"errors"
	"fmt"

	"github.com/fijimunkii/mlx-swarm/internal/generation"
)

// OwnershipAwarePlan keeps embedding/output ownership off the layer-heavy
// middle stages. It is an explicit proof layout, not automatic placement.
func OwnershipAwarePlan(
	model generation.ExecutionModel,
	inventoryRevision string,
	targetIDs []string,
	edgeReserveLayers int,
) (generation.ExecutionPlan, error) {
	if len(targetIDs) < 2 {
		return generation.ExecutionPlan{}, errors.New("ownership-aware plan requires at least two targets")
	}
	if model.LayerCount < len(targetIDs) {
		return generation.ExecutionPlan{}, fmt.Errorf(
			"stage count %d exceeds layer count %d", len(targetIDs), model.LayerCount,
		)
	}
	if edgeReserveLayers < 0 {
		return generation.ExecutionPlan{}, errors.New("edge reserve must be non-negative")
	}

	// Balance a virtual layer count that includes the non-transformer work on
	// the input/output owners, then remove that reserve from the edge ranges.
	virtualLayers := model.LayerCount + 2*edgeReserveLayers
	sizes := make([]int, len(targetIDs))
	base, remainder := virtualLayers/len(targetIDs), virtualLayers%len(targetIDs)
	for index := range sizes {
		sizes[index] = base
		if index < remainder {
			sizes[index]++
		}
	}
	sizes[0] -= edgeReserveLayers
	sizes[len(sizes)-1] -= edgeReserveLayers
	for index, size := range sizes {
		if size <= 0 {
			return generation.ExecutionPlan{}, fmt.Errorf(
				"edge reserve leaves stage %d with %d transformer layers", index, size,
			)
		}
	}

	stages := make([]generation.ExecutionStage, len(targetIDs))
	start := 0
	for index, targetID := range targetIDs {
		end := start + sizes[index]
		mode := generation.StageResponseTensor
		if index == len(targetIDs)-1 {
			mode = generation.StageResponseSampledToken
		}
		stages[index] = generation.ExecutionStage{
			Name: fmt.Sprintf("stage-%d", index), TargetID: targetID,
			LayerStart: start, LayerEnd: end,
			OwnsInput: index == 0, OwnsOutput: index == len(targetIDs)-1,
			ResponseMode: mode,
		}
		start = end
	}
	return generation.BuildExecutionPlan(model, inventoryRevision, stages)
}

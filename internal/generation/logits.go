package generation

import (
	"errors"
	"fmt"
	"math"

	"github.com/fijimunkii/mlx-swarm/internal/tensorcheck"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

func greedyToken(tensor workerproc.WireTensor) (int32, error) {
	if len(tensor.Shape) == 0 {
		return 0, errors.New("logit tensor has no dimensions")
	}
	vocabulary := tensor.Shape[len(tensor.Shape)-1]
	values, err := tensorcheck.FinalValues(tensor, vocabulary)
	if err != nil {
		return 0, err
	}
	bestIndex := 0
	bestValue := values[0]
	if math.IsNaN(bestValue) {
		return 0, errors.New("NaN at vocabulary index 0")
	}
	for index := 1; index < len(values); index++ {
		if math.IsNaN(values[index]) {
			return 0, fmt.Errorf("NaN at vocabulary index %d", index)
		}
		if values[index] > bestValue {
			bestIndex = index
			bestValue = values[index]
		}
	}
	if bestIndex > math.MaxInt32 {
		return 0, fmt.Errorf("greedy token index %d exceeds int32", bestIndex)
	}
	return int32(bestIndex), nil
}

func sampledToken(sampled *int32, logits workerproc.WireTensor) (int32, error) {
	if sampled == nil {
		if len(logits.Data) == 0 {
			return 0, errors.New("sampled-token response contains neither a token nor logits")
		}
		return greedyToken(logits)
	}
	if *sampled < 0 {
		return 0, fmt.Errorf("sampled-token response contains negative token ID %d", *sampled)
	}
	if len(logits.Data) != 0 {
		return 0, errors.New("sampled-token response also contains logits")
	}
	return *sampled, nil
}

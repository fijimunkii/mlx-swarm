package benchmark

import (
	"errors"
	"fmt"
	"sort"

	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

// PlannedSummary preserves measurements for every stage in an arbitrary
// execution plan while also reporting end-to-end and reference distributions.
type PlannedSummary struct {
	SampleCount                 int                    `json:"sampleCount"`
	StageCount                  int                    `json:"stageCount"`
	Stages                      []PlannedStageSummary  `json:"stages"`
	DistributedEndToEndMicros   Distribution           `json:"distributedEndToEndMicros"`
	SamplingMicros              Distribution           `json:"samplingMicros"`
	TokenLatencyMicros          Distribution           `json:"tokenLatencyMicros"`
	ReferenceWallMicros         Distribution           `json:"referenceWallMicros"`
	ReferenceComputeMicros      Distribution           `json:"referenceComputeMicros"`
	ReferenceSamplingMicros     Distribution           `json:"referenceSamplingMicros"`
	ReferenceTokenLatencyMicros Distribution           `json:"referenceTokenLatencyMicros"`
	TokensPerSecond             float64                `json:"tokensPerSecond"`
	ReferenceTokensPerSecond    float64                `json:"referenceTokensPerSecond"`
	MaxReferenceKVCacheBytes    int                    `json:"maxReferenceKVCacheBytes"`
	ReferenceMemoryHighWater    workerproc.StageMemory `json:"referenceMemoryHighWater"`
}

// PlannedStageSummary is one stable stage identity plus distributions and
// high-water marks collected across all prefill/decode traversals.
type PlannedStageSummary struct {
	Index                       int                       `json:"index"`
	Stage                       generation.ExecutionStage `json:"stage"`
	SampleCount                 int                       `json:"sampleCount"`
	WallMicros                  Distribution              `json:"wallMicros"`
	ComputeMicros               Distribution              `json:"computeMicros"`
	OverheadMicros              Distribution              `json:"overheadMicros"`
	RequestSerializationMicros  Distribution              `json:"requestSerializationMicros"`
	ResponseSerializationMicros Distribution              `json:"responseSerializationMicros"`
	InputTensorBytes            ByteDistribution          `json:"inputTensorBytes"`
	InputWireBytes              ByteDistribution          `json:"inputWireBytes"`
	ResponseTensorBytes         ByteDistribution          `json:"responseTensorBytes"`
	ResponseWireBytes           ByteDistribution          `json:"responseWireBytes"`
	MaxKVCacheBytes             int                       `json:"maxKVCacheBytes"`
	MemoryHighWater             workerproc.StageMemory    `json:"memoryHighWater"`
}

// SummarizePlanned validates that observations describe one stable plan shape
// and aggregates each stage independently instead of folding it into fixed
// producer/consumer fields.
func SummarizePlanned(samples []generation.PlannedStageSample) (PlannedSummary, error) {
	summary := PlannedSummary{SampleCount: len(samples)}
	if len(samples) == 0 {
		return summary, nil
	}
	stageCount := len(samples[0].Stages)
	if stageCount == 0 {
		return PlannedSummary{}, errors.New("planned benchmark sample has no stages")
	}
	summary.StageCount = stageCount
	summary.Stages = make([]PlannedStageSummary, stageCount)
	for index, execution := range samples[0].Stages {
		if execution.Index != index {
			return PlannedSummary{}, fmt.Errorf(
				"planned benchmark first sample stage %d has index %d", index, execution.Index,
			)
		}
		summary.Stages[index] = PlannedStageSummary{
			Index: index, Stage: execution.Stage, SampleCount: len(samples),
		}
	}
	for sampleIndex, sample := range samples {
		if len(sample.Stages) != stageCount {
			return PlannedSummary{}, fmt.Errorf(
				"planned benchmark sample %d has %d stages; expected %d",
				sampleIndex, len(sample.Stages), stageCount,
			)
		}
		for stageIndex, execution := range sample.Stages {
			if execution.Index != stageIndex || execution.Stage != summary.Stages[stageIndex].Stage {
				return PlannedSummary{}, fmt.Errorf(
					"planned benchmark sample %d stage %d changed identity",
					sampleIndex, stageIndex,
				)
			}
		}
		summary.MaxReferenceKVCacheBytes = max(
			summary.MaxReferenceKVCacheBytes, sample.ReferenceKVCacheBytes,
		)
		mergeMemoryHighWater(&summary.ReferenceMemoryHighWater, sample.ReferenceMemory)
	}

	summary.DistributedEndToEndMicros = plannedMicros(samples, func(sample generation.PlannedStageSample) int64 {
		return sample.DistributedEndToEndMicros
	})
	summary.SamplingMicros = plannedMicros(samples, func(sample generation.PlannedStageSample) int64 {
		return sample.SamplingMicros
	})
	summary.TokenLatencyMicros = plannedMicros(samples, func(sample generation.PlannedStageSample) int64 {
		return sample.TokenLatencyMicros
	})
	summary.ReferenceWallMicros = plannedMicros(samples, func(sample generation.PlannedStageSample) int64 {
		return sample.ReferenceWallMicros
	})
	summary.ReferenceComputeMicros = plannedMicros(samples, func(sample generation.PlannedStageSample) int64 {
		return int64(sample.ReferenceComputeMicros)
	})
	summary.ReferenceSamplingMicros = plannedMicros(samples, func(sample generation.PlannedStageSample) int64 {
		return sample.ReferenceSamplingMicros
	})
	summary.ReferenceTokenLatencyMicros = plannedMicros(samples, func(sample generation.PlannedStageSample) int64 {
		return sample.ReferenceTokenLatencyMicros
	})
	summary.TokensPerSecond = plannedThroughput(samples, func(sample generation.PlannedStageSample) int64 {
		return sample.TokenLatencyMicros
	})
	summary.ReferenceTokensPerSecond = plannedThroughput(samples, func(sample generation.PlannedStageSample) int64 {
		return sample.ReferenceTokenLatencyMicros
	})
	for stageIndex := range summary.Stages {
		stage := &summary.Stages[stageIndex]
		stage.WallMicros = plannedStageMicros(samples, stageIndex, func(execution generation.StageExecution) int64 {
			return execution.WallMicros
		})
		stage.ComputeMicros = plannedStageMicros(samples, stageIndex, func(execution generation.StageExecution) int64 {
			return int64(execution.ComputeMicros)
		})
		stage.OverheadMicros = plannedStageMicros(samples, stageIndex, func(execution generation.StageExecution) int64 {
			return execution.OverheadMicros
		})
		stage.RequestSerializationMicros = plannedStageMicros(samples, stageIndex, func(execution generation.StageExecution) int64 {
			return execution.RequestSerializationMicros
		})
		stage.ResponseSerializationMicros = plannedStageMicros(samples, stageIndex, func(execution generation.StageExecution) int64 {
			return execution.ResponseSerializationMicros
		})
		stage.InputTensorBytes = plannedStageBytes(samples, stageIndex, func(execution generation.StageExecution) int64 {
			return int64(execution.InputTensorBytes)
		})
		stage.InputWireBytes = plannedStageBytes(samples, stageIndex, func(execution generation.StageExecution) int64 {
			return int64(execution.InputWireBytes)
		})
		stage.ResponseTensorBytes = plannedStageBytes(samples, stageIndex, func(execution generation.StageExecution) int64 {
			return int64(execution.ResponseTensorBytes)
		})
		stage.ResponseWireBytes = plannedStageBytes(samples, stageIndex, func(execution generation.StageExecution) int64 {
			return int64(execution.ResponseWireBytes)
		})
		for _, sample := range samples {
			execution := sample.Stages[stageIndex]
			stage.MaxKVCacheBytes = max(stage.MaxKVCacheBytes, execution.KVCacheBytes)
			mergeMemoryHighWater(&stage.MemoryHighWater, execution.Memory)
		}
	}
	return summary, nil
}

func plannedThroughput(
	samples []generation.PlannedStageSample,
	duration func(generation.PlannedStageSample) int64,
) float64 {
	var totalMicros int64
	for _, sample := range samples {
		totalMicros += duration(sample)
	}
	if totalMicros <= 0 {
		return 0
	}
	return float64(len(samples)) * 1_000_000 / float64(totalMicros)
}

func plannedMicros(
	samples []generation.PlannedStageSample,
	value func(generation.PlannedStageSample) int64,
) Distribution {
	values := make([]int64, len(samples))
	for index, sample := range samples {
		values[index] = value(sample)
	}
	return summarizeMicros(values)
}

func plannedStageMicros(
	samples []generation.PlannedStageSample,
	stageIndex int,
	value func(generation.StageExecution) int64,
) Distribution {
	values := make([]int64, len(samples))
	for index, sample := range samples {
		values[index] = value(sample.Stages[stageIndex])
	}
	return summarizeMicros(values)
}

func plannedStageBytes(
	samples []generation.PlannedStageSample,
	stageIndex int,
	value func(generation.StageExecution) int64,
) ByteDistribution {
	values := make([]int64, len(samples))
	for index, sample := range samples {
		values[index] = value(sample.Stages[stageIndex])
	}
	return summarizeBytes(values)
}

func summarizeMicros(values []int64) Distribution {
	if len(values) == 0 {
		return Distribution{}
	}
	values = append([]int64(nil), values...)
	var sum int64
	for _, value := range values {
		sum += value
	}
	sort.Slice(values, func(left, right int) bool { return values[left] < values[right] })
	return Distribution{
		Count: len(values), MinMicros: values[0],
		P50Micros: percentile(values, 0.50), P95Micros: percentile(values, 0.95),
		MaxMicros: values[len(values)-1], MeanMicros: float64(sum) / float64(len(values)),
	}
}

func summarizeBytes(values []int64) ByteDistribution {
	if len(values) == 0 {
		return ByteDistribution{}
	}
	values = append([]int64(nil), values...)
	var sum int64
	for _, value := range values {
		sum += value
	}
	sort.Slice(values, func(left, right int) bool { return values[left] < values[right] })
	return ByteDistribution{
		Count: len(values), MinBytes: values[0],
		P50Bytes: percentile(values, 0.50), P95Bytes: percentile(values, 0.95),
		MaxBytes: values[len(values)-1], SumBytes: sum,
	}
}

func mergeMemoryHighWater(highWater *workerproc.StageMemory, value workerproc.StageMemory) {
	highWater.ActiveBytes = max(highWater.ActiveBytes, value.ActiveBytes)
	highWater.CacheBytes = max(highWater.CacheBytes, value.CacheBytes)
	highWater.PeakBytes = max(highWater.PeakBytes, value.PeakBytes)
	highWater.ProcessPhysicalBytes = max(highWater.ProcessPhysicalBytes, value.ProcessPhysicalBytes)
	highWater.ProcessPeakPhysicalBytes = max(
		highWater.ProcessPeakPhysicalBytes, value.ProcessPeakPhysicalBytes,
	)
}

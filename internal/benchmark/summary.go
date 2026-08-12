// Package benchmark aggregates generation-stage observations into stable,
// machine-readable benchmark summaries.
package benchmark

import (
	"math"
	"sort"

	"github.com/fijimunkii/mlx-swarm/internal/generation"
)

// Distribution uses nearest-rank percentiles. Integer microseconds keep the
// output reproducible and avoid implying more timer precision than was read.
type Distribution struct {
	Count      int     `json:"count"`
	MinMicros  int64   `json:"minMicros"`
	P50Micros  int64   `json:"p50Micros"`
	P95Micros  int64   `json:"p95Micros"`
	MaxMicros  int64   `json:"maxMicros"`
	MeanMicros float64 `json:"meanMicros"`
}

type ByteDistribution struct {
	Count    int   `json:"count"`
	MinBytes int64 `json:"minBytes"`
	P50Bytes int64 `json:"p50Bytes"`
	P95Bytes int64 `json:"p95Bytes"`
	MaxBytes int64 `json:"maxBytes"`
	SumBytes int64 `json:"sumBytes"`
}

type StageSummary struct {
	SampleCount                         int              `json:"sampleCount"`
	ProducerWallMicros                  Distribution     `json:"producerWallMicros"`
	ProducerComputeMicros               Distribution     `json:"producerComputeMicros"`
	ProducerOverheadMicros              Distribution     `json:"producerOverheadMicros"`
	BoundarySerializationMicros         Distribution     `json:"boundarySerializationMicros"`
	ConsumerResponseSerializationMicros Distribution     `json:"consumerResponseSerializationMicros"`
	ConsumerRoundTripMicros             Distribution     `json:"consumerRoundTripMicros"`
	ConsumerComputeMicros               Distribution     `json:"consumerComputeMicros"`
	TransportOverheadMicros             Distribution     `json:"transportOverheadMicros"`
	DistributedEndToEndMicros           Distribution     `json:"distributedEndToEndMicros"`
	SamplingMicros                      Distribution     `json:"samplingMicros"`
	TokenLatencyMicros                  Distribution     `json:"tokenLatencyMicros"`
	ReferenceWallMicros                 Distribution     `json:"referenceWallMicros"`
	ReferenceComputeMicros              Distribution     `json:"referenceComputeMicros"`
	ReferenceSamplingMicros             Distribution     `json:"referenceSamplingMicros"`
	ReferenceTokenLatencyMicros         Distribution     `json:"referenceTokenLatencyMicros"`
	TokensPerSecond                     float64          `json:"tokensPerSecond"`
	ReferenceTokensPerSecond            float64          `json:"referenceTokensPerSecond"`
	BoundaryTensorBytes                 ByteDistribution `json:"boundaryTensorBytes"`
	BoundaryWireBytes                   ByteDistribution `json:"boundaryWireBytes"`
	ConsumerResponseTensorBytes         ByteDistribution `json:"consumerResponseTensorBytes"`
	ConsumerResponseWireBytes           ByteDistribution `json:"consumerResponseWireBytes"`
	MaxProducerKVCacheBytes             int              `json:"maxProducerKVCacheBytes"`
	MaxConsumerKVCacheBytes             int              `json:"maxConsumerKVCacheBytes"`
	MaxReferenceKVCacheBytes            int              `json:"maxReferenceKVCacheBytes"`
	ProducerPeakMemoryBytes             int              `json:"producerPeakMemoryBytes"`
	ConsumerPeakMemoryBytes             int              `json:"consumerPeakMemoryBytes"`
	ReferencePeakMemoryBytes            int              `json:"referencePeakMemoryBytes"`
}

func Summarize(samples []generation.StageSample) StageSummary {
	summary := StageSummary{SampleCount: len(samples)}
	summary.ProducerWallMicros = micros(samples, func(sample generation.StageSample) int64 {
		return sample.ProducerWallMicros
	})
	summary.ProducerComputeMicros = micros(samples, func(sample generation.StageSample) int64 {
		return int64(sample.ProducerComputeMicros)
	})
	summary.ProducerOverheadMicros = micros(samples, func(sample generation.StageSample) int64 {
		return sample.ProducerOverheadMicros
	})
	summary.BoundarySerializationMicros = micros(samples, func(sample generation.StageSample) int64 {
		return sample.BoundarySerializationMicros
	})
	summary.ConsumerResponseSerializationMicros = micros(samples, func(sample generation.StageSample) int64 {
		return sample.ConsumerResponseSerializationMicros
	})
	summary.ConsumerRoundTripMicros = micros(samples, func(sample generation.StageSample) int64 {
		return sample.ConsumerRoundTripMicros
	})
	summary.ConsumerComputeMicros = micros(samples, func(sample generation.StageSample) int64 {
		return int64(sample.ConsumerComputeMicros)
	})
	summary.TransportOverheadMicros = micros(samples, func(sample generation.StageSample) int64 {
		return sample.TransportOverheadMicros
	})
	summary.DistributedEndToEndMicros = micros(samples, func(sample generation.StageSample) int64 {
		return sample.DistributedEndToEndMicros
	})
	summary.SamplingMicros = micros(samples, func(sample generation.StageSample) int64 {
		return sample.SamplingMicros
	})
	summary.TokenLatencyMicros = micros(samples, func(sample generation.StageSample) int64 {
		return sample.TokenLatencyMicros
	})
	summary.ReferenceWallMicros = micros(samples, func(sample generation.StageSample) int64 {
		return sample.ReferenceWallMicros
	})
	summary.ReferenceComputeMicros = micros(samples, func(sample generation.StageSample) int64 {
		return int64(sample.ReferenceComputeMicros)
	})
	summary.ReferenceSamplingMicros = micros(samples, func(sample generation.StageSample) int64 {
		return sample.ReferenceSamplingMicros
	})
	summary.ReferenceTokenLatencyMicros = micros(samples, func(sample generation.StageSample) int64 {
		return sample.ReferenceTokenLatencyMicros
	})
	summary.TokensPerSecond = throughput(samples, func(sample generation.StageSample) int64 {
		return sample.TokenLatencyMicros
	})
	summary.ReferenceTokensPerSecond = throughput(samples, func(sample generation.StageSample) int64 {
		return sample.ReferenceTokenLatencyMicros
	})
	summary.BoundaryTensorBytes = bytes(samples, func(sample generation.StageSample) int64 {
		return int64(sample.BoundaryTensorBytes)
	})
	summary.BoundaryWireBytes = bytes(samples, func(sample generation.StageSample) int64 {
		return int64(sample.BoundaryWireBytes)
	})
	summary.ConsumerResponseTensorBytes = bytes(samples, func(sample generation.StageSample) int64 {
		return int64(sample.ConsumerResponseTensorBytes)
	})
	summary.ConsumerResponseWireBytes = bytes(samples, func(sample generation.StageSample) int64 {
		return int64(sample.ConsumerResponseWireBytes)
	})
	for _, sample := range samples {
		summary.MaxProducerKVCacheBytes = max(summary.MaxProducerKVCacheBytes, sample.ProducerKVCacheBytes)
		summary.MaxConsumerKVCacheBytes = max(summary.MaxConsumerKVCacheBytes, sample.ConsumerKVCacheBytes)
		summary.MaxReferenceKVCacheBytes = max(summary.MaxReferenceKVCacheBytes, sample.ReferenceKVCacheBytes)
		summary.ProducerPeakMemoryBytes = max(summary.ProducerPeakMemoryBytes, sample.ProducerMemory.PeakBytes)
		summary.ConsumerPeakMemoryBytes = max(summary.ConsumerPeakMemoryBytes, sample.ConsumerMemory.PeakBytes)
		summary.ReferencePeakMemoryBytes = max(summary.ReferencePeakMemoryBytes, sample.ReferenceMemory.PeakBytes)
	}
	return summary
}

func micros(
	samples []generation.StageSample,
	value func(generation.StageSample) int64,
) Distribution {
	values := make([]int64, len(samples))
	var sum int64
	for index, sample := range samples {
		values[index] = value(sample)
		sum += values[index]
	}
	if len(values) == 0 {
		return Distribution{}
	}
	sort.Slice(values, func(left, right int) bool { return values[left] < values[right] })
	return Distribution{
		Count: len(values), MinMicros: values[0],
		P50Micros: percentile(values, 0.50), P95Micros: percentile(values, 0.95),
		MaxMicros: values[len(values)-1], MeanMicros: float64(sum) / float64(len(values)),
	}
}

func bytes(
	samples []generation.StageSample,
	value func(generation.StageSample) int64,
) ByteDistribution {
	values := make([]int64, len(samples))
	var sum int64
	for index, sample := range samples {
		values[index] = value(sample)
		sum += values[index]
	}
	if len(values) == 0 {
		return ByteDistribution{}
	}
	sort.Slice(values, func(left, right int) bool { return values[left] < values[right] })
	return ByteDistribution{
		Count: len(values), MinBytes: values[0],
		P50Bytes: percentile(values, 0.50), P95Bytes: percentile(values, 0.95),
		MaxBytes: values[len(values)-1], SumBytes: sum,
	}
}

func percentile(sorted []int64, quantile float64) int64 {
	index := int(math.Ceil(quantile*float64(len(sorted)))) - 1
	if index < 0 {
		index = 0
	}
	return sorted[index]
}

func throughput(
	samples []generation.StageSample,
	duration func(generation.StageSample) int64,
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

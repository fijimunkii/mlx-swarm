package benchmark

import (
	"math"
	"testing"

	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

func TestSummarizeUsesNearestRankPercentilesAndTotals(t *testing.T) {
	samples := make([]generation.StageSample, 5)
	for index, latency := range []int64{50, 10, 40, 20, 30} {
		samples[index] = generation.StageSample{
			TokenLatencyMicros: latency, ReferenceTokenLatencyMicros: latency * 2,
			ProducerWallMicros: latency, BoundaryTensorBytes: 10,
			BoundaryWireBytes: 20, ConsumerResponseTensorBytes: 30,
			ConsumerResponseWireBytes: 50, ProducerKVCacheBytes: index,
			ConsumerKVCacheBytes: index * 2, ReferenceKVCacheBytes: index * 3,
			ProducerMemory:  workerproc.StageMemory{PeakBytes: index * 4},
			ConsumerMemory:  workerproc.StageMemory{PeakBytes: index * 5},
			ReferenceMemory: workerproc.StageMemory{PeakBytes: index * 6},
		}
	}

	summary := Summarize(samples)
	if summary.TokenLatencyMicros.P50Micros != 30 || summary.TokenLatencyMicros.P95Micros != 50 {
		t.Fatalf("unexpected percentiles: %+v", summary.TokenLatencyMicros)
	}
	if summary.BoundaryTensorBytes.SumBytes != 50 || summary.BoundaryWireBytes.SumBytes != 100 {
		t.Fatalf("unexpected byte totals: %+v %+v", summary.BoundaryTensorBytes, summary.BoundaryWireBytes)
	}
	if summary.ConsumerResponseTensorBytes.SumBytes != 150 ||
		summary.ConsumerResponseWireBytes.SumBytes != 250 {
		t.Fatalf(
			"unexpected response byte totals: %+v %+v",
			summary.ConsumerResponseTensorBytes, summary.ConsumerResponseWireBytes,
		)
	}
	if math.Abs(summary.TokensPerSecond-(5_000_000.0/150.0)) > 0.001 {
		t.Fatalf("tokens per second = %f", summary.TokensPerSecond)
	}
	if summary.MaxProducerKVCacheBytes != 4 || summary.MaxConsumerKVCacheBytes != 8 ||
		summary.MaxReferenceKVCacheBytes != 12 || summary.ReferencePeakMemoryBytes != 24 {
		t.Fatalf("unexpected high-water marks: %+v", summary)
	}
}

func TestSummarizeEmpty(t *testing.T) {
	if summary := Summarize(nil); summary.SampleCount != 0 || summary.TokensPerSecond != 0 {
		t.Fatalf("unexpected empty summary: %+v", summary)
	}
}

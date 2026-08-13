package benchmark

import (
	"math"
	"strings"
	"testing"

	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

func TestSummarizePlannedPreservesEveryStage(t *testing.T) {
	stage := func(index int) generation.ExecutionStage {
		return generation.ExecutionStage{
			Name: "stage", TargetID: "target", ShardID: "shard",
			LayerStart: index, LayerEnd: index + 1,
		}
	}
	samples := make([]generation.PlannedStageSample, 2)
	for sampleIndex := range samples {
		samples[sampleIndex] = generation.PlannedStageSample{
			DistributedEndToEndMicros: int64(30 + sampleIndex),
			TokenLatencyMicros:        int64(40 + sampleIndex),
			ReferenceKVCacheBytes:     90 + sampleIndex,
			ReferenceMemory: workerproc.StageMemory{
				ProcessPhysicalBytes: uint64(100 + sampleIndex),
			},
			Stages: []generation.StageExecution{
				{
					Index: 0, Stage: stage(0), WallMicros: int64(10 + sampleIndex),
					ComputeMicros: 7, InputTensorBytes: 11, InputWireBytes: 13,
					ResponseTensorBytes: 17, ResponseWireBytes: 19,
					KVCacheBytes: 23 + sampleIndex,
					Memory:       workerproc.StageMemory{PeakBytes: 29 + sampleIndex},
				},
				{
					Index: 1, Stage: stage(1), WallMicros: int64(20 + sampleIndex),
					ComputeMicros: 14, InputTensorBytes: 31, InputWireBytes: 37,
					ResponseTensorBytes: 0, ResponseWireBytes: 41,
					KVCacheBytes: 43 + sampleIndex,
					Memory:       workerproc.StageMemory{ActiveBytes: 47 + sampleIndex},
				},
			},
		}
	}

	summary, err := SummarizePlanned(samples)
	if err != nil {
		t.Fatal(err)
	}
	if summary.SampleCount != 2 || summary.StageCount != 2 || len(summary.Stages) != 2 {
		t.Fatalf("unexpected plan summary shape: %+v", summary)
	}
	if want := 2_000_000.0 / 81.0; math.Abs(summary.TokensPerSecond-want) > 0.001 {
		t.Fatalf("planned throughput = %f", summary.TokensPerSecond)
	}
	if summary.Stages[0].WallMicros.P50Micros != 10 ||
		summary.Stages[0].ResponseWireBytes.SumBytes != 38 ||
		summary.Stages[0].MaxKVCacheBytes != 24 ||
		summary.Stages[0].MemoryHighWater.PeakBytes != 30 {
		t.Fatalf("unexpected first-stage summary: %+v", summary.Stages[0])
	}
	if summary.Stages[1].ResponseTensorBytes.MaxBytes != 0 ||
		summary.Stages[1].MaxKVCacheBytes != 44 ||
		summary.Stages[1].MemoryHighWater.ActiveBytes != 48 {
		t.Fatalf("unexpected terminal-stage summary: %+v", summary.Stages[1])
	}
	if summary.MaxReferenceKVCacheBytes != 91 ||
		summary.ReferenceMemoryHighWater.ProcessPhysicalBytes != 101 {
		t.Fatalf("unexpected reference summary: %+v", summary)
	}
}

func TestSummarizePlannedRejectsChangingPlanShape(t *testing.T) {
	stage := generation.ExecutionStage{Name: "stage", TargetID: "target", ShardID: "shard"}
	_, err := SummarizePlanned([]generation.PlannedStageSample{
		{Stages: []generation.StageExecution{{Index: 0, Stage: stage}}},
		{Stages: []generation.StageExecution{{Index: 0, Stage: stage}, {Index: 1, Stage: stage}}},
	})
	if err == nil || !strings.Contains(err.Error(), "expected 1") {
		t.Fatalf("error = %v, want stable-shape rejection", err)
	}
}

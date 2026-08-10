package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

type shardResult struct {
	Model              string `json:"model"`
	Layers             int    `json:"layers"`
	SplitLayer         int    `json:"splitLayer"`
	BoundaryBytes      int    `json:"boundaryBytes"`
	BoundaryDType      string `json:"boundaryDType"`
	OutputShape        []int  `json:"outputShape"`
	MatchesSingleRange bool   `json:"matchesSingleRange"`
}

type runSummary struct {
	Worker              string  `json:"worker"`
	BoundaryWireBytes   int     `json:"boundaryWireBytes"`
	BoundaryTensorBytes int     `json:"boundaryTensorBytes"`
	ProducerMillis      float64 `json:"producerMillis"`
	ConsumerMillis      float64 `json:"consumerMillis"`
	MatchesSingleRange  bool    `json:"matchesSingleRange"`
	Model               string  `json:"model"`
	Layers              int     `json:"layers"`
	SplitLayer          int     `json:"splitLayer"`
	BoundaryDType       string  `json:"boundaryDType"`
}

func main() {
	worker := flag.String("worker", workerproc.DefaultPath(), "path to the built MLXWorker executable")
	flag.Parse()

	client := workerproc.Client{Path: *worker}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	producer, err := client.Run(ctx, []string{"shard-produce-stdio"}, nil)
	if err != nil {
		fatalf("producer failed: %v", err)
	}

	consumer, err := client.Run(ctx, []string{"shard-finish-stdio"}, producer.Output)
	if err != nil {
		fatalf("consumer failed: %v", err)
	}

	var result shardResult
	if err := json.Unmarshal(bytes.TrimSpace(consumer.Output), &result); err != nil {
		fatalf("decode consumer result: %v; output=%q", err, consumer.Output)
	}
	if !result.MatchesSingleRange {
		fatalf("distributed result did not match single-range reference")
	}

	summary := runSummary{
		Worker:              *worker,
		BoundaryWireBytes:   len(producer.Output),
		BoundaryTensorBytes: result.BoundaryBytes,
		ProducerMillis:      millis(producer.Duration),
		ConsumerMillis:      millis(consumer.Duration),
		MatchesSingleRange:  result.MatchesSingleRange,
		Model:               result.Model,
		Layers:              result.Layers,
		SplitLayer:          result.SplitLayer,
		BoundaryDType:       result.BoundaryDType,
	}

	encoded, err := json.Marshal(summary)
	if err != nil {
		fatalf("encode summary: %v", err)
	}
	fmt.Println(string(encoded))
}

func millis(d time.Duration) float64 {
	return float64(d.Microseconds()) / 1000.0
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}

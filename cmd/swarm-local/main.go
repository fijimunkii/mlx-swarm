package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"time"
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
	Worker             string  `json:"worker"`
	BoundaryWireBytes  int     `json:"boundaryWireBytes"`
	BoundaryTensorBytes int    `json:"boundaryTensorBytes"`
	ProducerMillis     float64 `json:"producerMillis"`
	ConsumerMillis     float64 `json:"consumerMillis"`
	MatchesSingleRange bool    `json:"matchesSingleRange"`
	Model              string  `json:"model"`
	Layers             int     `json:"layers"`
	SplitLayer         int     `json:"splitLayer"`
	BoundaryDType      string  `json:"boundaryDType"`
}

func main() {
	worker := flag.String("worker", defaultWorkerPath(), "path to the built MLXWorker executable")
	flag.Parse()

	payload, producerDuration, err := runWorker(*worker, []string{"shard-produce-stdio"}, nil)
	if err != nil {
		fatalf("producer failed: %v", err)
	}

	output, consumerDuration, err := runWorker(*worker, []string{"shard-finish-stdio"}, payload)
	if err != nil {
		fatalf("consumer failed: %v", err)
	}

	var result shardResult
	if err := json.Unmarshal(bytes.TrimSpace(output), &result); err != nil {
		fatalf("decode consumer result: %v; output=%q", err, output)
	}
	if !result.MatchesSingleRange {
		fatalf("distributed result did not match single-range reference")
	}

	summary := runSummary{
		Worker:              *worker,
		BoundaryWireBytes:   len(payload),
		BoundaryTensorBytes: result.BoundaryBytes,
		ProducerMillis:      millis(producerDuration),
		ConsumerMillis:      millis(consumerDuration),
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

func defaultWorkerPath() string {
	if path := os.Getenv("MLX_SWARM_WORKER"); path != "" {
		return path
	}
	return "worker/mlx/.build/debug/MLXWorker"
}

func runWorker(worker string, args []string, input []byte) ([]byte, time.Duration, error) {
	cmd := exec.Command(worker, args...)
	if input != nil {
		cmd.Stdin = bytes.NewReader(input)
	}
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	start := time.Now()
	err := cmd.Run()
	duration := time.Since(start)
	if err != nil {
		return nil, duration, fmt.Errorf("%w: %s", err, bytes.TrimSpace(stderr.Bytes()))
	}
	return stdout.Bytes(), duration, nil
}

func millis(d time.Duration) float64 {
	return float64(d.Microseconds()) / 1000.0
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}

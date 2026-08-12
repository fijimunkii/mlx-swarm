package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

type stageMemory struct {
	ActiveBytes int `json:"activeBytes"`
	CacheBytes  int `json:"cacheBytes"`
	PeakBytes   int `json:"peakBytes"`
}

type checkpointShardResult struct {
	Model                      string      `json:"model"`
	ModelType                  string      `json:"modelType"`
	Layers                     int         `json:"layers"`
	SplitLayer                 int         `json:"splitLayer"`
	WorkerBudgetBytes          int         `json:"workerBudgetBytes"`
	FullCheckpointAfterForward stageMemory `json:"fullCheckpointAfterForward"`
	FirstStageAfterForward     stageMemory `json:"firstStageAfterForward"`
	SecondStageAfterForward    stageMemory `json:"secondStageAfterForward"`
	BoundaryBytes              int         `json:"boundaryBytes"`
	BoundaryDType              string      `json:"boundaryDType"`
	OutputShape                []int       `json:"outputShape"`
	Rtol                       float64     `json:"rtol"`
	Atol                       float64     `json:"atol"`
	MatchesFullCheckpoint      bool        `json:"matchesFullCheckpoint"`
	PassesMemoryProof          bool        `json:"passesMemoryProof"`
}

type networkSummary struct {
	Peer                    string  `json:"peer"`
	BoundaryWireBytes       int     `json:"boundaryWireBytes"`
	BoundaryTensorBytes     int     `json:"boundaryTensorBytes"`
	ProducerMillis          float64 `json:"producerMillis"`
	NetworkMillis           float64 `json:"networkMillis"`
	RemoteWorkerMicros      string  `json:"remoteWorkerMicros,omitempty"`
	RemoteWorkerMillis      float64 `json:"remoteWorkerMillis"`
	TransportOverheadMillis float64 `json:"transportOverheadMillis"`
	TotalMillis             float64 `json:"totalMillis"`
	WireEffectiveMBps       float64 `json:"wireEffectiveMBps"`
	MatchesFullCheckpoint   bool    `json:"matchesFullCheckpoint"`
	PassesMemoryProof       bool    `json:"passesMemoryProof"`
	Model                   string  `json:"model"`
	ModelType               string  `json:"modelType"`
	Layers                  int     `json:"layers"`
	SplitLayer              int     `json:"splitLayer"`
	BoundaryDType           string  `json:"boundaryDType"`
	OutputShape             []int   `json:"outputShape"`
	WorkerBudgetBytes       int     `json:"workerBudgetBytes"`
	FullCheckpointPeakBytes int     `json:"fullCheckpointPeakBytes"`
	FirstStagePeakBytes     int     `json:"firstStagePeakBytes"`
	SecondStagePeakBytes    int     `json:"secondStagePeakBytes"`
	Rtol                    float64 `json:"rtol"`
	Atol                    float64 `json:"atol"`
}

func main() {
	peer := flag.String("peer", "http://127.0.0.1:8080", "base URL of the remote swarmd")
	worker := flag.String("worker", workerproc.DefaultPath(), "path to the local built MLXWorker executable")
	timeout := flag.Duration("timeout", 2*time.Minute, "overall experiment timeout")
	flag.Parse()

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	local := workerproc.Client{Path: *worker}
	producer, err := local.Run(ctx, []string{"checkpoint-shard-produce-stdio"}, nil)
	if err != nil {
		fatalf("local producer failed: %v", err)
	}

	endpoint := strings.TrimRight(*peer, "/") + "/v1/debug/checkpoint-shard/finish"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(producer.Output))
	if err != nil {
		fatalf("create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/vnd.mlx-swarm.checkpoint-boundary+json")

	start := time.Now()
	resp, err := (&http.Client{}).Do(req)
	networkDuration := time.Since(start)
	if err != nil {
		fatalf("request remote worker: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		fatalf("read remote response: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		fatalf("remote returned %s: %s", resp.Status, bytes.TrimSpace(body))
	}

	var result checkpointShardResult
	if err := json.Unmarshal(bytes.TrimSpace(body), &result); err != nil {
		fatalf("decode remote result: %v; output=%q", err, body)
	}
	if !result.MatchesFullCheckpoint {
		fatalf("remote checkpoint shards did not match full-checkpoint logits")
	}
	if !result.PassesMemoryProof {
		fatalf("remote checkpoint shards did not pass the configured memory-budget proof")
	}

	producerMillis := millis(producer.Duration)
	networkMillis := millis(networkDuration)
	remoteWorkerMicros := resp.Header.Get("X-MLX-Swarm-Worker-Micros")
	remoteWorkerMillis := microsHeaderMillis(remoteWorkerMicros)
	transportOverheadMillis := networkMillis - remoteWorkerMillis
	if transportOverheadMillis < 0 {
		transportOverheadMillis = 0
	}

	summary := networkSummary{
		Peer:                    *peer,
		BoundaryWireBytes:       len(producer.Output),
		BoundaryTensorBytes:     result.BoundaryBytes,
		ProducerMillis:          producerMillis,
		NetworkMillis:           networkMillis,
		RemoteWorkerMicros:      remoteWorkerMicros,
		RemoteWorkerMillis:      remoteWorkerMillis,
		TransportOverheadMillis: transportOverheadMillis,
		TotalMillis:             producerMillis + networkMillis,
		WireEffectiveMBps:       effectiveMBps(len(producer.Output), networkDuration),
		MatchesFullCheckpoint:   result.MatchesFullCheckpoint,
		PassesMemoryProof:       result.PassesMemoryProof,
		Model:                   result.Model,
		ModelType:               result.ModelType,
		Layers:                  result.Layers,
		SplitLayer:              result.SplitLayer,
		BoundaryDType:           result.BoundaryDType,
		OutputShape:             result.OutputShape,
		WorkerBudgetBytes:       result.WorkerBudgetBytes,
		FullCheckpointPeakBytes: result.FullCheckpointAfterForward.PeakBytes,
		FirstStagePeakBytes:     result.FirstStageAfterForward.PeakBytes,
		SecondStagePeakBytes:    result.SecondStageAfterForward.PeakBytes,
		Rtol:                    result.Rtol,
		Atol:                    result.Atol,
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

func microsHeaderMillis(value string) float64 {
	if value == "" {
		return 0
	}
	micros, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return 0
	}
	return float64(micros) / 1000.0
}

func effectiveMBps(bytes int, d time.Duration) float64 {
	seconds := d.Seconds()
	if seconds <= 0 {
		return 0
	}
	return (float64(bytes) / 1_000_000.0) / seconds
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}

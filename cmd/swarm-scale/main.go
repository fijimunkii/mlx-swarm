package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/pooledproof"
	"github.com/fijimunkii/mlx-swarm/internal/scaleproof"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

type nodeFlags []string

func (values *nodeFlags) String() string { return strings.Join(*values, ",") }
func (values *nodeFlags) Set(value string) error {
	*values = append(*values, value)
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	var nodeValues nodeFlags
	flag.Var(&nodeValues, "node", "stable worker ID and swarmd URL as id=url; repeat exactly five times")
	referencePath := flag.String("reference", "testdata/pooled-memory/gemma-3-12b-it-4bit.json", "checked-in deterministic 12B reference")
	smallModel := flag.String("small-model", scaleproof.DefaultSmallModelID, "small model used for logit correctness and scaling")
	prompt := flag.String("prompt", scaleproof.DefaultPrompt, "small-model correctness prompt")
	tokens := flag.Int("tokens", scaleproof.DefaultTokenCount, "small-model generated token count")
	coordinatorID := flag.String("coordinator-id", "linux-coordinator", "stable coordinator identity recorded in evidence")
	runID := flag.String("run-id", "local", "proof/run identity recorded in evidence")
	inventoryRevision := flag.String("inventory-revision", "five-mac-static", "explicit inventory revision pinned into plans")
	controlURL := flag.String("control", "", "swarm-control URL containing the real worker membership")
	syntheticPeers := flag.Int("synthetic-peers", scaleproof.DefaultSyntheticPeerCount, "incompatible synthetic Linux peers added to the hybrid inventory")
	edgeReserve := flag.Int("edge-reserve-layers", 2, "virtual transformer layers reserved for each 12B edge owner")
	memoryThreshold := flag.Int("memory-threshold-bytes", pooledproof.DefaultWorkerMemoryThreshold, "required MLX memory threshold on every worker")
	rtol := flag.Float64("rtol", 1e-4, "relative logit tolerance")
	atol := flag.Float64("atol", 1e-4, "absolute logit tolerance")
	forwardTimeout := flag.Duration("forward-timeout", 2*time.Minute, "per-stage request timeout")
	timeout := flag.Duration("timeout", 40*time.Minute, "overall proof timeout")
	flag.Parse()

	if len(nodeValues) != scaleproof.RequiredNodeCount {
		return fmt.Errorf("-node must be supplied exactly %d times", scaleproof.RequiredNodeCount)
	}
	if *timeout <= 0 || *forwardTimeout < time.Millisecond {
		return errors.New("timeouts must be positive and per-stage timeout must be at least 1ms")
	}
	if strings.TrimSpace(*controlURL) == "" {
		return errors.New("-control is required for the hybrid membership proof")
	}
	if *syntheticPeers < scaleproof.DefaultSyntheticPeerCount {
		return fmt.Errorf(
			"-synthetic-peers must be at least %d", scaleproof.DefaultSyntheticPeerCount,
		)
	}
	reference, err := pooledproof.LoadReference(*referencePath)
	if err != nil {
		return err
	}
	signalContext, stopSignals := signal.NotifyContext(
		context.Background(), os.Interrupt, syscall.SIGTERM,
	)
	defer stopSignals()
	ctx, cancel := context.WithTimeout(signalContext, *timeout)
	defer cancel()

	httpClient := &http.Client{Transport: http.DefaultTransport}
	nodes := make([]scaleproof.Node, 0, len(nodeValues))
	for _, value := range nodeValues {
		id, endpoint, ok := strings.Cut(value, "=")
		id, endpoint = strings.TrimSpace(id), strings.TrimRight(strings.TrimSpace(endpoint), "/")
		if !ok || id == "" || endpoint == "" {
			return fmt.Errorf("invalid -node %q; want id=url", value)
		}
		probeMicros, err := probe(ctx, httpClient, endpoint)
		if err != nil {
			return fmt.Errorf("node %s health probe: %w", id, err)
		}
		capabilities, err := pooledproof.FetchRemoteCapabilities(ctx, httpClient, endpoint)
		if err != nil {
			return fmt.Errorf("node %s capabilities: %w", id, err)
		}
		target, err := workerproc.OpenPersistentTargetWithHTTPClient("", endpoint, httpClient)
		if err != nil {
			return fmt.Errorf("node %s: %w", id, err)
		}
		nodes = append(nodes, scaleproof.Node{
			ID: id, Endpoint: endpoint, Capabilities: capabilities,
			ProbeMicros: probeMicros, Caller: target.Caller,
		})
	}

	result, proofErr := scaleproof.Run(ctx, scaleproof.RunConfig{
		Coordinator: scaleproof.CoordinatorEvidence{
			ID: *coordinatorID, RunID: *runID,
			OS: runtime.GOOS, Architecture: runtime.GOARCH,
		},
		Nodes: nodes, SmallModel: *smallModel, Prompt: *prompt, TokenCount: *tokens,
		Reference: reference, InventoryRevision: *inventoryRevision,
		EdgeReserveLayers:            *edgeReserve,
		ExpectedMemoryThresholdBytes: *memoryThreshold,
		RTol:                         *rtol, ATol: *atol, ForwardTimeout: *forwardTimeout,
		ControlURL: *controlURL, HTTPClient: httpClient, SyntheticPeerCount: *syntheticPeers,
	})
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(result); err != nil {
		return fmt.Errorf("encode scale proof: %w", err)
	}
	return proofErr
}

func probe(ctx context.Context, client *http.Client, endpoint string) (int64, error) {
	probeContext, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	request, err := http.NewRequestWithContext(
		probeContext, http.MethodGet, endpoint+"/healthz", nil,
	)
	if err != nil {
		return 0, err
	}
	started := time.Now()
	response, err := client.Do(request)
	wallMicros := time.Since(started).Microseconds()
	if err != nil {
		return 0, err
	}
	defer response.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 1024))
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return 0, fmt.Errorf("HTTP %d", response.StatusCode)
	}
	return wallMicros, nil
}

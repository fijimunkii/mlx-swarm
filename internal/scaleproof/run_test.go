package scaleproof

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/pooledproof"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

type inertCaller struct{}

func (inertCaller) Call(context.Context, workerproc.PersistentRequest) (workerproc.PersistentResponse, error) {
	return workerproc.PersistentResponse{}, nil
}

func TestValidateNodesRequiresDistinctHosts(t *testing.T) {
	nodes := testNodes()
	nodes[4].Endpoint = nodes[0].Endpoint
	err := validateNodes(nodes)
	if err == nil || !strings.Contains(err.Error(), "duplicated") {
		t.Fatalf("error = %v, want duplicate endpoint rejection", err)
	}
}

func TestNormalizeConfigRequiresFiveNodesAndBoundedRequests(t *testing.T) {
	config := RunConfig{
		Nodes: testNodes()[:4], TokenCount: DefaultTokenCount,
		ExpectedMemoryThresholdBytes: 1, ForwardTimeout: time.Second,
	}
	err := normalizeConfig(&config)
	if err == nil || !strings.Contains(err.Error(), "exactly 5") {
		t.Fatalf("error = %v, want five-node requirement", err)
	}
}

func TestNormalizeSyntheticPeerCountBoundsAllocation(t *testing.T) {
	tests := []struct {
		name      string
		input     int
		want      int
		wantError bool
	}{
		{name: "default", input: 0, want: DefaultSyntheticPeerCount},
		{name: "minimum", input: DefaultSyntheticPeerCount, want: DefaultSyntheticPeerCount},
		{name: "maximum", input: MaxSyntheticPeerCount, want: MaxSyntheticPeerCount},
		{name: "below", input: DefaultSyntheticPeerCount - 1, wantError: true},
		{name: "above", input: MaxSyntheticPeerCount + 1, wantError: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			count := test.input
			err := normalizeSyntheticPeerCount(&count)
			if (err != nil) != test.wantError {
				t.Fatalf("error = %v, wantError = %t", err, test.wantError)
			}
			if !test.wantError && count != test.want {
				t.Fatalf("count = %d, want %d", count, test.want)
			}
		})
	}
}

func testNodes() []Node {
	nodes := make([]Node, RequiredNodeCount)
	for index := range nodes {
		nodes[index] = Node{
			ID: string(rune('a' + index)), Endpoint: "http://node-" + string(rune('a'+index)),
			Caller:       inertCaller{},
			Capabilities: pooledproof.Capabilities{PhysicalMemoryBytes: 1},
		}
	}
	return nodes
}

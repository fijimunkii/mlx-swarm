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

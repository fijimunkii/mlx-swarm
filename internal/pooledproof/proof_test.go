package pooledproof

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

func TestCheckedInReferenceIsValid(t *testing.T) {
	reference, err := LoadReference(filepath.Join(
		"..", "..", "testdata", "pooled-memory", "gemma-3-12b-it-6bit.json",
	))
	if err != nil {
		t.Fatal(err)
	}
	if reference.Model != DefaultModelID {
		t.Fatalf("model = %q, want %q", reference.Model, DefaultModelID)
	}
	const runnerPhysicalBytes = uint64(7516192768)
	if reference.CheckpointBytes <= runnerPhysicalBytes {
		t.Fatalf(
			"checkpoint has %d bytes, want more than a 7 GiB runner's %d",
			reference.CheckpointBytes,
			runnerPhysicalBytes,
		)
	}
	if reference.FullCheckpointMemory.MaxProcessPhysicalBytes <= runnerPhysicalBytes {
		t.Fatalf(
			"full-model process uses %d bytes, want more than a 7 GiB runner's %d",
			reference.FullCheckpointMemory.MaxProcessPhysicalBytes,
			runnerPhysicalBytes,
		)
	}
}

func TestRemoteCapabilities(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/debug/worker/capabilities" {
			t.Fatalf("path = %q", request.URL.Path)
		}
		_, _ = w.Write([]byte(`{
			"runtime":"mlx-swift",
			"device":"gpu",
			"physicalMemoryBytes":7516192768,
			"mlxMemoryLimitBytes":6442450944,
			"mlxCacheLimitBytes":67108864
		}`))
	}))
	defer server.Close()

	capabilities, err := remoteCapabilities(context.Background(), server.Client(), server.URL)
	if err != nil {
		t.Fatal(err)
	}
	state := &workerproc.PersistentWorkerState{
		PhysicalMemoryBytes: capabilities.PhysicalMemoryBytes,
		MLXMemoryLimitBytes: capabilities.MLXMemoryLimitBytes,
		MLXCacheLimitBytes:  capabilities.MLXCacheLimitBytes,
	}
	if !configuredMLXThreshold(capabilities, state, DefaultWorkerMemoryThreshold) {
		t.Fatalf("capabilities did not preserve configured threshold: %+v", capabilities)
	}
}

func TestRunRejectsDirtyWorkersBeforePreparingSession(t *testing.T) {
	reference, err := LoadReference(filepath.Join(
		"..", "..", "testdata", "pooled-memory", "gemma-3-12b-it-6bit.json",
	))
	if err != nil {
		t.Fatal(err)
	}

	dirty := &workerproc.PersistentWorkerState{
		LoadedShards: []workerproc.PersistentShardSnapshot{{ShardID: "already-loaded"}},
	}
	producer, producerCommands := newProofWorkerServer(t, dirty)
	defer producer.Close()
	consumer, consumerCommands := newProofWorkerServer(t, &workerproc.PersistentWorkerState{})
	defer consumer.Close()

	_, err = Run(context.Background(), reference, RunConfig{
		ProducerURL: producer.URL, ConsumerURL: consumer.URL,
		ForwardTimeout: time.Second,
	})
	if err == nil || !strings.Contains(err.Error(), "workers must be clean") {
		t.Fatalf("Run error = %v, want clean-worker rejection", err)
	}
	if got := producerCommands(); len(got) != 1 || got[0] != "state" {
		t.Fatalf("producer commands = %v, want only initial state", got)
	}
	if got := consumerCommands(); len(got) != 1 || got[0] != "state" {
		t.Fatalf("consumer commands = %v, want only initial state", got)
	}
}

func TestRunUsesConfiguredClientAndUnloadsPartialSession(t *testing.T) {
	reference, err := LoadReference(filepath.Join(
		"..", "..", "testdata", "pooled-memory", "gemma-3-12b-it-6bit.json",
	))
	if err != nil {
		t.Fatal(err)
	}

	producer, producerState, producerCommands := newSessionSetupWorkerServer(t, reference, false)
	defer producer.Close()
	consumer, _, _ := newSessionSetupWorkerServer(t, reference, true)
	defer consumer.Close()

	client := &http.Client{Transport: roundTripperFunc(func(request *http.Request) (*http.Response, error) {
		clone := request.Clone(request.Context())
		clone.Header = request.Header.Clone()
		clone.Header.Set("X-Pooled-Proof-Client", "configured")
		return http.DefaultTransport.RoundTrip(clone)
	})}
	_, err = Run(context.Background(), reference, RunConfig{
		ProducerURL: producer.URL, ConsumerURL: consumer.URL,
		ForwardTimeout: time.Second, HTTPClient: client,
	})
	if err == nil || !strings.Contains(err.Error(), "consumer shard") {
		t.Fatalf("Run error = %v, want consumer shard setup failure", err)
	}
	if got := producerState(); len(got.LoadedShards) != 0 {
		t.Fatalf("producer retained shards after rollback: %+v", got.LoadedShards)
	}
	if got := producerCommands(); !containsString(got, "unloadShard") {
		t.Fatalf("producer commands = %v, want unloadShard rollback", got)
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (fn roundTripperFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return fn(request)
}

func newSessionSetupWorkerServer(
	t *testing.T,
	reference Reference,
	failLoad bool,
) (*httptest.Server, func() workerproc.PersistentWorkerState, func() []string) {
	t.Helper()
	var mu sync.Mutex
	state := workerproc.PersistentWorkerState{}
	var commands []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if request.Header.Get("X-Pooled-Proof-Client") != "configured" {
			http.Error(w, "configured client header is required", http.StatusUnauthorized)
			return
		}
		if request.URL.Path == "/v1/debug/worker/capabilities" {
			_, _ = w.Write([]byte(`{
				"runtime":"mlx-swift",
				"device":"gpu",
				"physicalMemoryBytes":7516192768,
				"mlxMemoryLimitBytes":6442450944,
				"mlxCacheLimitBytes":67108864
			}`))
			return
		}
		if request.URL.Path != "/v1/worker/request" {
			http.NotFound(w, request)
			return
		}

		var frame workerproc.PersistentRequest
		if err := json.NewDecoder(request.Body).Decode(&frame); err != nil {
			t.Errorf("decode persistent request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		mu.Lock()
		defer mu.Unlock()
		commands = append(commands, frame.Command)
		response := workerproc.PersistentResponse{RequestID: frame.RequestID, OK: true}
		switch frame.Command {
		case "state":
			snapshot := state
			snapshot.LoadedShards = append([]workerproc.PersistentShardSnapshot(nil), state.LoadedShards...)
			response.Result = &workerproc.PersistentWorkerResult{State: &snapshot}
		case "modelInfo":
			response.Result = &workerproc.PersistentWorkerResult{Model: &workerproc.PersistentModelResult{
				ModelID: reference.Model, ModelType: reference.ModelType, LayerCount: reference.LayerCount,
				CheckpointFingerprint: reference.CheckpointFingerprint, CheckpointBytes: reference.CheckpointBytes,
			}}
		case "loadShard":
			if failLoad {
				response.OK = false
				response.Error = "injected load failure"
				break
			}
			load := frame.LoadShard
			shard := workerproc.PersistentShardSnapshot{
				ShardID: load.ShardID, ModelID: load.ModelID, ModelType: reference.ModelType,
				CheckpointFingerprint: load.CheckpointFingerprint,
				LayerStart:            load.LayerStart, LayerEnd: load.LayerEnd,
				OwnsInput: load.OwnsInput, OwnsOutput: load.OwnsOutput,
			}
			state.LoadedShards = append(state.LoadedShards, shard)
			response.Result = &workerproc.PersistentWorkerResult{Shard: &shard}
		case "unloadShard":
			for index, shard := range state.LoadedShards {
				if shard.ShardID == frame.Shard.ShardID {
					state.LoadedShards = append(state.LoadedShards[:index], state.LoadedShards[index+1:]...)
					break
				}
			}
			snapshot := state
			response.Result = &workerproc.PersistentWorkerResult{State: &snapshot}
		default:
			response.OK = false
			response.Error = "unexpected command " + frame.Command
		}
		_ = json.NewEncoder(w).Encode(response)
	}))
	return server, func() workerproc.PersistentWorkerState {
			mu.Lock()
			defer mu.Unlock()
			snapshot := state
			snapshot.LoadedShards = append([]workerproc.PersistentShardSnapshot(nil), state.LoadedShards...)
			return snapshot
		}, func() []string {
			mu.Lock()
			defer mu.Unlock()
			return append([]string(nil), commands...)
		}
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func newProofWorkerServer(
	t *testing.T,
	state *workerproc.PersistentWorkerState,
) (*httptest.Server, func() []string) {
	t.Helper()
	var mu sync.Mutex
	var commands []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/v1/debug/worker/capabilities":
			_, _ = w.Write([]byte(`{
				"runtime":"mlx-swift",
				"device":"gpu",
				"physicalMemoryBytes":7516192768,
				"mlxMemoryLimitBytes":6442450944,
				"mlxCacheLimitBytes":67108864
			}`))
		case "/v1/worker/request":
			var frame workerproc.PersistentRequest
			if err := json.NewDecoder(request.Body).Decode(&frame); err != nil {
				t.Errorf("decode persistent request: %v", err)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			mu.Lock()
			commands = append(commands, frame.Command)
			mu.Unlock()
			_ = json.NewEncoder(w).Encode(workerproc.PersistentResponse{
				RequestID: frame.RequestID,
				OK:        true,
				Result:    &workerproc.PersistentWorkerResult{State: state},
			})
		default:
			http.NotFound(w, request)
		}
	}))
	return server, func() []string {
		mu.Lock()
		defer mu.Unlock()
		return append([]string(nil), commands...)
	}
}

func TestMemoryEvidenceUsesMLXAndProcessPeaks(t *testing.T) {
	evidence := MemoryEvidence{Load: workerproc.StageMemory{
		ActiveBytes: 40, PeakBytes: 40,
		ProcessPhysicalBytes: 100, ProcessPeakPhysicalBytes: 110,
	}}
	updateMaxObserved(&evidence, evidence.Load)
	observePhase(&evidence, "prefill", workerproc.StageMemory{
		ActiveBytes: 50, CacheBytes: 7, PeakBytes: 100,
		ProcessPhysicalBytes: 120, ProcessPeakPhysicalBytes: 130,
	})
	observePhase(&evidence, "decode", workerproc.StageMemory{
		ActiveBytes: 80, CacheBytes: 9, PeakBytes: 120,
		ProcessPhysicalBytes: 140, ProcessPeakPhysicalBytes: 150,
	})
	if evidence.MaxObservedBytes != 129 {
		t.Fatalf("max observed = %d, want 129", evidence.MaxObservedBytes)
	}
	if evidence.Prefill.PeakBytes != 100 || evidence.Decode.PeakBytes != 120 {
		t.Fatalf("unexpected phase evidence: %+v", evidence)
	}
	if evidence.MaxProcessPhysicalBytes != 150 {
		t.Fatalf("max process physical = %d, want 150", evidence.MaxProcessPhysicalBytes)
	}
	if !completeMemoryEvidence(evidence) {
		t.Fatalf("complete evidence was rejected: %+v", evidence)
	}
}

func TestComplementaryShardsRejectsFullModel(t *testing.T) {
	plan := generation.ShardPlan{
		Producer: generation.Shard{ID: "producer", LayerStart: 0, LayerEnd: 24},
		Consumer: generation.Shard{ID: "consumer", LayerStart: 24, LayerEnd: 48},
	}
	producer := &workerproc.PersistentWorkerState{LoadedShards: []workerproc.PersistentShardSnapshot{{
		ShardID: "producer", LayerStart: 0, LayerEnd: 48, OwnsInput: true, OwnsOutput: true,
	}}}
	consumer := &workerproc.PersistentWorkerState{LoadedShards: []workerproc.PersistentShardSnapshot{{
		ShardID: "consumer", LayerStart: 24, LayerEnd: 48, OwnsOutput: true,
	}}}
	if complementaryShards(producer, consumer, plan, 48) {
		t.Fatal("full-model producer was accepted as a complementary shard")
	}

	producer.LoadedShards[0].LayerEnd = 24
	producer.LoadedShards[0].OwnsOutput = false
	if !complementaryShards(producer, consumer, plan, 48) {
		t.Fatal("valid complementary shards were rejected")
	}
}

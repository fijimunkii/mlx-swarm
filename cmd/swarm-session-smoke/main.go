package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

const (
	defaultModelID = "mlx-community/gemma-3-270m-it-4bit"
	sequenceA      = "sequence-a"
	sequenceB      = "sequence-b"
)

type summary struct {
	Model                  string                 `json:"model"`
	ModelType              string                 `json:"modelType"`
	ShardID                string                 `json:"shardID"`
	LayerStart             int                    `json:"layerStart"`
	LayerEnd               int                    `json:"layerEnd"`
	LoadCount              int                    `json:"loadCount"`
	ForwardCount           int                    `json:"forwardCount"`
	OpenSequenceCount      int                    `json:"openSequenceCount"`
	BoundaryBytes          int                    `json:"boundaryBytes"`
	BoundarySHA256         string                 `json:"boundarySHA256"`
	OutputsStable          bool                   `json:"outputsStable"`
	SequenceIsolation      bool                   `json:"sequenceIsolation"`
	ShardIsolation         bool                   `json:"shardIsolation"`
	CancellationReported   bool                   `json:"cancellationReported"`
	CrashReported          bool                   `json:"crashReported"`
	LoadedMemory           workerproc.StageMemory `json:"loadedMemory"`
	AfterForwardMemory     workerproc.StageMemory `json:"afterForwardMemory"`
	AfterUnloadActiveBytes int                    `json:"afterUnloadActiveBytes"`
	AfterUnloadCacheBytes  int                    `json:"afterUnloadCacheBytes"`
}

func main() {
	worker := flag.String("worker", workerproc.DefaultPath(), "path to the built MLXWorker executable")
	peer := flag.String("peer", "", "optional swarmd base URL; lifecycle requests use its persistent worker")
	model := flag.String("model", defaultModelID, "real checkpoint model ID")
	shardID := flag.String("shard", "fixture-producer", "logical shard ID")
	layerStart := flag.Int("layer-start", 0, "first transformer layer in the retained range")
	layerEnd := flag.Int("layer-end", 9, "exclusive end of the retained layer range")
	forwards := flag.Int("forwards", 100, "number of forwards to reuse the loaded shard")
	timeout := flag.Duration("timeout", 2*time.Minute, "overall smoke timeout")
	flag.Parse()
	if *forwards < 100 {
		fatalf("forwards must be at least 100")
	}

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	var client workerproc.PersistentCaller
	var directClient *workerproc.PersistentClient
	cleanShutdown := false
	if *peer == "" {
		var err error
		directClient, err = workerproc.StartPersistent(*worker)
		if err != nil {
			fatalf("start persistent worker: %v", err)
		}
		client = directClient
		defer func() {
			if !cleanShutdown {
				_ = directClient.Kill()
			}
		}()
	} else {
		var err error
		client, err = workerproc.NewHTTPPersistentClient(*peer, nil)
		if err != nil {
			fatalf("configure swarmd client: %v", err)
		}
	}

	mustCall(ctx, client, workerproc.PersistentRequest{Command: "health"})
	canceledContext, cancelRequest := context.WithCancel(ctx)
	cancelRequest()
	_, cancellationErr := client.Call(
		canceledContext,
		workerproc.PersistentRequest{Command: "state"},
	)
	cancellationReported := errors.Is(cancellationErr, context.Canceled)
	if !cancellationReported {
		fatalf("canceled request returned %v", cancellationErr)
	}
	mustCall(ctx, client, workerproc.PersistentRequest{Command: "health"})
	loaded := mustCall(ctx, client, workerproc.PersistentRequest{
		Command: "loadShard",
		LoadShard: &workerproc.PersistentLoadShardRequest{
			ModelID:    *model,
			ShardID:    *shardID,
			LayerStart: *layerStart,
			LayerEnd:   *layerEnd,
			OwnsInput:  true,
			OwnsOutput: false,
		},
	})
	if loaded.Result == nil || loaded.Result.Shard == nil {
		fatalf("loadShard returned no shard snapshot")
	}

	for _, sequenceID := range []string{sequenceA, sequenceB} {
		mustCall(ctx, client, workerproc.PersistentRequest{
			Command: "openSequence",
			Sequence: &workerproc.PersistentSequenceRequest{
				ShardID:    *shardID,
				SequenceID: sequenceID,
			},
		})
	}

	tokens := tokenTensor([]int32{1, 2, 3, 4, 5, 6})
	var expectedDigest [sha256.Size]byte
	var boundaryBytes int
	var afterForward workerproc.StageMemory
	outputsStable := true
	for index := 0; index < *forwards; index++ {
		sequenceID := sequenceA
		if index%2 == 1 {
			sequenceID = sequenceB
		}
		response := mustCall(ctx, client, workerproc.PersistentRequest{
			Command: "forward",
			Forward: &workerproc.PersistentForwardRequest{
				ShardID:    *shardID,
				SequenceID: sequenceID,
				Position:   0,
				InputKind:  "tokens",
				Input:      tokens,
			},
		})
		if response.Result == nil || response.Result.Forward == nil {
			fatalf("forward %d returned no tensor", index)
		}
		forward := response.Result.Forward
		digest := sha256.Sum256(forward.Output.Data)
		if index == 0 {
			expectedDigest = digest
			boundaryBytes = len(forward.Output.Data)
		} else if digest != expectedDigest {
			outputsStable = false
		}
		afterForward = forward.Memory
	}

	_, isolationErr := client.Call(ctx, workerproc.PersistentRequest{
		Command: "forward",
		Forward: &workerproc.PersistentForwardRequest{
			ShardID:    *shardID,
			SequenceID: "not-open",
			Position:   0,
			InputKind:  "tokens",
			Input:      tokens,
		},
	})
	var responseErr *workerproc.WorkerResponseError
	sequenceIsolation := errors.As(isolationErr, &responseErr)
	_, shardIsolationErr := client.Call(ctx, workerproc.PersistentRequest{
		Command: "forward",
		Forward: &workerproc.PersistentForwardRequest{
			ShardID:    "not-loaded",
			SequenceID: sequenceA,
			Position:   0,
			InputKind:  "tokens",
			Input:      tokens,
		},
	})
	responseErr = nil
	shardIsolation := errors.As(shardIsolationErr, &responseErr)

	stateResponse := mustCall(ctx, client, workerproc.PersistentRequest{Command: "state"})
	if stateResponse.Result == nil || stateResponse.Result.State == nil {
		fatalf("state returned no snapshot")
	}
	state := stateResponse.Result.State
	if len(state.LoadedShards) != 1 {
		fatalf("expected one loaded shard, got %d", len(state.LoadedShards))
	}
	shard := state.LoadedShards[0]
	if state.LoadCount != 1 || state.ForwardCount != *forwards || shard.ForwardCount != *forwards {
		fatalf("unexpected reuse counters: loads=%d forwards=%d shardForwards=%d", state.LoadCount, state.ForwardCount, shard.ForwardCount)
	}
	if shard.OpenSequenceCount != 2 {
		fatalf("expected two open sequences, got %d", shard.OpenSequenceCount)
	}
	if !outputsStable || !sequenceIsolation || !shardIsolation {
		fatalf("persistent worker output or identifier isolation failed")
	}

	for _, sequenceID := range []string{sequenceA, sequenceB} {
		mustCall(ctx, client, workerproc.PersistentRequest{
			Command: "closeSequence",
			Sequence: &workerproc.PersistentSequenceRequest{
				ShardID:    *shardID,
				SequenceID: sequenceID,
			},
		})
	}
	unloaded := mustCall(ctx, client, workerproc.PersistentRequest{
		Command: "unloadShard",
		Shard:   &workerproc.PersistentShardRequest{ShardID: *shardID},
	})
	if unloaded.Result == nil || unloaded.Result.State == nil || len(unloaded.Result.State.LoadedShards) != 0 {
		fatalf("unload did not clear the shard")
	}
	afterUnloadActive := unloaded.Result.State.Memory.ActiveBytes
	afterUnloadCache := unloaded.Result.State.Memory.CacheBytes
	if afterUnloadActive > 1<<20 || afterUnloadCache > 1<<20 {
		fatalf("unload retained MLX memory: active=%d cache=%d", afterUnloadActive, afterUnloadCache)
	}
	if directClient != nil {
		if err := directClient.Shutdown(ctx); err != nil {
			fatalf("clean shutdown: %v", err)
		}
		cleanShutdown = true
	}

	crashReported := proveCrashReported(ctx, *worker)
	if !crashReported {
		fatalf("killed worker did not return a bounded error")
	}

	result := summary{
		Model:                  *model,
		ModelType:              shard.ModelType,
		ShardID:                shard.ShardID,
		LayerStart:             shard.LayerStart,
		LayerEnd:               shard.LayerEnd,
		LoadCount:              state.LoadCount,
		ForwardCount:           state.ForwardCount,
		OpenSequenceCount:      shard.OpenSequenceCount,
		BoundaryBytes:          boundaryBytes,
		BoundarySHA256:         hex.EncodeToString(expectedDigest[:]),
		OutputsStable:          outputsStable,
		SequenceIsolation:      sequenceIsolation,
		ShardIsolation:         shardIsolation,
		CancellationReported:   cancellationReported,
		CrashReported:          crashReported,
		LoadedMemory:           shard.LoadedMemory,
		AfterForwardMemory:     afterForward,
		AfterUnloadActiveBytes: afterUnloadActive,
		AfterUnloadCacheBytes:  afterUnloadCache,
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		fatalf("encode result: %v", err)
	}
	fmt.Println(string(encoded))
}

func mustCall(
	ctx context.Context,
	client workerproc.PersistentCaller,
	request workerproc.PersistentRequest,
) workerproc.PersistentResponse {
	response, err := client.Call(ctx, request)
	if err != nil {
		fatalf("%s: %v", request.Command, err)
	}
	return response
}

func tokenTensor(tokens []int32) workerproc.WireTensor {
	data := make([]byte, len(tokens)*4)
	for index, token := range tokens {
		binary.LittleEndian.PutUint32(data[index*4:], uint32(token))
	}
	return workerproc.WireTensor{
		Shape: []int{1, len(tokens)},
		DType: "int32",
		Data:  data,
	}
}

func proveCrashReported(ctx context.Context, worker string) bool {
	client, err := workerproc.StartPersistent(worker)
	if err != nil {
		return false
	}
	if _, err := client.Call(ctx, workerproc.PersistentRequest{Command: "health"}); err != nil {
		_ = client.Kill()
		return false
	}
	if err := client.Kill(); err != nil {
		return false
	}
	waitCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	_ = client.Wait(waitCtx)
	_, err = client.Call(waitCtx, workerproc.PersistentRequest{Command: "state"})
	return err != nil
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}

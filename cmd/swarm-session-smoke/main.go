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
	"strings"
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
	InputKindValidation    bool                   `json:"inputKindValidation"`
	TensorValidation       bool                   `json:"tensorValidation"`
	CancellationReported   bool                   `json:"cancellationReported"`
	CrashReported          *bool                  `json:"crashReported,omitempty"`
	LoadedMemory           workerproc.StageMemory `json:"loadedMemory"`
	AfterForwardMemory     workerproc.StageMemory `json:"afterForwardMemory"`
	AfterUnloadActiveBytes int                    `json:"afterUnloadActiveBytes"`
	AfterUnloadCacheBytes  int                    `json:"afterUnloadCacheBytes"`
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
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
		return errors.New("forwards must be at least 100")
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
			return fmt.Errorf("start persistent worker: %w", err)
		}
		client = directClient
		defer func() {
			if !cleanShutdown {
				terminatePersistentClient(directClient)
			}
		}()
	} else {
		var err error
		client, err = workerproc.NewHTTPPersistentClient(*peer, nil)
		if err != nil {
			return fmt.Errorf("configure swarmd client: %w", err)
		}
	}

	if _, err := call(ctx, client, workerproc.PersistentRequest{Command: "health"}); err != nil {
		return err
	}
	canceledContext, cancelRequest := context.WithCancel(ctx)
	cancelRequest()
	_, cancellationErr := client.Call(
		canceledContext,
		workerproc.PersistentRequest{Command: "state"},
	)
	cancellationReported := errors.Is(cancellationErr, context.Canceled)
	if !cancellationReported {
		return fmt.Errorf("canceled request returned %v", cancellationErr)
	}
	if _, err := call(ctx, client, workerproc.PersistentRequest{Command: "health"}); err != nil {
		return err
	}
	loaded, err := call(ctx, client, workerproc.PersistentRequest{
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
	if err != nil {
		return err
	}
	if loaded.Result == nil || loaded.Result.Shard == nil {
		return errors.New("loadShard returned no shard snapshot")
	}

	for _, sequenceID := range []string{sequenceA, sequenceB} {
		if _, err := call(ctx, client, workerproc.PersistentRequest{
			Command: "openSequence",
			Sequence: &workerproc.PersistentSequenceRequest{
				ShardID:    *shardID,
				SequenceID: sequenceID,
			},
		}); err != nil {
			return err
		}
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
		response, err := call(ctx, client, workerproc.PersistentRequest{
			Command: "forward",
			Forward: &workerproc.PersistentForwardRequest{
				ShardID:    *shardID,
				SequenceID: sequenceID,
				Position:   0,
				InputKind:  "tokens",
				Input:      tokens,
			},
		})
		if err != nil {
			return err
		}
		if response.Result == nil || response.Result.Forward == nil {
			return fmt.Errorf("forward %d returned no tensor", index)
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
	_, inputKindErr := client.Call(ctx, workerproc.PersistentRequest{
		Command: "forward",
		Forward: &workerproc.PersistentForwardRequest{
			ShardID:    *shardID,
			SequenceID: sequenceA,
			Position:   0,
			InputKind:  "hidden",
			Input:      tokens,
		},
	})
	responseErr = nil
	inputKindValidation := errors.As(inputKindErr, &responseErr)
	validateTensor := func(input workerproc.WireTensor, want string) bool {
		_, err := client.Call(ctx, workerproc.PersistentRequest{
			Command: "forward",
			Forward: &workerproc.PersistentForwardRequest{
				ShardID:    *shardID,
				SequenceID: sequenceA,
				Position:   0,
				InputKind:  "tokens",
				Input:      input,
			},
		})
		var workerErr *workerproc.WorkerResponseError
		return errors.As(err, &workerErr) && strings.Contains(workerErr.Message, want)
	}
	tensorValidation := validateTensor(
		workerproc.WireTensor{Shape: []int{1, 2}, DType: "int32", Data: []byte{0}},
		"invalid wire tensor byte count",
	) && validateTensor(
		workerproc.WireTensor{Shape: []int{1, 1}, DType: "float32", Data: make([]byte, 4)},
		"token tensor dtype must be int32",
	) && validateTensor(
		workerproc.WireTensor{Shape: []int{1, 1, 1}, DType: "int32", Data: make([]byte, 4)},
		"token tensor shape must have batch size 1 and 2 positive dimensions",
	) && validateTensor(
		tokenTensor([]int32{-1}),
		"token ID -1 is outside vocabulary size",
	)
	if _, err := call(ctx, client, workerproc.PersistentRequest{Command: "health"}); err != nil {
		return fmt.Errorf("health after malformed tensor: %w", err)
	}

	stateResponse, err := call(ctx, client, workerproc.PersistentRequest{Command: "state"})
	if err != nil {
		return err
	}
	if stateResponse.Result == nil || stateResponse.Result.State == nil {
		return errors.New("state returned no snapshot")
	}
	state := stateResponse.Result.State
	if len(state.LoadedShards) != 1 {
		return fmt.Errorf("expected one loaded shard, got %d", len(state.LoadedShards))
	}
	shard := state.LoadedShards[0]
	if state.LoadCount != 1 || state.ForwardCount != *forwards || shard.ForwardCount != *forwards {
		return fmt.Errorf("unexpected reuse counters: loads=%d forwards=%d shardForwards=%d", state.LoadCount, state.ForwardCount, shard.ForwardCount)
	}
	if shard.OpenSequenceCount != 2 {
		return fmt.Errorf("expected two open sequences, got %d", shard.OpenSequenceCount)
	}
	if !outputsStable || !sequenceIsolation || !shardIsolation || !inputKindValidation || !tensorValidation {
		return errors.New("persistent worker validation or identifier isolation failed")
	}

	for _, sequenceID := range []string{sequenceA, sequenceB} {
		if _, err := call(ctx, client, workerproc.PersistentRequest{
			Command: "closeSequence",
			Sequence: &workerproc.PersistentSequenceRequest{
				ShardID:    *shardID,
				SequenceID: sequenceID,
			},
		}); err != nil {
			return err
		}
	}
	unloaded, err := call(ctx, client, workerproc.PersistentRequest{
		Command: "unloadShard",
		Shard:   &workerproc.PersistentShardRequest{ShardID: *shardID},
	})
	if err != nil {
		return err
	}
	if unloaded.Result == nil || unloaded.Result.State == nil || len(unloaded.Result.State.LoadedShards) != 0 {
		return errors.New("unload did not clear the shard")
	}
	afterUnloadActive := unloaded.Result.State.Memory.ActiveBytes
	afterUnloadCache := unloaded.Result.State.Memory.CacheBytes
	if afterUnloadActive > 1<<20 || afterUnloadCache > 1<<20 {
		return fmt.Errorf("unload retained MLX memory: active=%d cache=%d", afterUnloadActive, afterUnloadCache)
	}
	if directClient != nil {
		if err := directClient.Shutdown(ctx); err != nil {
			return fmt.Errorf("clean shutdown: %w", err)
		}
		cleanShutdown = true
	}

	var crashReported *bool
	if directClient != nil {
		proved := proveCrashReported(ctx, *worker)
		if !proved {
			return errors.New("killed worker did not return a bounded error")
		}
		crashReported = &proved
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
		InputKindValidation:    inputKindValidation,
		TensorValidation:       tensorValidation,
		CancellationReported:   cancellationReported,
		CrashReported:          crashReported,
		LoadedMemory:           shard.LoadedMemory,
		AfterForwardMemory:     afterForward,
		AfterUnloadActiveBytes: afterUnloadActive,
		AfterUnloadCacheBytes:  afterUnloadCache,
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		return fmt.Errorf("encode result: %w", err)
	}
	fmt.Println(string(encoded))
	return nil
}

func call(
	ctx context.Context,
	client workerproc.PersistentCaller,
	request workerproc.PersistentRequest,
) (workerproc.PersistentResponse, error) {
	response, err := client.Call(ctx, request)
	if err != nil {
		return workerproc.PersistentResponse{}, fmt.Errorf("%s: %w", request.Command, err)
	}
	return response, nil
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
	defer terminatePersistentClient(client)
	if _, err := client.Call(ctx, workerproc.PersistentRequest{Command: "health"}); err != nil {
		return false
	}
	if err := client.Kill(); err != nil {
		return false
	}
	waitCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	waitErr := client.Wait(waitCtx)
	cancel()
	if waitErr == nil || isContextError(waitErr) {
		return false
	}
	callCtx, cancelCall := context.WithTimeout(ctx, 5*time.Second)
	defer cancelCall()
	_, callErr := client.Call(callCtx, workerproc.PersistentRequest{Command: "state"})
	return callErr != nil && !isContextError(callErr)
}

func isContextError(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

func terminatePersistentClient(client *workerproc.PersistentClient) {
	_ = client.Kill()
	reapContext, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = client.Wait(reapContext)
}

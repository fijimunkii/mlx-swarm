package generation

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

func TestGenerateStopsAtMaximumAndCleansSequences(t *testing.T) {
	producer, consumer, reference := fakeSwarm([]int32{3, 2, 3})
	session := newFakeSession(t, producer, consumer, reference)

	result, err := session.Generate(context.Background(), Request{
		Prompt: "hello", MaxTokens: 2, SequenceID: "maximum",
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.StopReason != "max_tokens" || len(result.GeneratedTokenIDs) != 2 {
		t.Fatalf("unexpected result: stop=%s tokens=%v", result.StopReason, result.GeneratedTokenIDs)
	}
	if result.Verification == nil || !result.Verification.GreedyTokenIDsMatch ||
		result.Verification.ComparedTokens != 2 {
		t.Fatalf("unexpected verification: %+v", result.Verification)
	}
	assertNoFakeSequences(t, producer, consumer, reference)
}

func TestGenerateStopsAtEOSAndCleansSequences(t *testing.T) {
	producer, consumer, reference := fakeSwarm([]int32{1, 3})
	session := newFakeSession(t, producer, consumer, reference)

	result, err := session.Generate(context.Background(), Request{
		Prompt: "hello", MaxTokens: 5, SequenceID: "eos",
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.StopReason != "eos" || len(result.GeneratedTokenIDs) != 1 ||
		result.GeneratedTokenIDs[0] != 1 {
		t.Fatalf("unexpected result: stop=%s tokens=%v", result.StopReason, result.GeneratedTokenIDs)
	}
	assertNoFakeSequences(t, producer, consumer, reference)
}

func TestTokenizerFailureDoesNotOpenSequences(t *testing.T) {
	producer, consumer, reference := fakeSwarm([]int32{3})
	producer.tokenizeErr = errors.New("missing tokenizer.json")
	session := newFakeSession(t, producer, consumer, reference)

	_, err := session.Generate(context.Background(), Request{
		Prompt: "hello", MaxTokens: 1, SequenceID: "tokenizer-failure",
	})
	if err == nil {
		t.Fatal("expected tokenizer failure")
	}
	assertNoFakeSequences(t, producer, consumer, reference)
}

func TestCancellationClosesOpenedSequences(t *testing.T) {
	producer, consumer, reference := fakeSwarm([]int32{3, 2})
	producer.blockDecode = true
	session := newFakeSession(t, producer, consumer, reference)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	_, err := session.Generate(ctx, Request{
		Prompt: "hello", MaxTokens: 2, SequenceID: "cancelled",
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected deadline error, got %v", err)
	}
	assertNoFakeSequences(t, producer, consumer, reference)
}

func TestOpenResponseFailureClosesAttemptedSequence(t *testing.T) {
	producer, consumer, reference := fakeSwarm([]int32{3})
	consumer.openErr = context.DeadlineExceeded
	session := newFakeSession(t, producer, consumer, reference)

	_, err := session.Generate(context.Background(), Request{
		Prompt: "hello", MaxTokens: 1, SequenceID: "ambiguous-open",
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected ambiguous open error, got %v", err)
	}
	assertNoFakeSequences(t, producer, consumer, reference)
}

func TestGreedyTokenUsesFinalPositionAndLowestTie(t *testing.T) {
	tensor := float32Tensor([]int{1, 2, 4}, []float32{
		99, 0, 0, 0,
		1, 7, 7, 2,
	})
	token, err := greedyToken(tensor)
	if err != nil {
		t.Fatal(err)
	}
	if token != 1 {
		t.Fatalf("greedy token = %d, want lowest tied index 1", token)
	}
}

func newFakeSession(
	t *testing.T,
	producer *fakeWorker,
	consumer *fakeWorker,
	reference *fakeWorker,
) *Session {
	t.Helper()
	session, err := NewSession(
		context.Background(),
		producer,
		consumer,
		reference,
		SessionConfig{Model: "test/model", RTol: 1e-4, ATol: 1e-4},
	)
	if err != nil {
		t.Fatal(err)
	}
	return session
}

func fakeSwarm(tokens []int32) (*fakeWorker, *fakeWorker, *fakeWorker) {
	return &fakeWorker{role: "producer", tokens: tokens, sequences: map[string]bool{}},
		&fakeWorker{role: "consumer", tokens: tokens, sequences: map[string]bool{}},
		&fakeWorker{role: "reference", tokens: tokens, sequences: map[string]bool{}}
}

func assertNoFakeSequences(t *testing.T, workers ...*fakeWorker) {
	t.Helper()
	for _, worker := range workers {
		worker.mu.Lock()
		count := len(worker.sequences)
		worker.mu.Unlock()
		if count != 0 {
			t.Fatalf("%s retained %d sequences", worker.role, count)
		}
	}
}

type fakeWorker struct {
	mu          sync.Mutex
	role        string
	tokens      []int32
	sequences   map[string]bool
	shards      []workerproc.PersistentShardSnapshot
	loadCount   int
	logitIndex  int
	tokenizeErr error
	openErr     error
	blockDecode bool
}

func (worker *fakeWorker) Call(
	ctx context.Context,
	request workerproc.PersistentRequest,
) (workerproc.PersistentResponse, error) {
	if err := ctx.Err(); err != nil {
		return workerproc.PersistentResponse{}, err
	}
	worker.mu.Lock()
	defer worker.mu.Unlock()
	result := &workerproc.PersistentWorkerResult{}
	switch request.Command {
	case "modelInfo":
		result.Model = &workerproc.PersistentModelResult{
			ModelID: request.Model.ModelID, ModelType: "test", LayerCount: 2,
		}
	case "state":
		kv := 0
		if len(worker.sequences) > 0 {
			kv = len(worker.sequences)
		}
		result.State = &workerproc.PersistentWorkerState{
			LoadedShards: worker.shards, LoadCount: worker.loadCount,
			KVCacheBytes: kv, RetainedBytes: kv,
		}
	case "loadShard":
		load := request.LoadShard
		snapshot := workerproc.PersistentShardSnapshot{
			ShardID: load.ShardID, ModelID: load.ModelID,
			LayerStart: load.LayerStart, LayerEnd: load.LayerEnd,
			OwnsInput: load.OwnsInput, OwnsOutput: load.OwnsOutput,
		}
		worker.shards = append(worker.shards, snapshot)
		worker.loadCount++
		result.Shard = &snapshot
	case "tokenize":
		if worker.tokenizeErr != nil {
			return workerproc.PersistentResponse{}, worker.tokenizeErr
		}
		eos := int32(1)
		result.Text = &workerproc.PersistentTextResult{
			ModelID: request.Text.ModelID, TokenIDs: []int32{2, 4}, EOSTokenID: &eos,
		}
	case "detokenize":
		text := "decoded"
		result.Text = &workerproc.PersistentTextResult{ModelID: request.Text.ModelID, Text: &text}
	case "openSequence":
		worker.sequences[request.Sequence.SequenceID] = true
		if worker.openErr != nil {
			return workerproc.PersistentResponse{}, worker.openErr
		}
	case "closeSequence":
		delete(worker.sequences, request.Sequence.SequenceID)
	case "prefill", "decode":
		if request.Command == "decode" && worker.blockDecode {
			worker.mu.Unlock()
			<-ctx.Done()
			worker.mu.Lock()
			return workerproc.PersistentResponse{}, ctx.Err()
		}
		forward := request.Forward
		output := workerproc.WireTensor{
			Shape: []int{1, 1, 1}, DType: "float32", Data: []byte{0, 0, 0, 0},
		}
		if worker.role != "producer" {
			token := worker.tokens[min(worker.logitIndex, len(worker.tokens)-1)]
			values := []float32{0, 0, 0, 0, 0}
			values[token] = 10
			output = float32Tensor([]int{1, 1, len(values)}, values)
			worker.logitIndex++
		}
		nextPosition := forward.Position + 1
		if request.Command == "prefill" {
			nextPosition = uint64(forward.Input.Shape[1])
		}
		result.Forward = &workerproc.PersistentForwardResult{
			ShardID: forward.ShardID, SequenceID: forward.SequenceID,
			Operation: request.Command, Position: forward.Position, NextPosition: nextPosition,
			Output: output, ComputeMicros: 1, KVCacheBytes: 1,
		}
	default:
		return workerproc.PersistentResponse{}, errors.New("unexpected command " + request.Command)
	}
	return workerproc.PersistentResponse{OK: true, Result: result}, nil
}

func float32Tensor(shape []int, values []float32) workerproc.WireTensor {
	data := make([]byte, len(values)*4)
	for index, value := range values {
		binary.LittleEndian.PutUint32(data[index*4:], math.Float32bits(value))
	}
	return workerproc.WireTensor{Shape: shape, DType: "float32", Data: data}
}

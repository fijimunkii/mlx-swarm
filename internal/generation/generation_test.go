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

func TestGenerateObservesHotPathSeparatelyFromReference(t *testing.T) {
	producer, consumer, reference := fakeSwarm([]int32{3, 2, 3})
	var samples []StageSample
	session, err := NewSession(
		context.Background(), producer, consumer, reference,
		SessionConfig{
			Model: "test/model", RTol: 1e-4, ATol: 1e-4,
			Observer: func(sample StageSample) { samples = append(samples, sample) },
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	_, err = session.Generate(context.Background(), Request{
		Prompt: "hello", MaxTokens: 2, SequenceID: "observed",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 2 || samples[0].Operation != "prefill" || samples[1].Operation != "decode" {
		t.Fatalf("unexpected observations: %+v", samples)
	}
	if samples[0].InputTokenCount != 2 || samples[1].InputTokenCount != 1 {
		t.Fatalf("unexpected input token counts: %+v", samples)
	}
	for _, sample := range samples {
		if sample.BoundaryTensorBytes <= 0 || sample.BoundaryWireBytes <= sample.BoundaryTensorBytes {
			t.Fatalf("invalid boundary accounting: %+v", sample)
		}
		if sample.ReferenceComputeMicros == 0 || sample.ReferenceKVCacheBytes == 0 {
			t.Fatalf("missing reference baseline: %+v", sample)
		}
		if sample.TokenLatencyMicros < sample.DistributedEndToEndMicros ||
			sample.ReferenceTokenLatencyMicros < sample.ReferenceWallMicros {
			t.Fatalf("invalid token latency: %+v", sample)
		}
	}
	info := session.Info()
	if info.ReferenceShardID == "" || info.ShardPlan.Producer.LayerEnd != 1 ||
		info.ShardPlan.Consumer.LayerStart != 1 {
		t.Fatalf("unexpected session info: %+v", info)
	}
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

func TestGenerateCanIgnoreEOSForFixedBenchmarkPlan(t *testing.T) {
	producer, consumer, reference := fakeSwarm([]int32{1, 3})
	session := newFakeSession(t, producer, consumer, reference)

	result, err := session.Generate(context.Background(), Request{
		Prompt: "hello", MaxTokens: 2, SequenceID: "ignore-eos", IgnoreEOS: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.StopReason != "max_tokens" || len(result.GeneratedTokenIDs) != 2 ||
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

func TestRejectedOpenDoesNotCloseExistingSequence(t *testing.T) {
	producer, consumer, reference := fakeSwarm([]int32{3})
	consumer.sequences["shared"] = "another-owner"
	session := newFakeSession(t, producer, consumer, reference)

	_, err := session.Generate(context.Background(), Request{
		Prompt: "hello", MaxTokens: 1, SequenceID: "shared",
	})
	var workerErr *workerproc.WorkerResponseError
	if !errors.As(err, &workerErr) {
		t.Fatalf("expected definitive worker rejection, got %v", err)
	}
	producer.mu.Lock()
	producerCount := len(producer.sequences)
	producer.mu.Unlock()
	consumer.mu.Lock()
	consumerOwner, consumerRetained := consumer.sequences["shared"]
	consumerCount := len(consumer.sequences)
	consumer.mu.Unlock()
	if producerCount != 0 || !consumerRetained || consumerOwner != "another-owner" || consumerCount != 1 {
		t.Fatalf(
			"unexpected sequence cleanup: producer=%d consumer=%d retained=%t",
			producerCount, consumerCount, consumerRetained,
		)
	}
	assertNoFakeSequences(t, producer, reference)
}

func TestAmbiguousRejectedOpenCannotCloseExistingSequence(t *testing.T) {
	producer, consumer, reference := fakeSwarm([]int32{3})
	consumer.sequences["shared"] = "another-owner"
	consumer.loseOpenRejection = true
	session := newFakeSession(t, producer, consumer, reference)

	_, err := session.Generate(context.Background(), Request{
		Prompt: "hello", MaxTokens: 1, SequenceID: "shared",
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected lost rejection response, got %v", err)
	}
	producer.mu.Lock()
	producerCount := len(producer.sequences)
	producer.mu.Unlock()
	consumer.mu.Lock()
	consumerOwner, consumerRetained := consumer.sequences["shared"]
	consumerCount := len(consumer.sequences)
	consumer.mu.Unlock()
	if producerCount != 0 || !consumerRetained || consumerOwner != "another-owner" || consumerCount != 1 {
		t.Fatalf(
			"ambiguous cleanup changed another sequence: producer=%d consumer=%d owner=%q",
			producerCount, consumerCount, consumerOwner,
		)
	}
	assertNoFakeSequences(t, producer, reference)
}

func TestNewSessionRejectsCheckpointFingerprintMismatch(t *testing.T) {
	producer, consumer, _ := fakeSwarm([]int32{3})
	consumer.checkpointFingerprint = "different-checkpoint"
	_, err := NewSession(
		context.Background(), producer, consumer, nil,
		SessionConfig{Model: "test/model", RTol: 1e-4, ATol: 1e-4},
	)
	if err == nil {
		t.Fatal("expected checkpoint fingerprint mismatch")
	}
	if producer.loadCount != 0 || consumer.loadCount != 0 {
		t.Fatalf("mismatched checkpoints loaded shards: producer=%d consumer=%d", producer.loadCount, consumer.loadCount)
	}
}

func TestNewSessionReusesConcurrentlyLoadedShard(t *testing.T) {
	producer, consumer, _ := fakeSwarm([]int32{3})
	consumer.concurrentLoad = true
	if _, err := NewSession(
		context.Background(), producer, consumer, nil,
		SessionConfig{Model: "test/model", RTol: 1e-4, ATol: 1e-4},
	); err != nil {
		t.Fatalf("reuse concurrently loaded shard: %v", err)
	}
	if consumer.loadCount != 1 || len(consumer.shards) != 1 {
		t.Fatalf("consumer load state: count=%d shards=%d", consumer.loadCount, len(consumer.shards))
	}
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

func TestNewSessionRejectsNonFiniteTolerances(t *testing.T) {
	producer, consumer, _ := fakeSwarm([]int32{3})
	for _, test := range []struct {
		name string
		rtol float64
		atol float64
	}{
		{name: "NaN rtol", rtol: math.NaN()},
		{name: "infinite rtol", rtol: math.Inf(1)},
		{name: "NaN atol", atol: math.NaN()},
		{name: "infinite atol", atol: math.Inf(1)},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := NewSession(
				context.Background(),
				producer,
				consumer,
				nil,
				SessionConfig{Model: "test/model", RTol: test.rtol, ATol: test.atol},
			)
			if err == nil {
				t.Fatal("expected non-finite tolerance error")
			}
		})
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
	return newFakeWorker("producer", tokens),
		newFakeWorker("consumer", tokens),
		newFakeWorker("reference", tokens)
}

func newFakeWorker(role string, tokens []int32) *fakeWorker {
	return &fakeWorker{
		role: role, tokens: tokens, sequences: map[string]string{},
		checkpointFingerprint: "test-checkpoint",
	}
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
	mu                    sync.Mutex
	role                  string
	tokens                []int32
	sequences             map[string]string
	shards                []workerproc.PersistentShardSnapshot
	checkpointFingerprint string
	loadCount             int
	logitIndex            int
	tokenizeErr           error
	openErr               error
	loseOpenRejection     bool
	concurrentLoad        bool
	blockDecode           bool
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
			CheckpointFingerprint: worker.checkpointFingerprint,
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
			CheckpointFingerprint: worker.checkpointFingerprint,
			LayerStart:            load.LayerStart, LayerEnd: load.LayerEnd,
			OwnsInput: load.OwnsInput, OwnsOutput: load.OwnsOutput,
		}
		worker.shards = append(worker.shards, snapshot)
		worker.loadCount++
		if worker.concurrentLoad {
			worker.concurrentLoad = false
			return workerproc.PersistentResponse{}, &workerproc.WorkerResponseError{
				RequestID: request.RequestID,
				Message:   "shard is already loaded",
			}
		}
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
		if ownerID, exists := worker.sequences[request.Sequence.SequenceID]; exists {
			if request.Sequence.OwnerID != "" && ownerID == request.Sequence.OwnerID {
				break
			}
			if worker.loseOpenRejection {
				return workerproc.PersistentResponse{}, context.DeadlineExceeded
			}
			return workerproc.PersistentResponse{}, &workerproc.WorkerResponseError{
				RequestID: request.RequestID,
				Message:   "sequence is already open",
			}
		}
		worker.sequences[request.Sequence.SequenceID] = request.Sequence.OwnerID
		if worker.openErr != nil {
			return workerproc.PersistentResponse{}, worker.openErr
		}
	case "closeSequence":
		ownerID, exists := worker.sequences[request.Sequence.SequenceID]
		if !exists {
			return workerproc.PersistentResponse{}, &workerproc.WorkerResponseError{
				RequestID: request.RequestID,
				Message:   "sequence is not open",
			}
		}
		if request.Sequence.OwnerID != "" && ownerID != request.Sequence.OwnerID {
			return workerproc.PersistentResponse{}, &workerproc.WorkerResponseError{
				RequestID: request.RequestID,
				Message:   "sequence is owned by another request",
			}
		}
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

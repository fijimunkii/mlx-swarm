package generation

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

func TestPlannedSessionGeneratesAndVerifiesAcrossFiveStages(t *testing.T) {
	workers, reference, targets := plannedFakeSwarm(t, []int32{3, 2, 3})
	var samples []PlannedStageSample
	session, err := NewPlannedSession(
		context.Background(), targets, reference,
		PlannedSessionConfig{
			Model: "test/model", RTol: 1e-4, ATol: 1e-4,
			Observer: func(sample PlannedStageSample) { samples = append(samples, sample) },
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	result, err := session.Generate(context.Background(), Request{
		Prompt: "hello", MaxTokens: 3, SequenceID: "five-stage",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.GeneratedTokenIDs, []int32{3, 2, 3}) {
		t.Fatalf("generated tokens = %v", result.GeneratedTokenIDs)
	}
	if result.StopReason != "max_tokens" {
		t.Fatalf("stop reason = %q", result.StopReason)
	}
	if len(result.ExecutionPlan.Stages) != 5 {
		t.Fatalf("execution stages = %d", len(result.ExecutionPlan.Stages))
	}
	if result.Verification == nil || !result.Verification.GreedyTokenIDsMatch ||
		result.Verification.ComparedTokens != 3 {
		t.Fatalf("verification = %+v", result.Verification)
	}
	if len(result.StageKVCacheBytes) != 5 || len(result.Timing.StageComputeMicros) != 5 {
		t.Fatalf("unexpected stage accounting: kv=%v compute=%v", result.StageKVCacheBytes, result.Timing.StageComputeMicros)
	}
	for index := 0; index < 5; index++ {
		if result.StageKVCacheBytes[index] <= 0 || result.Timing.StageComputeMicros[index] == 0 {
			t.Fatalf("stage %d accounting missing: kv=%d compute=%d", index, result.StageKVCacheBytes[index], result.Timing.StageComputeMicros[index])
		}
	}
	if len(samples) != 3 || samples[0].Operation != "prefill" ||
		samples[1].Operation != "decode" || samples[2].Operation != "decode" {
		t.Fatalf("samples = %+v", samples)
	}
	for sampleIndex, sample := range samples {
		if len(sample.Stages) != 5 {
			t.Fatalf("sample %d stages = %d", sampleIndex, len(sample.Stages))
		}
		for stageIndex, execution := range sample.Stages {
			if execution.Index != stageIndex || execution.KVCacheBytes <= 0 ||
				execution.WallMicros < 0 || execution.ComputeMicros == 0 {
				t.Fatalf("sample %d stage %d evidence = %+v", sampleIndex, stageIndex, execution)
			}
		}
		if sample.ReferenceComputeMicros == 0 || sample.ReferenceKVCacheBytes == 0 {
			t.Fatalf("sample %d missing reference evidence: %+v", sampleIndex, sample)
		}
	}
	assertPlannedNoSequences(t, append(workers, reference)...)
}

func TestPlannedSessionUsesTerminalSamplingOnlyOnFinalStage(t *testing.T) {
	workers, _, targets := plannedFakeSwarm(t, []int32{3, 2})
	var samples []PlannedStageSample
	session, err := NewPlannedSession(
		context.Background(), targets, nil,
		PlannedSessionConfig{
			Model: "test/model", TerminalSampling: true,
			Observer: func(sample PlannedStageSample) { samples = append(samples, sample) },
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := session.Generate(context.Background(), Request{
		Prompt: "hello", MaxTokens: 2, SequenceID: "terminal",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.GeneratedTokenIDs, []int32{3, 2}) {
		t.Fatalf("generated tokens = %v", result.GeneratedTokenIDs)
	}
	if result.Verification != nil {
		t.Fatalf("terminal sampling unexpectedly enabled verification: %+v", result.Verification)
	}
	if len(samples) != 2 {
		t.Fatalf("samples = %d", len(samples))
	}
	for sampleIndex, sample := range samples {
		for stageIndex, execution := range sample.Stages {
			wantTerminal := stageIndex == len(sample.Stages)-1
			if execution.TerminalSampling != wantTerminal {
				t.Fatalf("sample %d stage %d terminal=%t want=%t", sampleIndex, stageIndex, execution.TerminalSampling, wantTerminal)
			}
			if wantTerminal && execution.ResponseTensorBytes != 0 {
				t.Fatalf("terminal stage returned %d tensor bytes", execution.ResponseTensorBytes)
			}
		}
	}
	assertPlannedNoSequences(t, workers...)
}

func TestPlannedSessionReportsFailedMiddleStageAndCleansAllSequences(t *testing.T) {
	workers, _, targets := plannedFakeSwarm(t, []int32{3, 2})
	workers[2].failCommand = "decode"
	workers[2].inferErr = context.DeadlineExceeded
	session, err := NewPlannedSession(
		context.Background(), targets, nil,
		PlannedSessionConfig{Model: "test/model", ForwardTimeout: time.Second},
	)
	if err != nil {
		t.Fatal(err)
	}

	result, err := session.Generate(context.Background(), Request{
		Prompt: "hello", MaxTokens: 2, SequenceID: "middle-failure",
	})
	if err == nil {
		t.Fatal("expected middle-stage failure")
	}
	var generationErr *GenerationError
	if !errors.As(err, &generationErr) || result.Failure == nil {
		t.Fatalf("missing structured failure: result=%+v err=%v", result, err)
	}
	if result.Failure.Phase != "stage_2_decode" ||
		result.Failure.ShardID != targets[2].Stage.ShardID ||
		result.Failure.Operation != "decode" ||
		result.Failure.LastAcceptedTokenIndex != 0 ||
		result.Failure.LastAcceptedTokenID == nil || *result.Failure.LastAcceptedTokenID != 3 {
		t.Fatalf("unexpected failure: %+v", result.Failure)
	}
	assertPlannedNoSequences(t, workers...)
}

func TestPlannedSessionRejectsTerminalSamplingWithReference(t *testing.T) {
	_, reference, targets := plannedFakeSwarm(t, []int32{3})
	_, err := NewPlannedSession(
		context.Background(), targets, reference,
		PlannedSessionConfig{Model: "test/model", TerminalSampling: true},
	)
	if err == nil {
		t.Fatal("expected terminal sampling/reference rejection")
	}
}

func plannedFakeSwarm(
	t *testing.T,
	tokens []int32,
) ([]*plannedFakeWorker, *plannedFakeWorker, []ExecutionTarget) {
	t.Helper()
	plan, err := BuildBalancedExecutionPlan("test/model", "test-checkpoint", 5, 5)
	if err != nil {
		t.Fatal(err)
	}
	workers := make([]*plannedFakeWorker, 5)
	targets := make([]ExecutionTarget, 5)
	for index := range workers {
		workers[index] = newPlannedFakeWorker(fmt.Sprintf("stage-%d", index), 5, tokens)
		workers[index].terminal = index == len(workers)-1
		targets[index] = ExecutionTarget{Stage: plan.Stages[index], Caller: workers[index]}
	}
	reference := newPlannedFakeWorker("reference", 5, tokens)
	reference.terminal = true
	return workers, reference, targets
}

func assertPlannedNoSequences(t *testing.T, workers ...*plannedFakeWorker) {
	t.Helper()
	for _, worker := range workers {
		worker.mu.Lock()
		count := len(worker.sequences)
		worker.mu.Unlock()
		if count != 0 {
			t.Fatalf("%s retained %d sequences", worker.name, count)
		}
	}
}

type plannedFakeWorker struct {
	mu          sync.Mutex
	name        string
	layerCount  int
	tokens      []int32
	terminal    bool
	logitIndex  int
	loaded      []workerproc.PersistentShardSnapshot
	loadCount   int
	sequences   map[string]string
	failCommand string
	inferErr    error
}

func newPlannedFakeWorker(name string, layerCount int, tokens []int32) *plannedFakeWorker {
	return &plannedFakeWorker{
		name: name, layerCount: layerCount, tokens: tokens,
		sequences: map[string]string{},
	}
}

func (worker *plannedFakeWorker) Call(
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
			ModelID: request.Model.ModelID, ModelType: "test", LayerCount: worker.layerCount,
			CheckpointFingerprint: "test-checkpoint", CheckpointBytes: 123,
		}
	case "state":
		kv := 0
		if len(worker.sequences) > 0 {
			kv = len(worker.sequences)
		}
		result.State = &workerproc.PersistentWorkerState{
			LoadedShards: worker.loaded, LoadCount: worker.loadCount,
			KVCacheBytes: kv, RetainedBytes: kv,
		}
	case "loadShard":
		load := request.LoadShard
		snapshot := workerproc.PersistentShardSnapshot{
			ShardID: load.ShardID, ModelID: load.ModelID,
			CheckpointFingerprint: "test-checkpoint",
			LayerStart: load.LayerStart, LayerEnd: load.LayerEnd,
			OwnsInput: load.OwnsInput, OwnsOutput: load.OwnsOutput,
		}
		worker.loaded = append(worker.loaded, snapshot)
		worker.loadCount++
		result.Shard = &snapshot
	case "tokenize":
		eos := int32(1)
		result.Text = &workerproc.PersistentTextResult{
			ModelID: request.Text.ModelID, TokenIDs: []int32{2, 4}, EOSTokenID: &eos,
		}
	case "detokenize":
		text := "decoded"
		result.Text = &workerproc.PersistentTextResult{ModelID: request.Text.ModelID, Text: &text}
	case "openSequence":
		if request.Sequence == nil {
			return workerproc.PersistentResponse{}, errors.New("missing sequence request")
		}
		if owner, exists := worker.sequences[request.Sequence.SequenceID]; exists && owner != request.Sequence.OwnerID {
			return workerproc.PersistentResponse{}, &workerproc.WorkerResponseError{
				RequestID: request.RequestID, Message: "sequence is already open",
			}
		}
		worker.sequences[request.Sequence.SequenceID] = request.Sequence.OwnerID
	case "closeSequence":
		if request.Sequence == nil {
			return workerproc.PersistentResponse{}, errors.New("missing sequence request")
		}
		owner, exists := worker.sequences[request.Sequence.SequenceID]
		if !exists {
			return workerproc.PersistentResponse{}, &workerproc.WorkerResponseError{
				RequestID: request.RequestID, Message: "sequence is not open",
			}
		}
		if request.Sequence.OwnerID != "" && owner != request.Sequence.OwnerID {
			return workerproc.PersistentResponse{}, &workerproc.WorkerResponseError{
				RequestID: request.RequestID, Message: "sequence is owned by another request",
			}
		}
		delete(worker.sequences, request.Sequence.SequenceID)
	case "prefill", "decode":
		if worker.failCommand == request.Command && worker.inferErr != nil {
			return workerproc.PersistentResponse{}, worker.inferErr
		}
		if request.Forward == nil {
			return workerproc.PersistentResponse{}, errors.New("missing forward request")
		}
		forward := request.Forward
		output := workerproc.WireTensor{
			Shape: []int{1, 1, 1}, DType: "float32", Data: []byte{0, 0, 0, 0},
		}
		if worker.terminal {
			token := worker.tokens[min(worker.logitIndex, len(worker.tokens)-1)]
			values := []float32{0, 0, 0, 0, 0}
			values[token] = 10
			output = plannedFloat32Tensor([]int{1, 1, len(values)}, values)
			worker.logitIndex++
			if forward.ReturnSampledToken {
				resultToken := token
				result.Forward = &workerproc.PersistentForwardResult{
					ShardID: forward.ShardID, SequenceID: forward.SequenceID,
					Operation: request.Command, Position: forward.Position,
					SampledTokenID: &resultToken, ComputeMicros: 1, KVCacheBytes: 1,
				}
			}
		} else {
			data := append([]byte(nil), forward.Input.Data...)
			data = append(data, byte(len(worker.name)))
			output = workerproc.WireTensor{Shape: []int{1, len(data)}, DType: "uint8", Data: data}
		}
		if result.Forward == nil {
			result.Forward = &workerproc.PersistentForwardResult{
				ShardID: forward.ShardID, SequenceID: forward.SequenceID,
				Operation: request.Command, Position: forward.Position,
				Output: output, ComputeMicros: 1, KVCacheBytes: 1,
			}
		}
		if request.Command == "prefill" {
			result.Forward.NextPosition = uint64(max(1, len(forward.Input.Shape)))
		} else {
			result.Forward.NextPosition = forward.Position + 1
		}
	default:
		return workerproc.PersistentResponse{}, fmt.Errorf("unexpected command %q", request.Command)
	}
	return workerproc.PersistentResponse{OK: true, Result: result}, nil
}

func plannedFloat32Tensor(shape []int, values []float32) workerproc.WireTensor {
	data := make([]byte, len(values)*4)
	for index, value := range values {
		binary.LittleEndian.PutUint32(data[index*4:], math.Float32bits(value))
	}
	return workerproc.WireTensor{Shape: shape, DType: "float32", Data: data}
}

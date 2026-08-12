package failureharness

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"syscall"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

type faultWorker struct {
	role         string
	fault        string
	marker       string
	shards       map[string]workerproc.PersistentShardSnapshot
	sequences    map[string]map[string]*faultSequence
	loadCount    int
	forwardCount int
}

type faultSequence struct {
	ownerID   string
	position  uint64
	prefilled bool
}

func ServeFaultWorker(args []string, input io.Reader, output io.Writer) error {
	flags := flag.NewFlagSet("fault-worker", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	role := flags.String("role", "consumer", "worker role")
	fault := flags.String("fault", "", "fault mode")
	marker := flags.String("marker", "", "one-shot fault marker")
	if err := flags.Parse(args); err != nil {
		return err
	}
	worker := &faultWorker{
		role: *role, fault: *fault, marker: *marker,
		shards:    make(map[string]workerproc.PersistentShardSnapshot),
		sequences: make(map[string]map[string]*faultSequence),
	}
	scanner := bufio.NewScanner(input)
	scanner.Buffer(make([]byte, 64*1024), 16<<20)
	encoder := json.NewEncoder(output)
	for scanner.Scan() {
		var request workerproc.PersistentRequest
		if err := json.Unmarshal(scanner.Bytes(), &request); err != nil {
			return err
		}
		result, shouldStop, err := worker.handle(request)
		response := workerproc.PersistentResponse{RequestID: request.RequestID}
		if err != nil {
			response.Error = err.Error()
		} else {
			response.OK = true
			response.Result = result
		}
		if err := encoder.Encode(response); err != nil {
			return err
		}
		if shouldStop {
			return nil
		}
	}
	return scanner.Err()
}

func (worker *faultWorker) handle(
	request workerproc.PersistentRequest,
) (*workerproc.PersistentWorkerResult, bool, error) {
	switch request.Command {
	case "health":
		return &workerproc.PersistentWorkerResult{Status: "ok"}, false, nil
	case "modelInfo":
		return &workerproc.PersistentWorkerResult{Model: &workerproc.PersistentModelResult{
			ModelID: faultModelID, ModelType: "fault", LayerCount: 2,
			CheckpointFingerprint: "fault-checkpoint", CheckpointBytes: 1,
		}}, false, nil
	case "state":
		state := worker.state()
		return &workerproc.PersistentWorkerResult{State: &state}, false, nil
	case "loadShard":
		load := request.LoadShard
		if load == nil {
			return nil, false, errors.New("loadShard payload is missing")
		}
		snapshot := workerproc.PersistentShardSnapshot{
			ShardID: load.ShardID, ModelID: load.ModelID, ModelType: "fault",
			CheckpointFingerprint: "fault-checkpoint",
			LayerStart:            load.LayerStart, LayerEnd: load.LayerEnd,
			OwnsInput: load.OwnsInput, OwnsOutput: load.OwnsOutput,
			WeightKeyCount: 1,
		}
		worker.shards[load.ShardID] = snapshot
		worker.sequences[load.ShardID] = make(map[string]*faultSequence)
		worker.loadCount++
		return &workerproc.PersistentWorkerResult{Shard: &snapshot}, false, nil
	case "openSequence":
		sequence := request.Sequence
		if sequence == nil {
			return nil, false, errors.New("openSequence payload is missing")
		}
		sequences, ok := worker.sequences[sequence.ShardID]
		if !ok {
			return nil, false, fmt.Errorf("shard %s is not loaded", sequence.ShardID)
		}
		sequences[sequence.SequenceID] = &faultSequence{ownerID: sequence.OwnerID}
		state := worker.state()
		return &workerproc.PersistentWorkerResult{State: &state}, false, nil
	case "closeSequence":
		sequence := request.Sequence
		if sequence == nil {
			return nil, false, errors.New("closeSequence payload is missing")
		}
		sequences, ok := worker.sequences[sequence.ShardID]
		if !ok {
			return nil, false, fmt.Errorf("shard %s is not loaded", sequence.ShardID)
		}
		if _, ok := sequences[sequence.SequenceID]; !ok {
			return nil, false, fmt.Errorf(
				"sequence %s is not open on shard %s", sequence.SequenceID, sequence.ShardID,
			)
		}
		delete(sequences, sequence.SequenceID)
		state := worker.state()
		return &workerproc.PersistentWorkerResult{State: &state}, false, nil
	case "tokenize":
		eos := int32(1)
		return &workerproc.PersistentWorkerResult{Text: &workerproc.PersistentTextResult{
			ModelID: faultModelID, TokenIDs: []int32{2, 4}, EOSTokenID: &eos,
		}}, false, nil
	case "detokenize":
		text := "fault harness"
		return &workerproc.PersistentWorkerResult{Text: &workerproc.PersistentTextResult{
			ModelID: faultModelID, Text: &text,
		}}, false, nil
	case "prefill", "decode", "forward":
		result, err := worker.forward(request)
		return &workerproc.PersistentWorkerResult{Forward: result}, false, err
	case "shutdown":
		return &workerproc.PersistentWorkerResult{Shutdown: true}, true, nil
	default:
		return nil, false, fmt.Errorf("unknown command %s", request.Command)
	}
}

func (worker *faultWorker) forward(
	request workerproc.PersistentRequest,
) (*workerproc.PersistentForwardResult, error) {
	if request.DeadlineUnixMillis <= time.Now().UnixMilli() {
		return nil, errors.New("inference deadline is missing or expired")
	}
	forward := request.Forward
	if forward == nil {
		return nil, errors.New("forward payload is missing")
	}
	sequences, ok := worker.sequences[forward.ShardID]
	if !ok {
		return nil, fmt.Errorf("shard %s is not loaded", forward.ShardID)
	}
	sequence, ok := sequences[forward.SequenceID]
	if !ok {
		return nil, fmt.Errorf("sequence %s is not open on shard %s", forward.SequenceID, forward.ShardID)
	}
	if request.Command == "decode" && worker.role == "consumer" {
		if err := worker.injectFault(); err != nil {
			return nil, err
		}
	}
	inputLength := uint64(1)
	if len(forward.Input.Shape) > 1 {
		inputLength = uint64(forward.Input.Shape[1])
	}
	if request.Command == "prefill" {
		sequence.position = inputLength
		sequence.prefilled = true
	} else if request.Command == "decode" {
		if !sequence.prefilled || forward.Position != sequence.position {
			return nil, errors.New("invalid decode position")
		}
		sequence.position++
	}
	worker.forwardCount++
	output := hiddenTensor(int(inputLength))
	if worker.role != "producer" {
		output = logitsTensor(3)
	}
	return &workerproc.PersistentForwardResult{
		ShardID: forward.ShardID, SequenceID: forward.SequenceID,
		Operation: request.Command, Position: forward.Position,
		NextPosition: sequence.position, Output: output,
		ComputeMicros: 100, KVCacheBytes: int(sequence.position) * 16,
	}, nil
}

func (worker *faultWorker) injectFault() error {
	if worker.fault == "" {
		return nil
	}
	if worker.fault == "jitter" {
		delay := time.Duration(2+(worker.forwardCount%3)*3) * time.Millisecond
		time.Sleep(delay)
		return nil
	}
	if worker.marker != "" {
		file, err := os.OpenFile(worker.marker, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if errors.Is(err, os.ErrExist) {
			return nil
		}
		if err != nil {
			return err
		}
		_ = file.Close()
	}
	switch worker.fault {
	case "delay":
		time.Sleep(5 * time.Second)
	case "pause":
		if err := syscall.Kill(os.Getpid(), syscall.SIGSTOP); err != nil {
			return err
		}
	case "kill":
		os.Exit(70)
	default:
		return fmt.Errorf("unknown fault %s", worker.fault)
	}
	return nil
}

func (worker *faultWorker) state() workerproc.PersistentWorkerState {
	shards := make([]workerproc.PersistentShardSnapshot, 0, len(worker.shards))
	totalKV := 0
	for shardID, snapshot := range worker.shards {
		sequences := worker.sequences[shardID]
		snapshot.OpenSequenceCount = len(sequences)
		for _, sequence := range sequences {
			snapshot.KVCacheBytes += int(sequence.position) * 16
		}
		snapshot.RetainedBytes = snapshot.KVCacheBytes
		totalKV += snapshot.KVCacheBytes
		shards = append(shards, snapshot)
	}
	return workerproc.PersistentWorkerState{
		LoadedShards: shards, LoadCount: worker.loadCount,
		ForwardCount: worker.forwardCount, KVCacheBytes: totalKV,
		RetainedBytes: totalKV, RetainedByteBudget: 1 << 20,
	}
}

func hiddenTensor(length int) workerproc.WireTensor {
	return workerproc.WireTensor{
		Shape: []int{1, length, 2}, DType: "float32", Data: make([]byte, length*2*4),
	}
}

func logitsTensor(token int) workerproc.WireTensor {
	values := []float32{0, 0, 0, 0, 0}
	values[token] = 10
	data := make([]byte, len(values)*4)
	for index, value := range values {
		binary.LittleEndian.PutUint32(data[index*4:], math.Float32bits(value))
	}
	return workerproc.WireTensor{Shape: []int{1, 1, len(values)}, DType: "float32", Data: data}
}

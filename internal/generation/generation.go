package generation

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

const cleanupTimeout = 10 * time.Second

type SessionConfig struct {
	Model string
	RTol  float64
	ATol  float64
}

type Request struct {
	Prompt     string
	MaxTokens  int
	SequenceID string
}

type Shard struct {
	ID         string `json:"id"`
	LayerStart int    `json:"layerStart"`
	LayerEnd   int    `json:"layerEnd"`
}

type ShardPlan struct {
	Producer Shard `json:"producer"`
	Consumer Shard `json:"consumer"`
}

type Timing struct {
	SessionSetupMicros     int64  `json:"sessionSetupMicros"`
	TokenizeMicros         int64  `json:"tokenizeMicros"`
	PrefillMicros          int64  `json:"prefillMicros"`
	DecodeMicros           int64  `json:"decodeMicros"`
	DetokenizeMicros       int64  `json:"detokenizeMicros"`
	TotalMicros            int64  `json:"totalMicros"`
	ProducerComputeMicros  uint64 `json:"producerComputeMicros"`
	ConsumerComputeMicros  uint64 `json:"consumerComputeMicros"`
	ReferenceComputeMicros uint64 `json:"referenceComputeMicros,omitempty"`
}

type Verification struct {
	GreedyTokenIDsMatch   bool    `json:"greedyTokenIDsMatch"`
	ComparedTokens        int     `json:"comparedTokens"`
	MaxAbsoluteDifference float64 `json:"maxAbsoluteDifference"`
	MaxRelativeDifference float64 `json:"maxRelativeDifference"`
}

type Result struct {
	Model                string        `json:"model"`
	ModelType            string        `json:"modelType"`
	ShardPlan            ShardPlan     `json:"shardPlan"`
	SequenceID           string        `json:"sequenceID"`
	Prompt               string        `json:"prompt"`
	PromptTokenIDs       []int32       `json:"promptTokenIDs"`
	GeneratedTokenIDs    []int32       `json:"generatedTokenIDs"`
	Text                 string        `json:"text"`
	MaxTokens            int           `json:"maxTokens"`
	StopReason           string        `json:"stopReason"`
	EOSTokenID           *int32        `json:"eosTokenID,omitempty"`
	RTol                 float64       `json:"rtol"`
	ATol                 float64       `json:"atol"`
	ProducerKVCacheBytes int           `json:"producerKVCacheBytes"`
	ConsumerKVCacheBytes int           `json:"consumerKVCacheBytes"`
	Timing               Timing        `json:"timing"`
	Verification         *Verification `json:"verification,omitempty"`
}

type Session struct {
	producer       workerproc.PersistentCaller
	consumer       workerproc.PersistentCaller
	reference      workerproc.PersistentCaller
	config         SessionConfig
	model          workerproc.PersistentModelResult
	plan           ShardPlan
	referenceShard string
	setupMicros    int64
}

func NewSession(
	ctx context.Context,
	producer workerproc.PersistentCaller,
	consumer workerproc.PersistentCaller,
	reference workerproc.PersistentCaller,
	config SessionConfig,
) (*Session, error) {
	started := time.Now()
	if producer == nil || consumer == nil {
		return nil, errors.New("producer and consumer callers are required")
	}
	if config.Model == "" {
		return nil, errors.New("model is required")
	}
	if config.RTol < 0 || config.ATol < 0 {
		return nil, errors.New("numeric tolerances must be non-negative")
	}

	producerModel, err := modelInfo(ctx, producer, config.Model)
	if err != nil {
		return nil, fmt.Errorf("producer model info: %w", err)
	}
	consumerModel, err := modelInfo(ctx, consumer, config.Model)
	if err != nil {
		return nil, fmt.Errorf("consumer model info: %w", err)
	}
	if err := matchModel(producerModel, consumerModel, "consumer"); err != nil {
		return nil, err
	}
	if producerModel.LayerCount < 2 {
		return nil, fmt.Errorf("model %s has %d layers; at least two are required", config.Model, producerModel.LayerCount)
	}

	split := producerModel.LayerCount / 2
	modelHash := sha256.Sum256([]byte(config.Model))
	suffix := hex.EncodeToString(modelHash[:6])
	plan := ShardPlan{
		Producer: Shard{ID: "generate-producer-" + suffix, LayerStart: 0, LayerEnd: split},
		Consumer: Shard{ID: "generate-consumer-" + suffix, LayerStart: split, LayerEnd: producerModel.LayerCount},
	}
	if _, err := ensureShard(ctx, producer, workerproc.PersistentLoadShardRequest{
		ModelID: config.Model, ShardID: plan.Producer.ID,
		LayerStart: plan.Producer.LayerStart, LayerEnd: plan.Producer.LayerEnd, OwnsInput: true,
	}); err != nil {
		return nil, fmt.Errorf("producer shard: %w", err)
	}
	if _, err := ensureShard(ctx, consumer, workerproc.PersistentLoadShardRequest{
		ModelID: config.Model, ShardID: plan.Consumer.ID,
		LayerStart: plan.Consumer.LayerStart, LayerEnd: plan.Consumer.LayerEnd, OwnsOutput: true,
	}); err != nil {
		return nil, fmt.Errorf("consumer shard: %w", err)
	}

	referenceShard := ""
	if reference != nil {
		referenceModel, infoErr := modelInfo(ctx, reference, config.Model)
		if infoErr != nil {
			return nil, fmt.Errorf("reference model info: %w", infoErr)
		}
		if err := matchModel(producerModel, referenceModel, "reference"); err != nil {
			return nil, err
		}
		referenceShard = "generate-reference-" + suffix
		if _, err := ensureShard(ctx, reference, workerproc.PersistentLoadShardRequest{
			ModelID: config.Model, ShardID: referenceShard,
			LayerStart: 0, LayerEnd: producerModel.LayerCount, OwnsInput: true, OwnsOutput: true,
		}); err != nil {
			return nil, fmt.Errorf("reference shard: %w", err)
		}
	}

	return &Session{
		producer: producer, consumer: consumer, reference: reference,
		config: config, model: *producerModel, plan: plan, referenceShard: referenceShard,
		setupMicros: time.Since(started).Microseconds(),
	}, nil
}

func (s *Session) Generate(ctx context.Context, request Request) (result Result, returnErr error) {
	started := time.Now()
	defer func() { result.Timing.TotalMicros = time.Since(started).Microseconds() }()
	if err := ctx.Err(); err != nil {
		return Result{}, err
	}
	if request.Prompt == "" {
		return Result{}, errors.New("prompt is required")
	}
	if request.MaxTokens <= 0 {
		return Result{}, errors.New("max tokens must be positive")
	}
	if request.SequenceID == "" {
		sequenceID, err := randomSequenceID()
		if err != nil {
			return Result{}, err
		}
		request.SequenceID = sequenceID
	}

	result = Result{
		Model: s.config.Model, ModelType: s.model.ModelType, ShardPlan: s.plan,
		SequenceID: request.SequenceID, Prompt: request.Prompt, MaxTokens: request.MaxTokens,
		RTol: s.config.RTol, ATol: s.config.ATol,
		Timing: Timing{SessionSetupMicros: s.setupMicros},
	}

	tokenizeStarted := time.Now()
	tokenized, err := tokenize(ctx, s.producer, s.config.Model, request.Prompt)
	result.Timing.TokenizeMicros = time.Since(tokenizeStarted).Microseconds()
	if err != nil {
		return result, fmt.Errorf("tokenize prompt: %w", err)
	}
	if len(tokenized.TokenIDs) == 0 {
		return result, errors.New("tokenizer returned no prompt tokens")
	}
	result.PromptTokenIDs = tokenized.TokenIDs
	result.EOSTokenID = tokenized.EOSTokenID

	targets := []sequenceTarget{
		{name: "producer", caller: s.producer, shardID: s.plan.Producer.ID},
		{name: "consumer", caller: s.consumer, shardID: s.plan.Consumer.ID},
	}
	if s.reference != nil {
		targets = append(targets, sequenceTarget{name: "reference", caller: s.reference, shardID: s.referenceShard})
	}
	opened := make([]sequenceTarget, 0, len(targets))
	defer func() {
		cleanupErr := closeSequences(opened, request.SequenceID)
		if cleanupErr != nil {
			if returnErr == nil {
				returnErr = cleanupErr
			} else {
				returnErr = errors.Join(returnErr, cleanupErr)
			}
		}
	}()
	for _, target := range targets {
		// Once an open is attempted its outcome can be ambiguous: the worker may
		// apply the mutation after the caller's context expires but before the
		// response is observed. Conservatively close every attempted target.
		opened = append(opened, target)
		if err := sequenceCommand(ctx, target.caller, "openSequence", target.shardID, request.SequenceID); err != nil {
			return result, fmt.Errorf("open %s sequence: %w", target.name, err)
		}
	}

	prompt := tokenTensor(tokenized.TokenIDs)
	prefillStarted := time.Now()
	producerResult, err := infer(
		ctx, s.producer, "prefill", s.plan.Producer.ID, request.SequenceID, 0, "tokens", prompt,
	)
	if err != nil {
		return result, fmt.Errorf("producer prefill: %w", err)
	}
	consumerResult, err := infer(
		ctx, s.consumer, "prefill", s.plan.Consumer.ID, request.SequenceID, 0, "hidden", producerResult.Output,
	)
	if err != nil {
		return result, fmt.Errorf("consumer prefill: %w", err)
	}
	var referenceResult *workerproc.PersistentForwardResult
	if s.reference != nil {
		referenceResult, err = infer(
			ctx, s.reference, "prefill", s.referenceShard, request.SequenceID, 0, "tokens", prompt,
		)
		if err != nil {
			return result, fmt.Errorf("reference prefill: %w", err)
		}
	}
	result.Timing.PrefillMicros = time.Since(prefillStarted).Microseconds()
	result.Timing.ProducerComputeMicros += producerResult.ComputeMicros
	result.Timing.ConsumerComputeMicros += consumerResult.ComputeMicros
	result.ProducerKVCacheBytes = producerResult.KVCacheBytes
	result.ConsumerKVCacheBytes = consumerResult.KVCacheBytes
	if referenceResult != nil {
		result.Timing.ReferenceComputeMicros += referenceResult.ComputeMicros
		result.Verification = &Verification{GreedyTokenIDsMatch: true}
	}

	position := uint64(len(tokenized.TokenIDs))
	distributedLogits := consumerResult.Output
	var referenceLogits workerproc.WireTensor
	if referenceResult != nil {
		referenceLogits = referenceResult.Output
	}
	decodeStarted := time.Now()
	for len(result.GeneratedTokenIDs) < request.MaxTokens {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		nextToken, err := greedyToken(distributedLogits)
		if err != nil {
			return result, fmt.Errorf("sample distributed logits: %w", err)
		}
		if result.Verification != nil {
			absolute, relative, compareErr := compareFinalLogits(
				distributedLogits, referenceLogits, s.config.RTol, s.config.ATol,
			)
			if compareErr != nil {
				return result, fmt.Errorf("generation step %d logits: %w", len(result.GeneratedTokenIDs), compareErr)
			}
			result.Verification.MaxAbsoluteDifference = math.Max(
				result.Verification.MaxAbsoluteDifference, absolute,
			)
			result.Verification.MaxRelativeDifference = math.Max(
				result.Verification.MaxRelativeDifference, relative,
			)
			referenceToken, sampleErr := greedyToken(referenceLogits)
			if sampleErr != nil {
				return result, fmt.Errorf("sample reference logits: %w", sampleErr)
			}
			if nextToken != referenceToken {
				result.Verification.GreedyTokenIDsMatch = false
				return result, fmt.Errorf(
					"generation step %d chose token %d; reference chose %d",
					len(result.GeneratedTokenIDs), nextToken, referenceToken,
				)
			}
			result.Verification.ComparedTokens++
		}
		result.GeneratedTokenIDs = append(result.GeneratedTokenIDs, nextToken)
		if result.EOSTokenID != nil && nextToken == *result.EOSTokenID {
			result.StopReason = "eos"
			break
		}
		if len(result.GeneratedTokenIDs) == request.MaxTokens {
			result.StopReason = "max_tokens"
			break
		}

		token := tokenTensor([]int32{nextToken})
		producerResult, err = infer(
			ctx, s.producer, "decode", s.plan.Producer.ID, request.SequenceID, position, "tokens", token,
		)
		if err != nil {
			return result, fmt.Errorf("producer decode step %d: %w", len(result.GeneratedTokenIDs), err)
		}
		consumerResult, err = infer(
			ctx, s.consumer, "decode", s.plan.Consumer.ID, request.SequenceID, position, "hidden", producerResult.Output,
		)
		if err != nil {
			return result, fmt.Errorf("consumer decode step %d: %w", len(result.GeneratedTokenIDs), err)
		}
		result.Timing.ProducerComputeMicros += producerResult.ComputeMicros
		result.Timing.ConsumerComputeMicros += consumerResult.ComputeMicros
		result.ProducerKVCacheBytes = producerResult.KVCacheBytes
		result.ConsumerKVCacheBytes = consumerResult.KVCacheBytes
		distributedLogits = consumerResult.Output
		if s.reference != nil {
			referenceResult, err = infer(
				ctx, s.reference, "decode", s.referenceShard, request.SequenceID, position, "tokens", token,
			)
			if err != nil {
				return result, fmt.Errorf("reference decode step %d: %w", len(result.GeneratedTokenIDs), err)
			}
			result.Timing.ReferenceComputeMicros += referenceResult.ComputeMicros
			referenceLogits = referenceResult.Output
		}
		position++
	}
	result.Timing.DecodeMicros = time.Since(decodeStarted).Microseconds()

	detokenizeStarted := time.Now()
	text, err := detokenize(ctx, s.producer, s.config.Model, result.GeneratedTokenIDs)
	result.Timing.DetokenizeMicros = time.Since(detokenizeStarted).Microseconds()
	if err != nil {
		return result, fmt.Errorf("detokenize generated tokens: %w", err)
	}
	result.Text = text
	return result, nil
}

type sequenceTarget struct {
	name    string
	caller  workerproc.PersistentCaller
	shardID string
}

func closeSequences(targets []sequenceTarget, sequenceID string) error {
	var result error
	for index := len(targets) - 1; index >= 0; index-- {
		target := targets[index]
		ctx, cancel := context.WithTimeout(context.Background(), cleanupTimeout)
		if err := sequenceCommand(ctx, target.caller, "closeSequence", target.shardID, sequenceID); err != nil {
			result = errors.Join(result, fmt.Errorf("close %s sequence: %w", target.name, err))
		}
		cancel()
	}
	return result
}

func modelInfo(
	ctx context.Context,
	caller workerproc.PersistentCaller,
	modelID string,
) (*workerproc.PersistentModelResult, error) {
	response, err := call(ctx, caller, workerproc.PersistentRequest{
		Command: "modelInfo", Model: &workerproc.PersistentModelRequest{ModelID: modelID},
	})
	if err != nil {
		return nil, err
	}
	if response.Result == nil || response.Result.Model == nil {
		return nil, errors.New("modelInfo returned no model metadata")
	}
	if response.Result.Model.ModelID != modelID || response.Result.Model.ModelType == "" ||
		response.Result.Model.LayerCount <= 0 {
		return nil, fmt.Errorf("modelInfo returned invalid metadata: %+v", *response.Result.Model)
	}
	return response.Result.Model, nil
}

func matchModel(expected, actual *workerproc.PersistentModelResult, role string) error {
	if expected.ModelID != actual.ModelID || expected.ModelType != actual.ModelType ||
		expected.LayerCount != actual.LayerCount {
		return fmt.Errorf(
			"%s model mismatch: producer=%+v %s=%+v", role, *expected, role, *actual,
		)
	}
	return nil
}

func ensureShard(
	ctx context.Context,
	caller workerproc.PersistentCaller,
	request workerproc.PersistentLoadShardRequest,
) (*workerproc.PersistentShardSnapshot, error) {
	state, err := workerState(ctx, caller)
	if err != nil {
		return nil, err
	}
	for index := range state.LoadedShards {
		shard := &state.LoadedShards[index]
		if shard.ShardID != request.ShardID {
			continue
		}
		if shard.ModelID != request.ModelID || shard.LayerStart != request.LayerStart ||
			shard.LayerEnd != request.LayerEnd || shard.OwnsInput != request.OwnsInput ||
			shard.OwnsOutput != request.OwnsOutput {
			return nil, fmt.Errorf("loaded shard %s does not match requested plan", request.ShardID)
		}
		return shard, nil
	}
	response, err := call(ctx, caller, workerproc.PersistentRequest{Command: "loadShard", LoadShard: &request})
	if err != nil {
		return nil, err
	}
	if response.Result == nil || response.Result.Shard == nil {
		return nil, errors.New("loadShard returned no shard snapshot")
	}
	return response.Result.Shard, nil
}

func tokenize(
	ctx context.Context,
	caller workerproc.PersistentCaller,
	modelID string,
	prompt string,
) (*workerproc.PersistentTextResult, error) {
	addSpecialTokens := true
	response, err := call(ctx, caller, workerproc.PersistentRequest{
		Command: "tokenize",
		Text: &workerproc.PersistentTextRequest{
			ModelID: modelID, Text: &prompt, AddSpecialTokens: &addSpecialTokens,
		},
	})
	if err != nil {
		return nil, err
	}
	if response.Result == nil || response.Result.Text == nil {
		return nil, errors.New("tokenize returned no tokenization result")
	}
	if response.Result.Text.ModelID != modelID {
		return nil, fmt.Errorf(
			"tokenize returned model %q, want %q", response.Result.Text.ModelID, modelID,
		)
	}
	for _, token := range response.Result.Text.TokenIDs {
		if token < 0 {
			return nil, fmt.Errorf("tokenize returned negative token ID %d", token)
		}
	}
	return response.Result.Text, nil
}

func detokenize(
	ctx context.Context,
	caller workerproc.PersistentCaller,
	modelID string,
	tokens []int32,
) (string, error) {
	skipSpecialTokens := true
	response, err := call(ctx, caller, workerproc.PersistentRequest{
		Command: "detokenize",
		Text: &workerproc.PersistentTextRequest{
			ModelID: modelID, TokenIDs: tokens, SkipSpecialTokens: &skipSpecialTokens,
		},
	})
	if err != nil {
		return "", err
	}
	if response.Result == nil || response.Result.Text == nil || response.Result.Text.Text == nil {
		return "", errors.New("detokenize returned no text")
	}
	if response.Result.Text.ModelID != modelID {
		return "", fmt.Errorf(
			"detokenize returned model %q, want %q", response.Result.Text.ModelID, modelID,
		)
	}
	return *response.Result.Text.Text, nil
}

func workerState(
	ctx context.Context,
	caller workerproc.PersistentCaller,
) (*workerproc.PersistentWorkerState, error) {
	response, err := call(ctx, caller, workerproc.PersistentRequest{Command: "state"})
	if err != nil {
		return nil, err
	}
	if response.Result == nil || response.Result.State == nil {
		return nil, errors.New("state returned no worker snapshot")
	}
	return response.Result.State, nil
}

func sequenceCommand(
	ctx context.Context,
	caller workerproc.PersistentCaller,
	command string,
	shardID string,
	sequenceID string,
) error {
	_, err := call(ctx, caller, workerproc.PersistentRequest{
		Command:  command,
		Sequence: &workerproc.PersistentSequenceRequest{ShardID: shardID, SequenceID: sequenceID},
	})
	return err
}

func infer(
	ctx context.Context,
	caller workerproc.PersistentCaller,
	command string,
	shardID string,
	sequenceID string,
	position uint64,
	inputKind string,
	input workerproc.WireTensor,
) (*workerproc.PersistentForwardResult, error) {
	response, err := call(ctx, caller, workerproc.PersistentRequest{
		Command: command,
		Forward: &workerproc.PersistentForwardRequest{
			ShardID: shardID, SequenceID: sequenceID, Position: position,
			InputKind: inputKind, Input: input,
		},
	})
	if err != nil {
		return nil, err
	}
	if response.Result == nil || response.Result.Forward == nil {
		return nil, fmt.Errorf("%s returned no inference result", command)
	}
	result := response.Result.Forward
	if result.Operation != command || result.Position != position || result.KVCacheBytes <= 0 {
		return nil, fmt.Errorf(
			"%s returned inconsistent metadata: operation=%s position=%d kv=%d",
			command, result.Operation, result.Position, result.KVCacheBytes,
		)
	}
	return result, nil
}

func call(
	ctx context.Context,
	caller workerproc.PersistentCaller,
	request workerproc.PersistentRequest,
) (workerproc.PersistentResponse, error) {
	response, err := caller.Call(ctx, request)
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
	return workerproc.WireTensor{Shape: []int{1, len(tokens)}, DType: "int32", Data: data}
}

func randomSequenceID() (string, error) {
	bytes := make([]byte, 12)
	if _, err := rand.Read(bytes); err != nil {
		return "", fmt.Errorf("generate sequence ID: %w", err)
	}
	return "generation-" + hex.EncodeToString(bytes), nil
}

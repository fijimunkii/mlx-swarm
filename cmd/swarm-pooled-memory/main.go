package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/pooledproof"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	createReference := flag.Bool("create-reference", false, "run a separate local full-model oracle and emit a reference manifest")
	referencePath := flag.String("reference", "testdata/pooled-memory/gemma-3-12b-it-4bit.json", "checked-in deterministic reference manifest")
	worker := flag.String("worker", workerproc.DefaultPath(), "path to the built MLXWorker executable for reference creation")
	producerURL := flag.String("producer", "", "producer swarmd base URL")
	consumerURL := flag.String("peer", "", "consumer swarmd base URL")
	model := flag.String("model", pooledproof.DefaultModelID, "checkpoint model ID for reference creation")
	prompt := flag.String("prompt", pooledproof.DefaultPrompt, "deterministic prompt for reference creation")
	maxTokens := flag.Int("max-tokens", pooledproof.DefaultMinimumTokens, "reference tokens to generate")
	minimumTokens := flag.Int("minimum-tokens", pooledproof.DefaultMinimumTokens, "minimum proof token count")
	memoryThreshold := flag.Int("memory-threshold-bytes", pooledproof.DefaultWorkerMemoryThreshold, "required MLX scheduling threshold on each serving worker")
	rtol := flag.Float64("rtol", 1e-4, "relative reference-logit tolerance when creating a reference")
	atol := flag.Float64("atol", 1e-4, "absolute reference-logit tolerance when creating a reference")
	forwardTimeout := flag.Duration("forward-timeout", 2*time.Minute, "deadline for each prefill/decode request")
	timeout := flag.Duration("timeout", 30*time.Minute, "overall proof timeout")
	flag.Parse()

	if *timeout <= 0 {
		return errors.New("-timeout must be positive")
	}
	if *forwardTimeout < time.Millisecond {
		return errors.New("-forward-timeout must be at least 1ms")
	}
	signalContext, stopSignals := signal.NotifyContext(
		context.Background(), os.Interrupt, syscall.SIGTERM,
	)
	defer stopSignals()
	ctx, cancel := context.WithTimeout(signalContext, *timeout)
	defer cancel()

	if *createReference {
		reference, err := pooledproof.CreateReference(ctx, pooledproof.ReferenceConfig{
			WorkerPath: *worker, Model: *model, Prompt: *prompt, MaxTokens: *maxTokens,
			RTol: *rtol, ATol: *atol, ForwardTimeout: *forwardTimeout,
		})
		if err != nil {
			return err
		}
		return encode(reference)
	}
	if *producerURL == "" || *consumerURL == "" {
		return errors.New("-producer and -peer are required for the serving proof")
	}
	reference, err := pooledproof.LoadReference(*referencePath)
	if err != nil {
		return err
	}
	result, proofErr := pooledproof.Run(ctx, reference, pooledproof.RunConfig{
		ProducerURL: *producerURL, ConsumerURL: *consumerURL,
		ExpectedMemoryThresholdBytes: *memoryThreshold, MinimumGeneratedTokens: *minimumTokens,
		RTol: *rtol, ATol: *atol, ForwardTimeout: *forwardTimeout,
	})
	if err := encode(result); err != nil {
		return err
	}
	return proofErr
}

func encode(value any) error {
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		return fmt.Errorf("encode pooled-memory result: %w", err)
	}
	return nil
}

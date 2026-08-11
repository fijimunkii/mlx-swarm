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

	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

const defaultModelID = "mlx-community/gemma-3-270m-it-4bit"

type managedCaller struct {
	caller workerproc.PersistentCaller
	direct *workerproc.PersistentClient
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	worker := flag.String("worker", workerproc.DefaultPath(), "path to the built MLXWorker executable")
	producerURL := flag.String("producer", "", "optional producer swarmd base URL; empty starts a local worker")
	consumerURL := flag.String("peer", "", "optional consumer swarmd base URL; empty starts a local worker")
	model := flag.String("model", defaultModelID, "checkpoint model ID")
	prompt := flag.String("prompt", "", "prompt text to generate from")
	maxTokens := flag.Int("max-tokens", 32, "maximum number of new tokens")
	sequenceID := flag.String("sequence", "", "optional sequence ID; generated when empty")
	verify := flag.Bool("verify", false, "compare every greedy step with a full-model cached reference")
	rtol := flag.Float64("rtol", 1e-4, "relative reference-logit tolerance")
	atol := flag.Float64("atol", 1e-4, "absolute reference-logit tolerance")
	timeout := flag.Duration("timeout", 10*time.Minute, "overall generation timeout")
	flag.Parse()

	if *prompt == "" {
		return errors.New("-prompt is required")
	}
	if *maxTokens <= 0 {
		return errors.New("-max-tokens must be positive")
	}
	if *timeout <= 0 {
		return errors.New("-timeout must be positive")
	}

	signalContext, stopSignals := signal.NotifyContext(
		context.Background(), os.Interrupt, syscall.SIGTERM,
	)
	defer stopSignals()
	ctx, cancel := context.WithTimeout(signalContext, *timeout)
	defer cancel()

	producer, err := openCaller(*worker, *producerURL)
	if err != nil {
		return fmt.Errorf("producer: %w", err)
	}
	defer producer.close()
	consumer, err := openCaller(*worker, *consumerURL)
	if err != nil {
		return fmt.Errorf("consumer: %w", err)
	}
	defer consumer.close()

	var reference *managedCaller
	if *verify {
		reference, err = openCaller(*worker, "")
		if err != nil {
			return fmt.Errorf("reference: %w", err)
		}
		defer reference.close()
	}
	var referenceCaller workerproc.PersistentCaller
	if reference != nil {
		referenceCaller = reference.caller
	}

	session, err := generation.NewSession(
		ctx,
		producer.caller,
		consumer.caller,
		referenceCaller,
		generation.SessionConfig{Model: *model, RTol: *rtol, ATol: *atol},
	)
	if err != nil {
		return fmt.Errorf("prepare generation session: %w", err)
	}
	result, err := session.Generate(ctx, generation.Request{
		Prompt: *prompt, MaxTokens: *maxTokens, SequenceID: *sequenceID,
	})
	if err != nil {
		return fmt.Errorf("generate: %w", err)
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		return fmt.Errorf("encode generation result: %w", err)
	}
	fmt.Println(string(encoded))
	return nil
}

func openCaller(worker string, endpoint string) (*managedCaller, error) {
	if endpoint != "" {
		client, err := workerproc.NewHTTPPersistentClient(endpoint, nil)
		if err != nil {
			return nil, err
		}
		return &managedCaller{caller: client}, nil
	}
	client, err := workerproc.StartPersistent(worker)
	if err != nil {
		return nil, err
	}
	return &managedCaller{caller: client, direct: client}, nil
}

func (caller *managedCaller) close() {
	if caller == nil || caller.direct == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := caller.direct.Shutdown(ctx); err == nil {
		return
	}
	_ = caller.direct.Kill()
	_ = caller.direct.Wait(ctx)
}

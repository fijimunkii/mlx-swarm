package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/meshstress"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	config := meshstress.DefaultConfig()
	workers := flag.Int("workers", config.WorkerCount, "number of concurrently visible synthetic workers")
	searchLimit := flag.Uint64("max-search-operations", config.MaxSearchOperations, "placement work bound per decision")
	decisionLimit := flag.Duration("decision-bound", config.MaxDecisionDuration, "wall-time bound per placement decision")
	timeout := flag.Duration("timeout", 30*time.Second, "overall proof timeout")
	flag.Parse()
	if *timeout <= 0 {
		return fmt.Errorf("timeout must be positive")
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	result, proofErr := meshstress.Run(ctx, meshstress.Config{
		WorkerCount: *workers, MaxSearchOperations: *searchLimit,
		MaxDecisionDuration: *decisionLimit,
	})
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(result); err != nil {
		return fmt.Errorf("encode synthetic mesh proof: %w", err)
	}
	return proofErr
}

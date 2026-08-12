package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/failureharness"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "fault-worker" {
		if err := failureharness.ServeFaultWorker(os.Args[2:], os.Stdin, os.Stdout); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	forwardDeadline := flag.Duration("forward-deadline", 150*time.Millisecond, "deadline for each injected inference request")
	scenarioBound := flag.Duration("scenario-bound", 2*time.Second, "maximum allowed generation termination time")
	timeout := flag.Duration("timeout", 30*time.Second, "overall harness timeout")
	flag.Parse()
	if *forwardDeadline < time.Millisecond {
		return errors.New("-forward-deadline must be at least 1ms")
	}
	if *scenarioBound <= *forwardDeadline {
		return errors.New("-scenario-bound must exceed -forward-deadline")
	}
	if *timeout <= 0 {
		return errors.New("-timeout must be positive")
	}
	executable, err := os.Executable()
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	summary, err := failureharness.Run(ctx, executable, *forwardDeadline, *scenarioBound)
	if err != nil {
		return err
	}
	if !summary.AllTerminatedBounded || !summary.AllCleanupReleased || !summary.AllRecoveryReady {
		return errors.New("failure harness did not satisfy bounded cleanup and recovery invariants")
	}
	encoded, err := json.Marshal(summary)
	if err != nil {
		return err
	}
	fmt.Println(string(encoded))
	return nil
}

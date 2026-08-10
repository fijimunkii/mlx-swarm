package workerproc

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"time"
)

// Client executes the local MLX worker process. The swarm owns networking;
// Swift/MLX remains an isolated local execution backend.
type Client struct {
	Path string
}

type Result struct {
	Output   []byte
	Duration time.Duration
}

func DefaultPath() string {
	if path := os.Getenv("MLX_SWARM_WORKER"); path != "" {
		return path
	}
	return "worker/mlx/.build/xcode/Build/Products/Debug/MLXWorker"
}

func (c Client) Run(ctx context.Context, args []string, input []byte) (Result, error) {
	path := c.Path
	if path == "" {
		path = DefaultPath()
	}

	cmd := exec.CommandContext(ctx, path, args...)
	if input != nil {
		cmd.Stdin = bytes.NewReader(input)
	}

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	start := time.Now()
	err := cmd.Run()
	result := Result{Output: stdout.Bytes(), Duration: time.Since(start)}
	if err != nil {
		return result, fmt.Errorf("%w: %s", err, bytes.TrimSpace(stderr.Bytes()))
	}
	return result, nil
}

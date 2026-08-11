package workerproc

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"sync"
	"syscall"
)

const (
	maxPersistentResponseBytes = 128 << 20
	maxPersistentStderrBytes   = 64 << 10
)

var errPersistentWorkerStopped = errors.New("persistent worker is stopped")

type PersistentRequest struct {
	RequestID string                      `json:"requestID"`
	Command   string                      `json:"command"`
	LoadShard *PersistentLoadShardRequest `json:"loadShard,omitempty"`
	Shard     *PersistentShardRequest     `json:"shard,omitempty"`
	Sequence  *PersistentSequenceRequest  `json:"sequence,omitempty"`
	Forward   *PersistentForwardRequest   `json:"forward,omitempty"`
}

type PersistentLoadShardRequest struct {
	ModelID    string `json:"modelID"`
	ShardID    string `json:"shardID"`
	LayerStart int    `json:"layerStart"`
	LayerEnd   int    `json:"layerEnd"`
	OwnsInput  bool   `json:"ownsInput"`
	OwnsOutput bool   `json:"ownsOutput"`
}

type PersistentShardRequest struct {
	ShardID string `json:"shardID"`
}

type PersistentSequenceRequest struct {
	ShardID    string `json:"shardID"`
	SequenceID string `json:"sequenceID"`
}

type PersistentForwardRequest struct {
	ShardID    string     `json:"shardID"`
	SequenceID string     `json:"sequenceID"`
	Position   uint64     `json:"position"`
	InputKind  string     `json:"inputKind"`
	Input      WireTensor `json:"input"`
}

type WireTensor struct {
	Shape []int  `json:"shape"`
	DType string `json:"dtype"`
	Data  []byte `json:"data"`
}

type PersistentResponse struct {
	RequestID string                  `json:"requestID"`
	OK        bool                    `json:"ok"`
	Error     string                  `json:"error,omitempty"`
	Result    *PersistentWorkerResult `json:"result,omitempty"`
}

type PersistentWorkerResult struct {
	Status   string                   `json:"status,omitempty"`
	Shard    *PersistentShardSnapshot `json:"shard,omitempty"`
	Forward  *PersistentForwardResult `json:"forward,omitempty"`
	State    *PersistentWorkerState   `json:"state,omitempty"`
	Shutdown bool                     `json:"shutdown,omitempty"`
}

type PersistentForwardResult struct {
	ShardID       string      `json:"shardID"`
	SequenceID    string      `json:"sequenceID"`
	Position      uint64      `json:"position"`
	Output        WireTensor  `json:"output"`
	ComputeMicros uint64      `json:"computeMicros"`
	Memory        StageMemory `json:"memory"`
}

type PersistentShardSnapshot struct {
	ShardID           string      `json:"shardID"`
	ModelID           string      `json:"modelID"`
	ModelType         string      `json:"modelType"`
	LayerStart        int         `json:"layerStart"`
	LayerEnd          int         `json:"layerEnd"`
	OwnsInput         bool        `json:"ownsInput"`
	OwnsOutput        bool        `json:"ownsOutput"`
	WeightKeyCount    int         `json:"weightKeyCount"`
	OpenSequenceCount int         `json:"openSequenceCount"`
	ForwardCount      int         `json:"forwardCount"`
	LoadedMemory      StageMemory `json:"loadedMemory"`
}

type PersistentWorkerState struct {
	LoadedShards []PersistentShardSnapshot `json:"loadedShards"`
	LoadCount    int                       `json:"loadCount"`
	ForwardCount int                       `json:"forwardCount"`
	Memory       StageMemory               `json:"memory"`
}

type StageMemory struct {
	ActiveBytes int `json:"activeBytes"`
	CacheBytes  int `json:"cacheBytes"`
	PeakBytes   int `json:"peakBytes"`
}

type WorkerResponseError struct {
	RequestID string
	Message   string
}

func (e *WorkerResponseError) Error() string {
	return fmt.Sprintf("worker request %s: %s", e.RequestID, e.Message)
}

type cappedTailBuffer struct {
	mu        sync.Mutex
	data      []byte
	truncated bool
}

func (b *cappedTailBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	written := len(p)
	if written >= maxPersistentStderrBytes {
		b.truncated = b.truncated || len(b.data) > 0 || written > maxPersistentStderrBytes
		b.data = append(b.data[:0], p[written-maxPersistentStderrBytes:]...)
		return written, nil
	}
	if overflow := len(b.data) + written - maxPersistentStderrBytes; overflow > 0 {
		copy(b.data, b.data[overflow:])
		b.data = b.data[:len(b.data)-overflow]
		b.truncated = true
	}
	b.data = append(b.data, p...)
	return written, nil
}

func (b *cappedTailBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.truncated {
		return string(b.data)
	}
	return fmt.Sprintf(
		"[stderr truncated; showing last %d bytes]\n%s",
		maxPersistentStderrBytes,
		b.data,
	)
}

// PersistentClient supervises one long-lived MLX worker and multiplexes
// request/response JSON frames by requestID.
type PersistentClient struct {
	cmd          *exec.Cmd
	stdin        io.WriteCloser
	stderr       cappedTailBuffer
	done         chan struct{}
	writeMu      sync.Mutex
	mu           sync.Mutex
	pending      map[string]chan PersistentResponse
	nextID       uint64
	waitErr      error
	expectedExit bool
}

func StartPersistent(path string) (*PersistentClient, error) {
	if path == "" {
		path = DefaultPath()
	}
	cmd := exec.Command(path, "serve-stdio")
	// Keep terminal/service signals scoped to swarmd so it can ask the worker
	// to release MLX state and acknowledge shutdown instead of dying in parallel.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("worker stdin: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		_ = stdin.Close()
		return nil, fmt.Errorf("worker stdout: %w", err)
	}
	client := &PersistentClient{
		cmd:     cmd,
		stdin:   stdin,
		done:    make(chan struct{}),
		pending: make(map[string]chan PersistentResponse),
	}
	cmd.Stderr = &client.stderr
	if err := cmd.Start(); err != nil {
		_ = stdin.Close()
		return nil, fmt.Errorf("start persistent worker: %w", err)
	}
	go client.readLoop(stdout)
	return client, nil
}

func (c *PersistentClient) Call(
	ctx context.Context,
	request PersistentRequest,
) (PersistentResponse, error) {
	if request.Command == "" {
		return PersistentResponse{}, errors.New("persistent worker command is empty")
	}
	if err := ctx.Err(); err != nil {
		return PersistentResponse{}, err
	}

	c.mu.Lock()
	select {
	case <-c.done:
		c.mu.Unlock()
		return PersistentResponse{}, c.callError()
	default:
	}
	if request.RequestID == "" {
		c.nextID++
		request.RequestID = fmt.Sprintf("request-%d", c.nextID)
	}
	if _, exists := c.pending[request.RequestID]; exists {
		c.mu.Unlock()
		return PersistentResponse{}, fmt.Errorf("duplicate worker request ID %s", request.RequestID)
	}
	response := make(chan PersistentResponse, 1)
	c.pending[request.RequestID] = response
	c.mu.Unlock()

	payload, err := json.Marshal(request)
	if err != nil {
		c.removePending(request.RequestID)
		return PersistentResponse{}, fmt.Errorf("encode persistent worker request: %w", err)
	}
	payload = append(payload, '\n')

	c.writeMu.Lock()
	written, err := c.stdin.Write(payload)
	c.writeMu.Unlock()
	if err == nil && written != len(payload) {
		err = io.ErrShortWrite
	}
	if err != nil {
		_ = c.Kill()
		return PersistentResponse{}, fmt.Errorf("write persistent worker request: %w", err)
	}

	select {
	case result := <-response:
		return checkedResponse(result)
	case <-ctx.Done():
		return PersistentResponse{}, ctx.Err()
	case <-c.done:
		select {
		case result := <-response:
			return checkedResponse(result)
		default:
			return PersistentResponse{}, c.callError()
		}
	}
}

func (c *PersistentClient) Shutdown(ctx context.Context) error {
	response, err := c.Call(ctx, PersistentRequest{Command: "shutdown"})
	if err != nil {
		return err
	}
	if response.Result == nil || !response.Result.Shutdown {
		return errors.New("persistent worker did not acknowledge shutdown")
	}
	c.mu.Lock()
	c.expectedExit = true
	c.mu.Unlock()
	return c.Wait(ctx)
}

func (c *PersistentClient) Kill() error {
	if c.cmd.Process == nil {
		return errors.New("persistent worker has no process")
	}
	return c.cmd.Process.Kill()
}

func (c *PersistentClient) Wait(ctx context.Context) error {
	select {
	case <-c.done:
		return c.processError()
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (c *PersistentClient) readLoop(stdout io.Reader) {
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 64*1024), maxPersistentResponseBytes)
	for scanner.Scan() {
		var response PersistentResponse
		if err := json.Unmarshal(scanner.Bytes(), &response); err != nil {
			c.finish(fmt.Errorf("decode persistent worker response: %w", err))
			_ = c.cmd.Process.Kill()
			_ = c.cmd.Wait()
			return
		}
		c.mu.Lock()
		pending := c.pending[response.RequestID]
		delete(c.pending, response.RequestID)
		c.mu.Unlock()
		if pending != nil {
			pending <- response
		}
	}

	scanErr := scanner.Err()
	waitErr := c.cmd.Wait()
	if scanErr != nil {
		c.finish(fmt.Errorf("read persistent worker response: %w", scanErr))
		return
	}
	c.finish(waitErr)
}

func (c *PersistentClient) finish(err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	select {
	case <-c.done:
		return
	default:
	}
	c.waitErr = err
	c.pending = make(map[string]chan PersistentResponse)
	close(c.done)
}

func (c *PersistentClient) removePending(requestID string) {
	c.mu.Lock()
	delete(c.pending, requestID)
	c.mu.Unlock()
}

func (c *PersistentClient) processError() error {
	c.mu.Lock()
	err := c.waitErr
	expectedExit := c.expectedExit
	c.mu.Unlock()
	stderr := c.stderr.String()
	if err == nil {
		if expectedExit {
			return nil
		}
		if stderr == "" {
			return errors.New("persistent worker reached unexpected EOF")
		}
		return fmt.Errorf("persistent worker exited: %s", stderr)
	}
	if stderr == "" {
		return fmt.Errorf("persistent worker exited: %w", err)
	}
	return fmt.Errorf("persistent worker exited: %w: %s", err, stderr)
}

func (c *PersistentClient) callError() error {
	if err := c.processError(); err != nil {
		return err
	}
	return errPersistentWorkerStopped
}

func checkedResponse(response PersistentResponse) (PersistentResponse, error) {
	if !response.OK {
		return response, &WorkerResponseError{
			RequestID: response.RequestID,
			Message:   response.Error,
		}
	}
	return response, nil
}

package workerproc

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"syscall"
	"time"
)

const (
	maxPersistentResponseBytes = 128 << 20
	maxPersistentStderrBytes   = 64 << 10
	persistentEOFExitGrace     = 500 * time.Millisecond
	persistentEOFReapGrace     = time.Second
)

var errPersistentWorkerStopped = errors.New("persistent worker is stopped")

type PersistentRequest struct {
	RequestID          string                      `json:"requestID"`
	Command            string                      `json:"command"`
	DeadlineUnixMillis int64                       `json:"deadlineUnixMillis,omitempty"`
	LoadShard          *PersistentLoadShardRequest `json:"loadShard,omitempty"`
	Shard              *PersistentShardRequest     `json:"shard,omitempty"`
	Sequence           *PersistentSequenceRequest  `json:"sequence,omitempty"`
	Forward            *PersistentForwardRequest   `json:"forward,omitempty"`
	Model              *PersistentModelRequest     `json:"model,omitempty"`
	Text               *PersistentTextRequest      `json:"text,omitempty"`
}

type PersistentLoadShardRequest struct {
	ModelID               string `json:"modelID"`
	ShardID               string `json:"shardID"`
	CheckpointFingerprint string `json:"checkpointFingerprint,omitempty"`
	LayerStart            int    `json:"layerStart"`
	LayerEnd              int    `json:"layerEnd"`
	OwnsInput             bool   `json:"ownsInput"`
	OwnsOutput            bool   `json:"ownsOutput"`
}

type PersistentShardRequest struct {
	ShardID string `json:"shardID"`
}

type PersistentSequenceRequest struct {
	ShardID    string `json:"shardID"`
	SequenceID string `json:"sequenceID"`
	OwnerID    string `json:"ownerID,omitempty"`
}

type PersistentForwardRequest struct {
	ShardID    string     `json:"shardID"`
	SequenceID string     `json:"sequenceID"`
	Position   uint64     `json:"position"`
	InputKind  string     `json:"inputKind"`
	Input      WireTensor `json:"input"`
}

type PersistentModelRequest struct {
	ModelID string `json:"modelID"`
}

type PersistentTextRequest struct {
	ModelID           string  `json:"modelID"`
	Text              *string `json:"text,omitempty"`
	TokenIDs          []int32 `json:"tokenIDs,omitempty"`
	AddSpecialTokens  *bool   `json:"addSpecialTokens,omitempty"`
	SkipSpecialTokens *bool   `json:"skipSpecialTokens,omitempty"`
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
	Model    *PersistentModelResult   `json:"model,omitempty"`
	Text     *PersistentTextResult    `json:"text,omitempty"`
	State    *PersistentWorkerState   `json:"state,omitempty"`
	Shutdown bool                     `json:"shutdown,omitempty"`
}

type PersistentModelResult struct {
	ModelID               string `json:"modelID"`
	ModelType             string `json:"modelType"`
	LayerCount            int    `json:"layerCount"`
	CheckpointFingerprint string `json:"checkpointFingerprint"`
}

type PersistentTextResult struct {
	ModelID    string  `json:"modelID"`
	TokenIDs   []int32 `json:"tokenIDs,omitempty"`
	Text       *string `json:"text,omitempty"`
	EOSTokenID *int32  `json:"eosTokenID,omitempty"`
}

type PersistentForwardResult struct {
	ShardID       string      `json:"shardID"`
	SequenceID    string      `json:"sequenceID"`
	Operation     string      `json:"operation"`
	Position      uint64      `json:"position"`
	NextPosition  uint64      `json:"nextPosition"`
	Output        WireTensor  `json:"output"`
	ComputeMicros uint64      `json:"computeMicros"`
	KVCacheBytes  int         `json:"kvCacheBytes"`
	Memory        StageMemory `json:"memory"`
}

type PersistentShardSnapshot struct {
	ShardID               string      `json:"shardID"`
	ModelID               string      `json:"modelID"`
	ModelType             string      `json:"modelType"`
	CheckpointFingerprint string      `json:"checkpointFingerprint"`
	LayerStart            int         `json:"layerStart"`
	LayerEnd              int         `json:"layerEnd"`
	OwnsInput             bool        `json:"ownsInput"`
	OwnsOutput            bool        `json:"ownsOutput"`
	WeightKeyCount        int         `json:"weightKeyCount"`
	OpenSequenceCount     int         `json:"openSequenceCount"`
	MaxOpenSequenceCount  int         `json:"maxOpenSequenceCount"`
	ForwardCount          int         `json:"forwardCount"`
	KVCacheBytes          int         `json:"kvCacheBytes"`
	RetainedBytes         int         `json:"retainedBytes"`
	LoadedMemory          StageMemory `json:"loadedMemory"`
}

type PersistentWorkerState struct {
	LoadedShards       []PersistentShardSnapshot `json:"loadedShards"`
	LoadCount          int                       `json:"loadCount"`
	ForwardCount       int                       `json:"forwardCount"`
	KVCacheBytes       int                       `json:"kvCacheBytes"`
	RetainedBytes      int                       `json:"retainedBytes"`
	RetainedByteBudget int                       `json:"retainedByteBudget"`
	Memory             StageMemory               `json:"memory"`
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
	writes       chan persistentWrite
	mu           sync.Mutex
	pending      map[string]chan PersistentResponse
	nextID       uint64
	waitErr      error
	expectedExit bool
}

type persistentWrite struct {
	ctx       context.Context
	requestID string
	payload   []byte
	result    chan error
}

func StartPersistent(path string) (*PersistentClient, error) {
	return StartPersistentCommand(path, "serve-stdio")
}

// StartPersistentCommand starts a framed worker using explicit arguments.
// Production callers use StartPersistent; deterministic fault harnesses use
// this entry point to run a protocol-compatible child mode from their binary.
func StartPersistentCommand(path string, args ...string) (*PersistentClient, error) {
	if path == "" {
		path = DefaultPath()
	}
	cmd := exec.Command(path, args...)
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
		writes:  make(chan persistentWrite),
		pending: make(map[string]chan PersistentResponse),
	}
	cmd.Stderr = &client.stderr
	if err := cmd.Start(); err != nil {
		_ = stdin.Close()
		return nil, fmt.Errorf("start persistent worker: %w", err)
	}
	go client.writeLoop()
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
	callContext, cancel, prepared, err := RequestContext(ctx, request)
	if err != nil {
		return PersistentResponse{}, err
	}
	defer cancel()
	ctx = callContext
	request = prepared

	c.mu.Lock()
	select {
	case <-c.done:
		c.mu.Unlock()
		return PersistentResponse{}, c.callError()
	default:
	}
	if request.RequestID == "" {
		for {
			c.nextID++
			candidate := fmt.Sprintf("request-%d", c.nextID)
			if _, exists := c.pending[candidate]; !exists {
				request.RequestID = candidate
				break
			}
		}
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

	writeResult := make(chan error, 1)
	select {
	case c.writes <- persistentWrite{
		ctx: ctx, requestID: request.RequestID, payload: payload, result: writeResult,
	}:
	case <-ctx.Done():
		c.removePending(request.RequestID)
		c.abortTimedOutInference(request)
		return PersistentResponse{}, ctx.Err()
	case <-c.done:
		return PersistentResponse{}, c.callError()
	}

	select {
	case err := <-writeResult:
		if err != nil {
			return PersistentResponse{}, fmt.Errorf("write persistent worker request: %w", err)
		}
	case <-ctx.Done():
		c.abortTimedOutInference(request)
		return PersistentResponse{}, ctx.Err()
	case <-c.done:
		return PersistentResponse{}, c.callError()
	}

	select {
	case result := <-response:
		if err := ctx.Err(); err != nil {
			c.abortTimedOutInference(request)
			return PersistentResponse{}, err
		}
		return checkedResponse(result)
	case <-ctx.Done():
		c.abortTimedOutInference(request)
		return PersistentResponse{}, ctx.Err()
	case <-c.done:
		select {
		case result := <-response:
			if err := ctx.Err(); err != nil {
				c.abortTimedOutInference(request)
				return PersistentResponse{}, err
			}
			return checkedResponse(result)
		default:
			return PersistentResponse{}, c.callError()
		}
	}
}

func (c *PersistentClient) abortTimedOutInference(request PersistentRequest) {
	if isInferenceCommand(request.Command) {
		_ = c.Kill()
	}
}

func (c *PersistentClient) writeLoop() {
	for {
		select {
		case write := <-c.writes:
			// A later request may already be queued after the caller stops
			// waiting. Never let this canceled write overtake that request when
			// the worker becomes writable again.
			if err := write.ctx.Err(); err != nil {
				c.removePending(write.requestID)
				write.result <- err
				continue
			}
			written, err := c.stdin.Write(write.payload)
			if err == nil && written != len(write.payload) {
				err = io.ErrShortWrite
			}
			if err != nil {
				_ = c.Kill()
			}
			write.result <- err
			if err != nil {
				return
			}
		case <-c.done:
			return
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
	if err := syscall.Kill(-c.cmd.Process.Pid, syscall.SIGKILL); err != nil {
		if errors.Is(err, syscall.ESRCH) {
			return os.ErrProcessDone
		}
		return err
	}
	return nil
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
			c.failRead(fmt.Errorf("decode persistent worker response: %w", err))
			return
		}
		c.mu.Lock()
		pending, exists := c.pending[response.RequestID]
		if exists {
			delete(c.pending, response.RequestID)
		}
		c.mu.Unlock()
		if !exists {
			c.failRead(fmt.Errorf(
				"persistent worker returned unmatched request ID %q",
				response.RequestID,
			))
			return
		}
		pending <- response
	}

	scanErr := scanner.Err()
	if scanErr != nil {
		c.failRead(fmt.Errorf("read persistent worker response: %w", scanErr))
		return
	}
	c.finishEOF()
}

func (c *PersistentClient) finishEOF() {
	waitResult := make(chan error, 1)
	go func() {
		waitResult <- c.cmd.Wait()
	}()

	select {
	case waitErr := <-waitResult:
		c.finish(waitErr)
		return
	case <-time.After(persistentEOFExitGrace):
	}

	eofErr := errors.New("persistent worker closed stdout without exiting")
	killErr := c.Kill()
	_ = c.stdin.Close()
	if killErr != nil && !errors.Is(killErr, os.ErrProcessDone) {
		c.finish(fmt.Errorf("%w; kill persistent worker: %v", eofErr, killErr))
		return
	}

	select {
	case <-waitResult:
		c.finish(eofErr)
	case <-time.After(persistentEOFReapGrace):
		c.finish(fmt.Errorf("%w; timed out reaping process", eofErr))
	}
}

func (c *PersistentClient) failRead(readErr error) {
	killErr := c.Kill()
	if killErr != nil && !errors.Is(killErr, os.ErrProcessDone) {
		_ = c.stdin.Close()
		c.finish(fmt.Errorf("%w; kill persistent worker: %v", readErr, killErr))
		return
	}
	_ = c.cmd.Wait()
	c.finish(readErr)
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

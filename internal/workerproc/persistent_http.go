package workerproc

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
)

const (
	// InstanceBoundHTTPProtocol is advertised only by daemons that enforce the
	// worker and process-incarnation preconditions sent by bound clients.
	InstanceBoundHTTPProtocol    = "http-json-instance-v1"
	Base64JSONTensorEncoding     = "base64-json"
	ExpectedWorkerIDHeader       = "X-MLX-Swarm-Expected-Worker-ID"
	ExpectedWorkerInstanceHeader = "X-MLX-Swarm-Expected-Worker-Instance-ID"
)

// PersistentCaller is the shared request surface for a directly supervised
// worker process and the trusted-network swarmd proxy.
type PersistentCaller interface {
	Call(context.Context, PersistentRequest) (PersistentResponse, error)
}

// HTTPPersistentClient sends persistent worker frames through swarmd. The
// endpoint is intentionally limited to trusted networks.
type HTTPPersistentClient struct {
	endpoint           string
	client             *http.Client
	expectedWorkerID   string
	expectedInstanceID string
	mu                 sync.Mutex
	nextID             uint64
}

func NewHTTPPersistentClient(peer string, client *http.Client) (*HTTPPersistentClient, error) {
	return newHTTPPersistentClient(peer, client, "", "")
}

// NewBoundHTTPPersistentClient requires every request to reach the exact
// membership incarnation selected by the scheduler. A restarted daemon at the
// same endpoint rejects the request before it can mutate shard or sequence
// state.
func NewBoundHTTPPersistentClient(
	peer string,
	client *http.Client,
	workerID string,
	instanceID string,
) (*HTTPPersistentClient, error) {
	if strings.TrimSpace(workerID) == "" || strings.TrimSpace(instanceID) == "" {
		return nil, fmt.Errorf("bound swarmd peer requires worker and instance IDs")
	}
	return newHTTPPersistentClient(peer, client, workerID, instanceID)
}

func newHTTPPersistentClient(
	peer string,
	client *http.Client,
	expectedWorkerID string,
	expectedInstanceID string,
) (*HTTPPersistentClient, error) {
	parsed, err := url.Parse(strings.TrimRight(peer, "/"))
	if err != nil {
		return nil, fmt.Errorf("parse swarmd peer: %w", err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return nil, fmt.Errorf("swarmd peer must use http or https")
	}
	if parsed.Host == "" {
		return nil, fmt.Errorf("swarmd peer has no host")
	}
	if client == nil {
		client = http.DefaultClient
	}
	return &HTTPPersistentClient{
		endpoint:           strings.TrimRight(peer, "/") + "/v1/worker/request",
		client:             client,
		expectedWorkerID:   expectedWorkerID,
		expectedInstanceID: expectedInstanceID,
	}, nil
}

func (c *HTTPPersistentClient) Call(
	ctx context.Context,
	request PersistentRequest,
) (PersistentResponse, error) {
	if request.Command == "" {
		return PersistentResponse{}, fmt.Errorf("persistent worker command is empty")
	}
	if err := contextCompletionError(ctx); err != nil {
		return PersistentResponse{}, err
	}
	callContext, cancel, prepared, err := RequestContext(ctx, request)
	if err != nil {
		return PersistentResponse{}, err
	}
	defer cancel()
	ctx = callContext
	request = prepared
	if request.RequestID == "" {
		c.mu.Lock()
		c.nextID++
		request.RequestID = fmt.Sprintf("http-request-%d", c.nextID)
		c.mu.Unlock()
	}
	payload, err := json.Marshal(request)
	if err != nil {
		return PersistentResponse{}, fmt.Errorf("encode persistent worker request: %w", err)
	}
	httpRequest, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		c.endpoint,
		bytes.NewReader(payload),
	)
	if err != nil {
		return PersistentResponse{}, fmt.Errorf("create persistent worker request: %w", err)
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	if c.expectedWorkerID != "" {
		httpRequest.Header.Set(ExpectedWorkerIDHeader, c.expectedWorkerID)
		httpRequest.Header.Set(ExpectedWorkerInstanceHeader, c.expectedInstanceID)
	}
	httpResponse, err := c.client.Do(httpRequest)
	if err != nil {
		return PersistentResponse{}, fmt.Errorf("call swarmd persistent worker: %w", err)
	}
	defer httpResponse.Body.Close()
	body, err := io.ReadAll(io.LimitReader(httpResponse.Body, maxPersistentResponseBytes+1))
	if err != nil {
		return PersistentResponse{}, fmt.Errorf("read swarmd persistent worker response: %w", err)
	}
	if err := contextCompletionError(ctx); err != nil {
		return PersistentResponse{}, err
	}
	if len(body) > maxPersistentResponseBytes {
		return PersistentResponse{}, fmt.Errorf("swarmd persistent worker response exceeds %d bytes", maxPersistentResponseBytes)
	}

	var response PersistentResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return PersistentResponse{}, fmt.Errorf(
			"decode swarmd persistent worker response (HTTP %d): %w",
			httpResponse.StatusCode,
			err,
		)
	}
	if httpResponse.StatusCode < 200 || httpResponse.StatusCode >= 300 {
		if response.Error == "" {
			return response, fmt.Errorf("swarmd persistent worker returned HTTP %d", httpResponse.StatusCode)
		}
		if response.RequestID == "" {
			return response, fmt.Errorf("swarmd persistent worker: %s", response.Error)
		}
	}
	return checkedResponse(response)
}

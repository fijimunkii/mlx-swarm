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

// PersistentCaller is the shared request surface for a directly supervised
// worker process and the trusted-network swarmd proxy.
type PersistentCaller interface {
	Call(context.Context, PersistentRequest) (PersistentResponse, error)
}

// HTTPPersistentClient sends persistent worker frames through swarmd. The v0
// endpoint is intentionally limited to trusted networks.
type HTTPPersistentClient struct {
	endpoint string
	client   *http.Client
	mu       sync.Mutex
	nextID   uint64
}

func NewHTTPPersistentClient(peer string, client *http.Client) (*HTTPPersistentClient, error) {
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
		endpoint: strings.TrimRight(peer, "/") + "/v1/worker/request",
		client:   client,
	}, nil
}

func (c *HTTPPersistentClient) Call(
	ctx context.Context,
	request PersistentRequest,
) (PersistentResponse, error) {
	if request.Command == "" {
		return PersistentResponse{}, fmt.Errorf("persistent worker command is empty")
	}
	if err := ctx.Err(); err != nil {
		return PersistentResponse{}, err
	}
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
	httpResponse, err := c.client.Do(httpRequest)
	if err != nil {
		return PersistentResponse{}, fmt.Errorf("call swarmd persistent worker: %w", err)
	}
	defer httpResponse.Body.Close()
	body, err := io.ReadAll(io.LimitReader(httpResponse.Body, maxPersistentResponseBytes+1))
	if err != nil {
		return PersistentResponse{}, fmt.Errorf("read swarmd persistent worker response: %w", err)
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

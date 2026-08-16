package registry

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
)

const maxMembershipBodyBytes = 1 << 20

type HTTPHandler struct {
	registry *Registry
}

func NewHTTPHandler(registry *Registry) http.Handler {
	handler := &HTTPHandler{registry: registry}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/membership", handler.inventory)
	mux.HandleFunc("POST /v1/membership/workers", handler.register)
	mux.HandleFunc("POST /v1/membership/workers/{id}/heartbeat", handler.heartbeat)
	mux.HandleFunc("DELETE /v1/membership/workers/{id}", handler.remove)
	return mux
}

func (h *HTTPHandler) inventory(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, h.registry.Snapshot())
}

func (h *HTTPHandler) register(w http.ResponseWriter, request *http.Request) {
	var input RegistrationRequest
	if err := decodeJSON(w, request, &input); err != nil {
		writeAPIError(w, http.StatusBadRequest, "invalid_request", err)
		return
	}
	mutation, err := h.registry.register(input.Registration, input.StatusFresh)
	if err != nil {
		writeRegistryError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, mutation)
}

func (h *HTTPHandler) heartbeat(w http.ResponseWriter, request *http.Request) {
	var input Heartbeat
	if err := decodeJSON(w, request, &input); err != nil {
		writeAPIError(w, http.StatusBadRequest, "invalid_request", err)
		return
	}
	mutation, err := h.registry.Heartbeat(request.PathValue("id"), input)
	if err != nil {
		writeRegistryError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, mutation)
}

func (h *HTTPHandler) remove(w http.ResponseWriter, request *http.Request) {
	instanceID := request.URL.Query().Get("instanceID")
	if err := h.registry.Remove(request.PathValue("id"), instanceID); err != nil {
		writeRegistryError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func decodeJSON(w http.ResponseWriter, request *http.Request, destination any) error {
	request.Body = http.MaxBytesReader(w, request.Body, maxMembershipBodyBytes)
	decoder := json.NewDecoder(request.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("request body must contain one JSON value")
		}
		return err
	}
	return nil
}

func writeRegistryError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, ErrDuplicateWorker):
		writeAPIError(w, http.StatusConflict, "duplicate_worker", err)
	case errors.Is(err, ErrDuplicateInstance):
		writeAPIError(w, http.StatusConflict, "duplicate_instance", err)
	case errors.Is(err, ErrDuplicateEndpoint):
		writeAPIError(w, http.StatusConflict, "duplicate_endpoint", err)
	case errors.Is(err, ErrStaleInstance):
		writeAPIError(w, http.StatusConflict, "stale_instance", err)
	case errors.Is(err, ErrLeaseExpired):
		writeAPIError(w, http.StatusGone, "lease_expired", err)
	case errors.Is(err, ErrWorkerNotFound):
		writeAPIError(w, http.StatusNotFound, "worker_not_found", err)
	default:
		writeAPIError(w, http.StatusBadRequest, "invalid_record", err)
	}
}

func writeAPIError(w http.ResponseWriter, status int, code string, err error) {
	writeJSON(w, status, APIError{Code: code, Error: err.Error()})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

type RemoteError struct {
	StatusCode int
	Code       string
	Message    string
}

func (e *RemoteError) Error() string {
	return fmt.Sprintf("membership API returned HTTP %d (%s): %s", e.StatusCode, e.Code, e.Message)
}

type Client struct {
	endpoint string
	http     *http.Client
}

func NewClient(endpoint string, httpClient *http.Client) (*Client, error) {
	parsed, err := url.Parse(strings.TrimRight(endpoint, "/"))
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" ||
		parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return nil, errors.New("membership endpoint must be an http(s) URL")
	}
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return &Client{endpoint: parsed.String(), http: httpClient}, nil
}

func (c *Client) Register(ctx context.Context, input Registration, statusFresh bool) (Mutation, error) {
	var result Mutation
	request := RegistrationRequest{Registration: input, StatusFresh: statusFresh}
	err := c.call(ctx, http.MethodPost, "/v1/membership/workers", request, &result)
	return result, err
}

func (c *Client) Heartbeat(ctx context.Context, id string, input Heartbeat) (Mutation, error) {
	var result Mutation
	path := "/v1/membership/workers/" + url.PathEscape(id) + "/heartbeat"
	err := c.call(ctx, http.MethodPost, path, input, &result)
	return result, err
}

func (c *Client) Remove(ctx context.Context, id, instanceID string) error {
	path := "/v1/membership/workers/" + url.PathEscape(id) + "?instanceID=" + url.QueryEscape(instanceID)
	return c.call(ctx, http.MethodDelete, path, nil, nil)
}

func (c *Client) Inventory(ctx context.Context) (Inventory, error) {
	var result Inventory
	err := c.call(ctx, http.MethodGet, "/v1/membership", nil, &result)
	return result, err
}

func (c *Client) call(ctx context.Context, method, path string, input, output any) error {
	var body io.Reader
	if input != nil {
		encoded, err := json.Marshal(input)
		if err != nil {
			return err
		}
		body = bytes.NewReader(encoded)
	}
	request, err := http.NewRequestWithContext(ctx, method, c.endpoint+path, body)
	if err != nil {
		return err
	}
	if input != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	response, err := c.http.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	reader := io.LimitReader(response.Body, maxMembershipBodyBytes+1)
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		var apiError APIError
		if err := json.NewDecoder(reader).Decode(&apiError); err != nil {
			return fmt.Errorf("membership API returned HTTP %d", response.StatusCode)
		}
		return &RemoteError{
			StatusCode: response.StatusCode, Code: apiError.Code, Message: apiError.Error,
		}
	}
	if output == nil || response.StatusCode == http.StatusNoContent {
		return nil
	}
	if err := json.NewDecoder(reader).Decode(output); err != nil {
		return fmt.Errorf("decode membership response: %w", err)
	}
	return nil
}

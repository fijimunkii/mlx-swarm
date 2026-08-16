package registry

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestMembershipHTTPClientLifecycle(t *testing.T) {
	server := httptest.NewServer(NewHTTPHandler(New(time.Minute)))
	defer server.Close()
	client, err := NewClient(server.URL, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	input := testRegistration("worker-a", "instance-a")
	registered, err := client.Register(ctx, input, true)
	if err != nil {
		t.Fatal(err)
	}
	if registered.Worker.ID != input.ID || registered.InventoryRevision != 1 {
		t.Fatalf("unexpected registration: %+v", registered)
	}
	statusObservedAt := registered.Worker.StatusObservedAt
	if statusObservedAt.IsZero() {
		t.Fatalf("registration omitted status observation time: %+v", registered)
	}

	inventory, err := client.Inventory(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if inventory.SchemaVersion != SchemaVersion || len(inventory.Workers) != 1 {
		t.Fatalf("unexpected inventory: %+v", inventory)
	}

	leaseOnly, err := client.Heartbeat(ctx, input.ID, Heartbeat{
		SchemaVersion: SchemaVersion, InstanceID: input.InstanceID,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !leaseOnly.Worker.StatusObservedAt.Equal(statusObservedAt) || leaseOnly.InventoryRevision != 1 {
		t.Fatalf("lease-only heartbeat changed status observation: %+v", leaseOnly)
	}

	input.Status.RestartCount = 2
	if _, err := client.Heartbeat(ctx, input.ID, Heartbeat{
		SchemaVersion: SchemaVersion, InstanceID: input.InstanceID, Status: &input.Status,
	}); err != nil {
		t.Fatal(err)
	}
	if err := client.Remove(ctx, input.ID, input.InstanceID); err != nil {
		t.Fatal(err)
	}
	inventory, err = client.Inventory(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(inventory.Workers) != 0 || inventory.Revision != 3 {
		t.Fatalf("worker was not removed: %+v", inventory)
	}
}

func TestMembershipHTTPReturnsMachineReadableConflicts(t *testing.T) {
	server := httptest.NewServer(NewHTTPHandler(New(time.Minute)))
	defer server.Close()
	client, err := NewClient(server.URL, server.Client())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Register(
		context.Background(), testRegistration("worker-a", "instance-a"), true,
	); err != nil {
		t.Fatal(err)
	}
	_, err = client.Register(context.Background(), testRegistration("worker-a", "instance-b"), true)
	var remote *RemoteError
	if !errors.As(err, &remote) || remote.StatusCode != http.StatusConflict || remote.Code != "duplicate_worker" {
		t.Fatalf("duplicate registration error = %#v", err)
	}
}

func TestMembershipHTTPRejectsUnknownAndMultipleJSONValues(t *testing.T) {
	server := httptest.NewServer(NewHTTPHandler(New(time.Minute)))
	defer server.Close()
	for _, body := range []string{
		`{"schemaVersion":1,"unknown":true}`,
		`{} {}`,
	} {
		request, err := http.NewRequest(http.MethodPost, server.URL+"/v1/membership/workers", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		response, err := server.Client().Do(request)
		if err != nil {
			t.Fatal(err)
		}
		var apiError APIError
		decodeErr := json.NewDecoder(response.Body).Decode(&apiError)
		response.Body.Close()
		if decodeErr != nil || response.StatusCode != http.StatusBadRequest || apiError.Code != "invalid_request" {
			t.Fatalf("body %q returned HTTP %d error=%+v decode=%v", body, response.StatusCode, apiError, decodeErr)
		}
	}
}

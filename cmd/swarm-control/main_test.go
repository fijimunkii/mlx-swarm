package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/registry"
)

func TestControlHandlerReportsInventoryHealth(t *testing.T) {
	membership := registry.New(time.Minute)
	server := httptest.NewServer(newControlHandler(membership))
	defer server.Close()
	response, err := server.Client().Get(server.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	var health healthResponse
	if err := json.NewDecoder(response.Body).Decode(&health); err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusOK || health.Status != "ok" ||
		health.SchemaVersion != registry.SchemaVersion || health.WorkerCount != 0 {
		t.Fatalf("unexpected health response: HTTP %d %+v", response.StatusCode, health)
	}

	response, err = server.Client().Get(server.URL + "/v1/membership")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	var inventory registry.Inventory
	if err := json.NewDecoder(response.Body).Decode(&inventory); err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusOK || inventory.SchemaVersion != registry.SchemaVersion {
		t.Fatalf("unexpected inventory response: HTTP %d %+v", response.StatusCode, inventory)
	}
}

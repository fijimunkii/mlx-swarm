package workerproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestBoundHTTPPersistentClientSendsWorkerIdentityPrecondition(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if got := request.Header.Get(ExpectedWorkerIDHeader); got != "worker-a" {
			t.Errorf("worker header = %q", got)
		}
		if got := request.Header.Get(ExpectedWorkerInstanceHeader); got != "instance-a" {
			t.Errorf("instance header = %q", got)
		}
		_ = json.NewEncoder(w).Encode(PersistentResponse{OK: true})
	}))
	defer server.Close()

	client, err := NewBoundHTTPPersistentClient(
		server.URL, server.Client(), "worker-a", "instance-a",
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Call(context.Background(), PersistentRequest{Command: "health"}); err != nil {
		t.Fatal(err)
	}
}

func TestBoundHTTPPersistentClientRequiresCompleteIdentity(t *testing.T) {
	for _, identity := range [][2]string{{"", "instance-a"}, {"worker-a", ""}, {" ", "instance-a"}} {
		if _, err := NewBoundHTTPPersistentClient(
			"http://worker-a.example:8080", nil, identity[0], identity[1],
		); err == nil {
			t.Fatalf("identity %q/%q was accepted", identity[0], identity[1])
		}
	}
}

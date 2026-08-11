package workerproc

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestPersistentClientReportsUnexpectedEOF(t *testing.T) {
	worker := writeWorkerScript(t, "#!/bin/sh\nexit 0\n")
	client, err := StartPersistent(worker)
	if err != nil {
		t.Fatalf("StartPersistent: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := client.Wait(ctx); err == nil || !strings.Contains(err.Error(), "unexpected EOF") {
		t.Fatalf("Wait error = %v, want unexpected EOF", err)
	}
}

func TestPersistentClientAcceptsAcknowledgedShutdown(t *testing.T) {
	worker := writeWorkerScript(t, `#!/bin/sh
read request
printf '%s\n' '{"requestID":"request-1","ok":true,"result":{"shutdown":true}}'
`)
	client, err := StartPersistent(worker)
	if err != nil {
		t.Fatalf("StartPersistent: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := client.Shutdown(ctx); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
}

func writeWorkerScript(t *testing.T, contents string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "fake-worker")
	if err := os.WriteFile(path, []byte(contents), 0o700); err != nil {
		t.Fatalf("write fake worker: %v", err)
	}
	return path
}

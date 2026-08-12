package workerproc

import (
	"context"
	"net/http"
	"time"
)

// PersistentTarget owns either a directly supervised worker or a remote
// swarmd caller. Remote worker lifecycle remains owned by swarmd.
type PersistentTarget struct {
	Caller PersistentCaller
	direct *PersistentClient
	closed bool
}

// OpenPersistentTarget opens a remote endpoint when endpoint is nonempty and
// otherwise starts a directly supervised worker process.
func OpenPersistentTarget(workerPath string, endpoint string) (*PersistentTarget, error) {
	return OpenPersistentTargetWithHTTPClient(workerPath, endpoint, nil)
}

// OpenPersistentTargetWithHTTPClient opens a persistent target using client
// for every remote swarmd request. The client is ignored for direct workers.
func OpenPersistentTargetWithHTTPClient(
	workerPath string,
	endpoint string,
	httpClient *http.Client,
) (*PersistentTarget, error) {
	if endpoint != "" {
		client, err := NewHTTPPersistentClient(endpoint, httpClient)
		if err != nil {
			return nil, err
		}
		return &PersistentTarget{Caller: client}, nil
	}
	client, err := StartPersistent(workerPath)
	if err != nil {
		return nil, err
	}
	return &PersistentTarget{Caller: client, direct: client}, nil
}

// IsDirect reports whether this target owns a local worker process.
func (target *PersistentTarget) IsDirect() bool {
	return target != nil && target.direct != nil
}

// Shutdown gracefully stops a directly supervised worker. It is a no-op for
// remote workers.
func (target *PersistentTarget) Shutdown(ctx context.Context) error {
	if target == nil || target.direct == nil || target.closed {
		return nil
	}
	if err := target.direct.Shutdown(ctx); err != nil {
		return err
	}
	target.closed = true
	return nil
}

// Cleanup releases a directly supervised worker, falling back to a kill when
// it did not complete a graceful shutdown.
func (target *PersistentTarget) Cleanup() {
	if target == nil || target.direct == nil || target.closed {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	if err := target.direct.Shutdown(ctx); err == nil {
		target.closed = true
		cancel()
		return
	}
	cancel()
	_ = target.direct.Kill()
	reapCtx, reapCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer reapCancel()
	_ = target.direct.Wait(reapCtx)
	target.closed = true
}

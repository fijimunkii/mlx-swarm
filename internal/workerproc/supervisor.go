package workerproc

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"time"
)

const supervisorReapTimeout = 2 * time.Second

type PersistentStarter func() (*PersistentClient, error)

// PersistentSupervisor replaces a failed worker before returning the
// triggering call error. Mutating requests are never retried; a subsequent
// session must reload its shards and start a fresh sequence.
type PersistentSupervisor struct {
	mu           sync.Mutex
	callGate     chan struct{}
	starter      PersistentStarter
	client       *PersistentClient
	closed       bool
	restartCount int
}

func StartPersistentSupervisor(path string) (*PersistentSupervisor, error) {
	return NewPersistentSupervisor(func() (*PersistentClient, error) {
		return StartPersistent(path)
	})
}

func NewPersistentSupervisor(starter PersistentStarter) (*PersistentSupervisor, error) {
	if starter == nil {
		return nil, errors.New("persistent worker starter is required")
	}
	client, err := starter()
	if err != nil {
		return nil, err
	}
	return &PersistentSupervisor{
		callGate: make(chan struct{}, 1), starter: starter, client: client,
	}, nil
}

func (supervisor *PersistentSupervisor) Call(
	ctx context.Context,
	request PersistentRequest,
) (PersistentResponse, error) {
	if request.Command == "" {
		return PersistentResponse{}, errors.New("persistent worker command is empty")
	}
	callContext, cancel, prepared, err := RequestContext(ctx, request)
	if err != nil {
		return PersistentResponse{}, err
	}
	defer cancel()
	ctx = callContext
	request = prepared
	if err := supervisor.acquireCall(ctx); err != nil {
		return PersistentResponse{}, err
	}
	defer supervisor.releaseCall()
	if err := ctx.Err(); err != nil {
		return PersistentResponse{}, err
	}

	supervisor.mu.Lock()
	if supervisor.closed {
		supervisor.mu.Unlock()
		return PersistentResponse{}, errPersistentWorkerStopped
	}
	client := supervisor.client
	supervisor.mu.Unlock()
	response, err := client.Call(ctx, request)
	if err == nil || isWorkerRejection(err) || isNotDispatched(err) ||
		errors.Is(err, ErrInferenceDeadlineRequired) {
		return response, err
	}
	if errors.Is(err, context.Canceled) && !isInferenceCommand(request.Command) {
		return response, err
	}
	if restartErr := supervisor.restart(client); restartErr != nil {
		return response, errors.Join(err, fmt.Errorf("restart persistent worker: %w", restartErr))
	}
	if isSafeSupervisorRetry(request.Command) && ctx.Err() == nil {
		supervisor.mu.Lock()
		retryClient := supervisor.client
		supervisor.mu.Unlock()
		return retryClient.Call(ctx, request)
	}
	return response, err
}

func (supervisor *PersistentSupervisor) RestartCount() int {
	supervisor.mu.Lock()
	defer supervisor.mu.Unlock()
	return supervisor.restartCount
}

func (supervisor *PersistentSupervisor) Shutdown(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := supervisor.acquireCall(ctx); err != nil {
		return err
	}
	defer supervisor.releaseCall()
	supervisor.mu.Lock()
	if supervisor.closed {
		supervisor.mu.Unlock()
		return nil
	}
	client := supervisor.client
	supervisor.mu.Unlock()
	if err := client.Shutdown(ctx); err != nil {
		return err
	}
	supervisor.mu.Lock()
	supervisor.closed = true
	supervisor.mu.Unlock()
	return nil
}

func (supervisor *PersistentSupervisor) Kill() error {
	supervisor.mu.Lock()
	defer supervisor.mu.Unlock()
	if supervisor.client == nil {
		return os.ErrProcessDone
	}
	return supervisor.client.Kill()
}

func (supervisor *PersistentSupervisor) Wait(ctx context.Context) error {
	supervisor.mu.Lock()
	client := supervisor.client
	supervisor.mu.Unlock()
	if client == nil {
		return os.ErrProcessDone
	}
	return client.Wait(ctx)
}

func (supervisor *PersistentSupervisor) acquireCall(ctx context.Context) error {
	select {
	case supervisor.callGate <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (supervisor *PersistentSupervisor) releaseCall() {
	<-supervisor.callGate
}

func (supervisor *PersistentSupervisor) restart(failed *PersistentClient) error {
	supervisor.mu.Lock()
	if supervisor.closed {
		supervisor.mu.Unlock()
		return errPersistentWorkerStopped
	}
	if supervisor.client != failed {
		supervisor.mu.Unlock()
		return nil
	}
	old := supervisor.client
	supervisor.mu.Unlock()
	if old != nil {
		killErr := old.Kill()
		reapContext, cancel := context.WithTimeout(context.Background(), supervisorReapTimeout)
		waitErr := old.Wait(reapContext)
		cancel()
		if errors.Is(waitErr, context.DeadlineExceeded) {
			if killErr != nil && !errors.Is(killErr, os.ErrProcessDone) {
				return fmt.Errorf("kill failed worker: %w", killErr)
			}
			return fmt.Errorf("reap failed worker: %w", waitErr)
		}
	}
	client, err := supervisor.starter()
	if err != nil {
		return err
	}
	supervisor.mu.Lock()
	supervisor.client = client
	supervisor.restartCount++
	supervisor.mu.Unlock()
	return nil
}

func isWorkerRejection(err error) bool {
	var responseError *WorkerResponseError
	return errors.As(err, &responseError)
}

func isSafeSupervisorRetry(command string) bool {
	return command == "health" || command == "state"
}

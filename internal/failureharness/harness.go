package failureharness

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"sync/atomic"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/smoke"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

const faultModelID = "fault-harness/model"

type ScenarioResult struct {
	Name               string              `json:"name"`
	ExpectedFailure    bool                `json:"expectedFailure"`
	ObservedFailure    bool                `json:"observedFailure"`
	TerminatedInMillis int64               `json:"terminatedInMillis"`
	BoundMillis        int64               `json:"boundMillis"`
	AcceptedTokens     int                 `json:"acceptedTokens"`
	FailedTokens       int                 `json:"failedTokens"`
	Failure            *generation.Failure `json:"failure,omitempty"`
	WorkerRestarts     int                 `json:"workerRestarts"`
	CleanupReleased    bool                `json:"cleanupReleased"`
	RecoveryReady      bool                `json:"recoveryReady"`
}

type Summary struct {
	SchemaVersion        string           `json:"schemaVersion"`
	ForwardDeadlineMS    int64            `json:"forwardDeadlineMillis"`
	ScenarioBoundMS      int64            `json:"scenarioBoundMillis"`
	Scenarios            []ScenarioResult `json:"scenarios"`
	AcceptedTokens       int              `json:"acceptedTokens"`
	FailedTokens         int              `json:"failedTokens"`
	FailedTokenRate      float64          `json:"failedTokenRate"`
	AllTerminatedBounded bool             `json:"allTerminatedBounded"`
	AllCleanupReleased   bool             `json:"allCleanupReleased"`
	AllRecoveryReady     bool             `json:"allRecoveryReady"`
}

func Run(
	ctx context.Context,
	executable string,
	forwardDeadline time.Duration,
	scenarioBound time.Duration,
) (Summary, error) {
	if executable == "" {
		return Summary{}, errors.New("fault worker executable is required")
	}
	if forwardDeadline <= 0 || scenarioBound <= forwardDeadline {
		return Summary{}, errors.New("scenario bound must exceed the positive forward deadline")
	}
	summary := Summary{
		SchemaVersion: "1", ForwardDeadlineMS: forwardDeadline.Milliseconds(),
		ScenarioBoundMS:      scenarioBound.Milliseconds(),
		AllTerminatedBounded: true, AllCleanupReleased: true, AllRecoveryReady: true,
	}
	for _, scenario := range []struct {
		name            string
		expectedFailure bool
		networkDrop     bool
	}{
		{name: "pause", expectedFailure: true},
		{name: "kill", expectedFailure: true},
		{name: "delay", expectedFailure: true},
		{name: "disconnect", expectedFailure: true, networkDrop: true},
		{name: "jitter"},
	} {
		result, err := runScenario(
			ctx, executable, scenario.name, scenario.expectedFailure,
			scenario.networkDrop, forwardDeadline, scenarioBound,
		)
		if err != nil {
			return summary, fmt.Errorf("%s scenario: %w", scenario.name, err)
		}
		summary.Scenarios = append(summary.Scenarios, result)
		summary.AcceptedTokens += result.AcceptedTokens
		summary.FailedTokens += result.FailedTokens
		summary.AllTerminatedBounded = summary.AllTerminatedBounded &&
			result.TerminatedInMillis <= result.BoundMillis
		summary.AllCleanupReleased = summary.AllCleanupReleased && result.CleanupReleased
		summary.AllRecoveryReady = summary.AllRecoveryReady && result.RecoveryReady
	}
	if attempted := summary.AcceptedTokens + summary.FailedTokens; attempted > 0 {
		summary.FailedTokenRate = float64(summary.FailedTokens) / float64(attempted)
	}
	return summary, nil
}

func runScenario(
	parent context.Context,
	executable string,
	name string,
	expectedFailure bool,
	networkDrop bool,
	forwardDeadline time.Duration,
	scenarioBound time.Duration,
) (ScenarioResult, error) {
	ctx, cancel := context.WithTimeout(parent, scenarioBound)
	defer cancel()
	markerDirectory, err := os.MkdirTemp("", "mlx-swarm-fault-")
	if err != nil {
		return ScenarioResult{}, err
	}
	defer os.RemoveAll(markerDirectory)
	marker := markerDirectory + "/injected"

	producer, err := startFaultClient(executable, "producer", "", "")
	if err != nil {
		return ScenarioResult{}, err
	}
	defer cleanupClient(producer)
	reference, err := startFaultClient(executable, "reference", "", "")
	if err != nil {
		return ScenarioResult{}, err
	}
	defer cleanupClient(reference)
	consumerSupervisor, err := workerproc.NewPersistentSupervisor(func() (*workerproc.PersistentClient, error) {
		fault := name
		if networkDrop {
			fault = ""
		}
		return startFaultClient(executable, "consumer", fault, marker)
	})
	if err != nil {
		return ScenarioResult{}, err
	}
	defer cleanupSupervisor(consumerSupervisor)

	var consumer workerproc.PersistentCaller = consumerSupervisor
	var closeProxy func()
	if networkDrop {
		consumer, closeProxy, err = startDisconnectProxy(consumerSupervisor)
		if err != nil {
			return ScenarioResult{}, err
		}
		defer closeProxy()
	}

	session, err := generation.NewSession(
		ctx, producer, consumer, reference,
		generation.SessionConfig{
			Model: faultModelID, RTol: 1e-4, ATol: 1e-4,
			ForwardTimeout: forwardDeadline,
		},
	)
	if err != nil {
		return ScenarioResult{}, fmt.Errorf("prepare session: %w", err)
	}
	started := time.Now()
	result, generationErr := session.Generate(ctx, generation.Request{
		Prompt: "deterministic fault plan", MaxTokens: 4,
		SequenceID: "fault-" + name, IgnoreEOS: true,
	})
	elapsed := time.Since(started)
	scenario := ScenarioResult{
		Name: name, ExpectedFailure: expectedFailure,
		ObservedFailure:    generationErr != nil,
		TerminatedInMillis: elapsed.Milliseconds(), BoundMillis: scenarioBound.Milliseconds(),
		AcceptedTokens: len(result.GeneratedTokenIDs), Failure: result.Failure,
		WorkerRestarts: consumerSupervisor.RestartCount(),
	}
	if generationErr != nil {
		scenario.FailedTokens = 1
	}
	if expectedFailure != (generationErr != nil) {
		return scenario, fmt.Errorf("failure observed=%t, want %t: %v", generationErr != nil, expectedFailure, generationErr)
	}
	if elapsed > scenarioBound {
		return scenario, fmt.Errorf("termination took %v, bound %v", elapsed, scenarioBound)
	}
	if expectedFailure {
		if result.Failure == nil || result.Failure.Phase != "consumer_decode" ||
			result.Failure.Operation != "decode" || result.Failure.LastAcceptedTokenIndex != 0 {
			return scenario, fmt.Errorf("unexpected structured failure: %+v", result.Failure)
		}
	}

	scenario.CleanupReleased = noSequenceState(ctx, producer, consumer, reference)
	if !scenario.CleanupReleased {
		return scenario, errors.New("fault retained sequence or KV state")
	}
	recoverySession, err := generation.NewSession(
		ctx, producer, consumer, reference,
		generation.SessionConfig{
			Model: faultModelID, RTol: 1e-4, ATol: 1e-4,
			ForwardTimeout: forwardDeadline,
		},
	)
	if err == nil {
		var recovery generation.Result
		recovery, err = recoverySession.Generate(ctx, generation.Request{
			Prompt: "recovery", MaxTokens: 1,
			SequenceID: "recovery-" + name, IgnoreEOS: true,
		})
		if err == nil && (len(recovery.GeneratedTokenIDs) != 1 || recovery.Failure != nil) {
			err = errors.New("recovery did not produce one clean token")
		}
	}
	scenario.RecoveryReady = err == nil && noSequenceState(ctx, producer, consumer, reference)
	if !scenario.RecoveryReady {
		return scenario, fmt.Errorf("worker did not recover for a new sequence: %w", err)
	}
	return scenario, nil
}

func startFaultClient(
	executable string,
	role string,
	fault string,
	marker string,
) (*workerproc.PersistentClient, error) {
	args := []string{"fault-worker", "-role", role}
	if fault != "" {
		args = append(args, "-fault", fault, "-marker", marker)
	}
	return workerproc.StartPersistentCommand(executable, args...)
}

func noSequenceState(ctx context.Context, callers ...workerproc.PersistentCaller) bool {
	return smoke.RequireNoSequenceState(ctx, callers...) == nil
}

func cleanupClient(client *workerproc.PersistentClient) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	if err := client.Shutdown(ctx); err != nil {
		_ = client.Kill()
		_ = client.Wait(ctx)
	}
	cancel()
}

func cleanupSupervisor(supervisor *workerproc.PersistentSupervisor) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	if err := supervisor.Shutdown(ctx); err != nil {
		_ = supervisor.Kill()
		_ = supervisor.Wait(ctx)
	}
	cancel()
}

func startDisconnectProxy(
	caller workerproc.PersistentCaller,
) (workerproc.PersistentCaller, func(), error) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, nil, err
	}
	var disconnected atomic.Bool
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/worker/request", func(w http.ResponseWriter, request *http.Request) {
		var persistent workerproc.PersistentRequest
		if err := json.NewDecoder(request.Body).Decode(&persistent); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		response, callErr := caller.Call(request.Context(), persistent)
		if persistent.Command == "decode" && disconnected.CompareAndSwap(false, true) {
			// Apply the mutation, then drop its response. This makes cleanup
			// exercise the ambiguous-delivery case over a fresh connection.
			connection, _, err := w.(http.Hijacker).Hijack()
			if err == nil {
				_ = connection.Close()
				return
			}
		}
		w.Header().Set("Content-Type", "application/json")
		if callErr != nil {
			var workerError *workerproc.WorkerResponseError
			if errors.As(callErr, &workerError) {
				w.WriteHeader(http.StatusUnprocessableEntity)
				_ = json.NewEncoder(w).Encode(response)
				return
			}
			w.WriteHeader(http.StatusBadGateway)
			_ = json.NewEncoder(w).Encode(workerproc.PersistentResponse{Error: callErr.Error()})
			return
		}
		_ = json.NewEncoder(w).Encode(response)
	})
	server := &http.Server{Handler: mux, ReadHeaderTimeout: time.Second}
	go func() { _ = server.Serve(listener) }()
	client, err := workerproc.NewHTTPPersistentClient("http://"+listener.Addr().String(), nil)
	if err != nil {
		_ = listener.Close()
		return nil, nil, err
	}
	closeServer := func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		_ = server.Shutdown(ctx)
		cancel()
	}
	return client, closeServer, nil
}

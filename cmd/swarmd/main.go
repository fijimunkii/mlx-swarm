package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

const maxDebugTensorPayload = 64 << 20

type healthResponse struct {
	Status         string `json:"status"`
	Name           string `json:"name"`
	WorkerRestarts int    `json:"workerRestarts"`
}

type errorResponse struct {
	Error string `json:"error"`
}

func main() {
	if err := run(); err != nil {
		log.Printf("swarmd: %v", err)
		os.Exit(1)
	}
}

func run() error {
	addr := os.Getenv("SWARMD_ADDR")
	if addr == "" {
		addr = "127.0.0.1:8080"
	}
	exitAfterDebugShard := os.Getenv("SWARMD_EXIT_AFTER_DEBUG_SHARD") == "1"
	allowDebugComplete := os.Getenv("SWARMD_ALLOW_DEBUG_COMPLETE") == "1"
	debugShardComplete := make(chan struct{}, 1)

	worker := workerproc.Client{Path: workerproc.DefaultPath()}
	persistentWorker, err := workerproc.StartPersistentSupervisor(worker.Path)
	if err != nil {
		return fmt.Errorf("start persistent MLX worker: %w", err)
	}
	defer shutdownPersistentWorker(persistentWorker)
	startupContext, cancelStartup := context.WithTimeout(context.Background(), 10*time.Second)
	_, err = persistentWorker.Call(startupContext, workerproc.PersistentRequest{Command: "health"})
	cancelStartup()
	if err != nil {
		return fmt.Errorf("persistent MLX worker health: %w", err)
	}
	membershipConfig, membershipEnabled, err := membershipConfigFromEnvironment()
	if err != nil {
		return err
	}
	var membership *membershipAgent
	if membershipEnabled {
		membershipContext, cancelMembership := context.WithTimeout(context.Background(), 10*time.Second)
		membership, err = newMembershipAgent(
			membershipContext, membershipConfig, worker, persistentWorker,
		)
		if err == nil {
			err = membership.Register(membershipContext)
		}
		cancelMembership()
		if err != nil {
			return fmt.Errorf("join mesh membership: %w", err)
		}
		log.Printf(
			"swarmd registered worker %s instance %s with %s",
			membershipConfig.WorkerID, membershipConfig.InstanceID, membershipConfig.ControlURL,
		)
	}
	mux := http.NewServeMux()

	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		if _, err := persistentWorker.Call(
			r.Context(),
			workerproc.PersistentRequest{Command: "health"},
		); err != nil {
			writeError(w, http.StatusServiceUnavailable, err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(healthResponse{
			Status: "ok", Name: "mlx-swarm", WorkerRestarts: persistentWorker.RestartCount(),
		})
	})

	mux.HandleFunc("GET /v1/worker/state", func(w http.ResponseWriter, r *http.Request) {
		response, err := persistentWorker.Call(
			r.Context(),
			workerproc.PersistentRequest{Command: "state"},
		)
		writePersistentResponse(w, response, err)
	})

	// Trusted-network worker API. The daemon owns worker process shutdown;
	// clients own shard and sequence lifecycle through framed requests.
	mux.HandleFunc("POST /v1/worker/request", func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, maxDebugTensorPayload)
		var request workerproc.PersistentRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		if request.Command == "shutdown" {
			writeError(w, http.StatusForbidden, errors.New("swarmd owns worker shutdown"))
			return
		}
		response, err := forwardPersistentRequest(r.Context(), persistentWorker, request)
		writePersistentResponse(w, response, err)
	})

	mux.HandleFunc("GET /v1/debug/worker/capabilities", func(w http.ResponseWriter, r *http.Request) {
		result, err := worker.Run(r.Context(), []string{"capabilities"}, nil)
		if err != nil {
			writeError(w, http.StatusBadGateway, err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-MLX-Swarm-Worker-Micros", strconv.FormatInt(result.Duration.Microseconds(), 10))
		_, _ = w.Write(result.Output)
	})

	// Workflow-only rendezvous for paired ephemeral workers. It is disabled by
	// default and carries no model data; successful proof clients use it to let
	// the remote runner exit after they have read its final worker state.
	mux.HandleFunc(
		"POST /v1/debug/complete",
		debugCompleteHandler(allowDebugComplete, debugShardComplete),
	)

	// Debug-only endpoint. It proves that the tensor emitted by a worker on
	// one machine can traverse the Go network layer and be consumed by the
	// complementary worker range on another machine. It is intentionally not
	// the final public protocol and has no authentication or encryption.
	mux.HandleFunc("POST /v1/debug/shard/finish", func(w http.ResponseWriter, r *http.Request) {
		finishShard(w, r, worker, []string{"shard-finish-stdio"}, exitAfterDebugShard, debugShardComplete)
	})

	// Real-checkpoint counterpart to the deterministic fixture above. The
	// producer envelope carries its hidden state and memory measurements; this
	// machine independently loads the complementary range plus norm/lm_head.
	mux.HandleFunc("POST /v1/debug/checkpoint-shard/finish", func(w http.ResponseWriter, r *http.Request) {
		finishShard(w, r, worker, []string{"checkpoint-shard-finish-stdio"}, exitAfterDebugShard, debugShardComplete)
	})

	server := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      2 * time.Minute,
		IdleTimeout:       2 * time.Minute,
	}
	signalContext, stopSignals := signal.NotifyContext(
		context.Background(),
		os.Interrupt,
		syscall.SIGTERM,
	)
	defer stopSignals()
	var membershipFailure <-chan error
	var membershipStopped <-chan struct{}
	stopMembership := func() {}
	if membership != nil {
		membershipContext, cancelMembership := context.WithCancel(context.Background())
		failures := make(chan error, 1)
		stopped := make(chan struct{})
		membershipFailure = failures
		membershipStopped = stopped
		stopMembership = cancelMembership
		go func() {
			defer close(stopped)
			if err := membership.Run(membershipContext); err != nil {
				failures <- err
			}
		}()
	}
	serverStopped := make(chan struct{})
	shutdownComplete := make(chan struct{})
	var membershipErr error
	var debugShutdown <-chan struct{}
	if exitAfterDebugShard || allowDebugComplete {
		debugShutdown = debugShardComplete
	}
	go func() {
		defer close(shutdownComplete)
		shutdownReason := ""
		select {
		case <-signalContext.Done():
			shutdownReason = "signal"
		case <-debugShutdown:
			shutdownReason = "debug one-shot"
		case membershipErr = <-membershipFailure:
			shutdownReason = "membership failure"
		case <-serverStopped:
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := server.Shutdown(ctx); err != nil {
			log.Printf("%s shutdown: %v", shutdownReason, err)
		}
	}()

	log.Printf("swarmd listening on %s", addr)
	serveErr := server.ListenAndServe()
	close(serverStopped)
	stopMembership()
	if membershipStopped != nil {
		<-membershipStopped
	}
	<-shutdownComplete
	if membershipErr == nil && membershipFailure != nil {
		select {
		case membershipErr = <-membershipFailure:
		default:
		}
	}
	if membershipErr != nil {
		return fmt.Errorf("membership: %w", membershipErr)
	}
	if serveErr != nil && serveErr != http.ErrServerClosed {
		return fmt.Errorf("serve: %w", serveErr)
	}
	return nil
}

func debugCompleteHandler(enabled bool, complete chan<- struct{}) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		if !enabled {
			writeError(w, http.StatusForbidden, errors.New("debug completion is disabled"))
			return
		}
		w.WriteHeader(http.StatusNoContent)
		select {
		case complete <- struct{}{}:
		default:
		}
	}
}

func forwardPersistentRequest(
	ctx context.Context,
	worker workerproc.PersistentCaller,
	request workerproc.PersistentRequest,
) (workerproc.PersistentResponse, error) {
	requestContext, cancel, prepared, err := workerproc.RequestContext(ctx, request)
	if err != nil {
		return workerproc.PersistentResponse{}, err
	}
	defer cancel()
	request = prepared
	callerRequestID := request.RequestID
	request.RequestID = ""
	response, err := worker.Call(requestContext, request)
	if callerRequestID != "" {
		response.RequestID = callerRequestID
	}
	return response, err
}

type persistentWorkerLifecycle interface {
	Shutdown(context.Context) error
	Kill() error
	Wait(context.Context) error
}

func shutdownPersistentWorker(worker persistentWorkerLifecycle) {
	shutdownContext, cancelShutdown := context.WithTimeout(context.Background(), 5*time.Second)
	err := worker.Shutdown(shutdownContext)
	cancelShutdown()
	if err == nil {
		return
	}
	log.Printf("persistent MLX worker shutdown: %v", err)
	if killErr := worker.Kill(); killErr != nil && !errors.Is(killErr, os.ErrProcessDone) {
		log.Printf("persistent MLX worker kill fallback: %v", killErr)
		return
	}
	reapContext, cancelReap := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancelReap()
	if waitErr := worker.Wait(reapContext); errors.Is(waitErr, context.DeadlineExceeded) {
		log.Printf("persistent MLX worker reap: %v", waitErr)
	}
}

func finishShard(
	w http.ResponseWriter,
	r *http.Request,
	worker workerproc.Client,
	workerArgs []string,
	exitAfter bool,
	complete chan<- struct{},
) {
	r.Body = http.MaxBytesReader(w, r.Body, maxDebugTensorPayload)
	payload, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if len(payload) == 0 {
		writeError(w, http.StatusBadRequest, io.ErrUnexpectedEOF)
		return
	}

	result, err := worker.Run(r.Context(), workerArgs, payload)
	if err != nil {
		writeError(w, http.StatusBadGateway, err)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-MLX-Swarm-Worker-Micros", strconv.FormatInt(result.Duration.Microseconds(), 10))
	_, _ = w.Write(result.Output)

	if exitAfter {
		select {
		case complete <- struct{}{}:
		default:
		}
	}
}

func writeError(w http.ResponseWriter, status int, err error) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(errorResponse{Error: err.Error()})
}

func writePersistentResponse(
	w http.ResponseWriter,
	response workerproc.PersistentResponse,
	err error,
) {
	if err != nil {
		var workerError *workerproc.WorkerResponseError
		if !errors.As(err, &workerError) {
			writeError(w, http.StatusBadGateway, err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnprocessableEntity)
		_ = json.NewEncoder(w).Encode(response)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(response)
}

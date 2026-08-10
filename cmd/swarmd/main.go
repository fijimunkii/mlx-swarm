package main

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

const maxDebugTensorPayload = 64 << 20

type healthResponse struct {
	Status string `json:"status"`
	Name   string `json:"name"`
}

type errorResponse struct {
	Error string `json:"error"`
}

func main() {
	addr := os.Getenv("SWARMD_ADDR")
	if addr == "" {
		addr = "127.0.0.1:8080"
	}

	worker := workerproc.Client{Path: workerproc.DefaultPath()}
	mux := http.NewServeMux()

	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(healthResponse{Status: "ok", Name: "mlx-swarm"})
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

	// Debug-only v0 endpoint. It proves that the tensor emitted by a worker on
	// one machine can traverse the Go network layer and be consumed by the
	// complementary worker range on another machine. It is intentionally not
	// the final public protocol and has no authentication or encryption.
	mux.HandleFunc("POST /v1/debug/shard/finish", func(w http.ResponseWriter, r *http.Request) {
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

		result, err := worker.Run(r.Context(), []string{"shard-finish-stdio"}, payload)
		if err != nil {
			writeError(w, http.StatusBadGateway, err)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-MLX-Swarm-Worker-Micros", strconv.FormatInt(result.Duration.Microseconds(), 10))
		_, _ = w.Write(result.Output)
	})

	server := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      2 * time.Minute,
		IdleTimeout:       2 * time.Minute,
	}

	log.Printf("swarmd listening on %s", addr)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}

func writeError(w http.ResponseWriter, status int, err error) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(errorResponse{Error: err.Error()})
}

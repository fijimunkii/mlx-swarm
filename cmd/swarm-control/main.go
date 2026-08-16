package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/registry"
)

type healthResponse struct {
	Status            string `json:"status"`
	SchemaVersion     int    `json:"schemaVersion"`
	InventoryRevision uint64 `json:"inventoryRevision"`
	WorkerCount       int    `json:"workerCount"`
}

func main() {
	if err := run(); err != nil {
		log.Printf("swarm-control: %v", err)
		os.Exit(1)
	}
}

func run() error {
	addr := os.Getenv("SWARM_CONTROL_ADDR")
	if addr == "" {
		addr = "127.0.0.1:9090"
	}
	leaseTTL := registry.DefaultLeaseTTL
	if value := os.Getenv("SWARM_CONTROL_LEASE_TTL"); value != "" {
		parsed, err := time.ParseDuration(value)
		if err != nil || parsed < time.Second {
			return fmt.Errorf("SWARM_CONTROL_LEASE_TTL must be at least 1s")
		}
		leaseTTL = parsed
	}

	membership := registry.New(leaseTTL)
	server := &http.Server{
		Addr:              addr,
		Handler:           newControlHandler(membership),
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      15 * time.Second,
		IdleTimeout:       time.Minute,
	}
	signalContext, stopSignals := signal.NotifyContext(
		context.Background(), os.Interrupt, syscall.SIGTERM,
	)
	defer stopSignals()
	shutdownComplete := make(chan struct{})
	go func() {
		defer close(shutdownComplete)
		<-signalContext.Done()
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := server.Shutdown(ctx); err != nil {
			log.Printf("shutdown: %v", err)
		}
	}()

	log.Printf("swarm-control listening on %s with %s worker leases", addr, leaseTTL)
	err := server.ListenAndServe()
	stopSignals()
	<-shutdownComplete
	if err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("serve: %w", err)
	}
	return nil
}

func newControlHandler(membership *registry.Registry) http.Handler {
	mux := http.NewServeMux()
	membershipHandler := registry.NewHTTPHandler(membership)
	mux.Handle("/v1/membership", membershipHandler)
	mux.Handle("/v1/membership/", membershipHandler)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		inventory := membership.Snapshot()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(healthResponse{
			Status: "ok", SchemaVersion: registry.SchemaVersion,
			InventoryRevision: inventory.Revision, WorkerCount: len(inventory.Workers),
		})
	})
	return mux
}

package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/registry"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

const defaultMembershipHeartbeatInterval = 10 * time.Second

var membershipOperations = []string{
	"closeSequence", "decode", "detokenize", "forward", "health", "loadShard",
	"modelInfo", "openSequence", "prefill", "state", "tokenize", "unloadShard",
}

type membershipConfig struct {
	ControlURL        string
	WorkerID          string
	InstanceID        string
	PublicURL         string
	Backend           string
	HeartbeatInterval time.Duration
}

type capabilityRunner interface {
	Run(context.Context, []string, []byte) (workerproc.Result, error)
}

type membershipWorker interface {
	workerproc.PersistentCaller
	RestartCount() int
}

type localCapabilities struct {
	Runtime                   string   `json:"runtime"`
	Device                    string   `json:"device"`
	CheckpointShardModelTypes []string `json:"checkpointShardModelTypes"`
	PhysicalMemoryBytes       uint64   `json:"physicalMemoryBytes"`
}

type membershipAgent struct {
	client          *registry.Client
	worker          membershipWorker
	registration    registry.Registration
	interval        time.Duration
	statusPending   bool
	statusSampledAt time.Time
	freshnessWindow time.Duration
}

func membershipConfigFromEnvironment() (membershipConfig, bool, error) {
	controlURL := os.Getenv("SWARMD_CONTROL_URL")
	if controlURL == "" {
		return membershipConfig{}, false, nil
	}
	config := membershipConfig{
		ControlURL:        controlURL,
		WorkerID:          os.Getenv("SWARMD_WORKER_ID"),
		InstanceID:        os.Getenv("SWARMD_INSTANCE_ID"),
		PublicURL:         os.Getenv("SWARMD_PUBLIC_URL"),
		Backend:           os.Getenv("SWARMD_BACKEND"),
		HeartbeatInterval: defaultMembershipHeartbeatInterval,
	}
	if config.WorkerID == "" || config.PublicURL == "" {
		return membershipConfig{}, false, errors.New(
			"SWARMD_WORKER_ID and SWARMD_PUBLIC_URL are required when SWARMD_CONTROL_URL is set",
		)
	}
	if config.Backend == "" {
		config.Backend = "mlx"
	}
	if config.InstanceID == "" {
		generated, err := randomInstanceID()
		if err != nil {
			return membershipConfig{}, false, err
		}
		config.InstanceID = generated
	}
	if value := os.Getenv("SWARMD_HEARTBEAT_INTERVAL"); value != "" {
		parsed, err := time.ParseDuration(value)
		if err != nil || parsed < time.Second {
			return membershipConfig{}, false, errors.New("SWARMD_HEARTBEAT_INTERVAL must be at least 1s")
		}
		config.HeartbeatInterval = parsed
	}
	return config, true, nil
}

func newMembershipAgent(
	ctx context.Context,
	config membershipConfig,
	runner capabilityRunner,
	worker membershipWorker,
) (*membershipAgent, error) {
	client, err := registry.NewClient(config.ControlURL, &http.Client{Timeout: 10 * time.Second})
	if err != nil {
		return nil, err
	}
	publicURL, err := url.Parse(strings.TrimRight(config.PublicURL, "/"))
	if err != nil || (publicURL.Scheme != "http" && publicURL.Scheme != "https") || publicURL.Host == "" {
		return nil, errors.New("SWARMD_PUBLIC_URL must be an http(s) URL")
	}
	result, err := runner.Run(ctx, []string{"capabilities"}, nil)
	if err != nil {
		return nil, fmt.Errorf("worker capabilities: %w", err)
	}
	var capabilities localCapabilities
	if err := json.Unmarshal(result.Output, &capabilities); err != nil {
		return nil, fmt.Errorf("decode worker capabilities: %w", err)
	}
	state, err := workerproc.State(ctx, worker)
	if err != nil {
		return nil, fmt.Errorf("worker membership state: %w", err)
	}
	if capabilities.PhysicalMemoryBytes == 0 {
		capabilities.PhysicalMemoryBytes = state.PhysicalMemoryBytes
	}
	if state.RetainedByteBudget <= 0 || state.MaxOpenSequencesPerShard <= 0 {
		return nil, errors.New("worker state has incomplete admission limits")
	}
	status := membershipStatus(state, worker.RestartCount(), capabilities.PhysicalMemoryBytes)
	registration := registry.Registration{
		SchemaVersion: registry.SchemaVersion,
		ID:            config.WorkerID, InstanceID: config.InstanceID, Endpoint: publicURL.String(),
		Capabilities: registry.Capabilities{
			Backend: config.Backend, Runtime: capabilities.Runtime,
			OS: runtime.GOOS, Architecture: runtime.GOARCH, Device: capabilities.Device,
			PhysicalMemoryBytes: capabilities.PhysicalMemoryBytes,
			Adapters:            append([]string(nil), capabilities.CheckpointShardModelTypes...),
			Operations:          append([]string(nil), membershipOperations...),
			Admission: registry.AdmissionLimits{
				MaxConcurrentRequests:    1,
				MaxOpenSequencesPerShard: state.MaxOpenSequencesPerShard,
				RetainedByteBudget:       uint64(state.RetainedByteBudget),
			},
			Transports: []registry.Transport{
				{
					Protocol:        "http-json-v1",
					TensorEncodings: []string{workerproc.Base64JSONTensorEncoding},
					MaxRequestBytes: maxDebugTensorPayload, TLS: publicURL.Scheme == "https",
				},
				{
					Protocol:        workerproc.InstanceBoundHTTPProtocol,
					TensorEncodings: []string{workerproc.Base64JSONTensorEncoding},
					MaxRequestBytes: maxDebugTensorPayload, TLS: publicURL.Scheme == "https",
				},
			},
		},
		Status: status,
	}
	return &membershipAgent{
		client: client, worker: worker, registration: registration, interval: config.HeartbeatInterval,
		statusPending: true, statusSampledAt: time.Now(),
	}, nil
}

func (agent *membershipAgent) Register(ctx context.Context) error {
	statusFresh := agent.freshStatusPending(time.Now())
	mutation, err := agent.client.Register(ctx, agent.registration, statusFresh)
	if err != nil {
		return err
	}
	leaseTTL := mutation.Worker.ExpiresAt.Sub(mutation.Worker.LastSeen)
	if agent.interval >= leaseTTL {
		_ = agent.client.Remove(ctx, agent.registration.ID, agent.registration.InstanceID)
		return fmt.Errorf("heartbeat interval %s must be shorter than lease TTL %s", agent.interval, leaseTTL)
	}
	agent.freshnessWindow = leaseTTL
	agent.statusPending = false
	return nil
}

func (agent *membershipAgent) Run(ctx context.Context) error {
	ticker := time.NewTicker(agent.interval)
	defer ticker.Stop()
	defer agent.remove()
	probeResults := make(chan registry.Status, 1)
	probeInFlight := false
	for {
		select {
		case <-ctx.Done():
			return nil
		case status := <-probeResults:
			agent.recordStatus(status)
			probeInFlight = false
		case <-ticker.C:
			if !probeInFlight {
				probeInFlight = true
				previous := agent.registration.Status
				go func() {
					status := agent.probeStatus(ctx, previous)
					select {
					case probeResults <- status:
					case <-ctx.Done():
					}
				}()
			}
			if err := agent.heartbeat(ctx); err != nil {
				return err
			}
		}
	}
}

func (agent *membershipAgent) recordStatus(status registry.Status) {
	agent.registration.Status = status
	agent.statusPending = true
	agent.statusSampledAt = time.Now()
}

func (agent *membershipAgent) freshStatusPending(now time.Time) bool {
	if !agent.statusPending {
		return false
	}
	if agent.statusSampledAt.IsZero() || agent.statusSampledAt.After(now) ||
		(agent.freshnessWindow > 0 && now.Sub(agent.statusSampledAt) > agent.freshnessWindow) {
		agent.statusPending = false
		return false
	}
	return true
}

func (agent *membershipAgent) probeStatus(ctx context.Context, previous registry.Status) registry.Status {
	state, stateErr := workerproc.State(ctx, agent.worker)
	if stateErr != nil {
		previous.Health = registry.HealthDegraded
		previous.AvailableMemoryBytes = 0
		previous.RestartCount = agent.worker.RestartCount()
		previous.RecentFailureCount++
		return previous
	}
	status := membershipStatus(
		state, agent.worker.RestartCount(), agent.registration.Capabilities.PhysicalMemoryBytes,
	)
	status.RecentFailureCount = previous.RecentFailureCount
	return status
}

func (agent *membershipAgent) heartbeat(ctx context.Context) error {
	heartbeat := registry.Heartbeat{
		SchemaVersion: registry.SchemaVersion,
		InstanceID:    agent.registration.InstanceID,
	}
	statusFresh := agent.freshStatusPending(time.Now())
	if statusFresh {
		status := agent.registration.Status
		heartbeat.Status = &status
	}
	_, err := agent.client.Heartbeat(ctx, agent.registration.ID, heartbeat)
	if err == nil {
		if statusFresh {
			agent.statusPending = false
		}
		return nil
	}
	var remote *registry.RemoteError
	if errors.As(err, &remote) && (remote.Code == "worker_not_found" || remote.Code == "lease_expired") {
		if registerErr := agent.Register(ctx); registerErr != nil {
			var registerRemote *registry.RemoteError
			if errors.As(registerErr, &registerRemote) && registerRemote.StatusCode < 500 {
				return fmt.Errorf("membership re-registration rejected: %w", registerErr)
			}
			log.Printf("swarmd membership re-registration: %v", registerErr)
		}
		return nil
	}
	if errors.As(err, &remote) && (remote.Code == "stale_instance" || remote.Code == "duplicate_worker") {
		return fmt.Errorf("membership lease lost: %w", err)
	}
	if errors.As(err, &remote) && remote.StatusCode < 500 {
		return fmt.Errorf("membership heartbeat rejected: %w", err)
	}
	log.Printf("swarmd membership heartbeat: %v", err)
	return nil
}

func (agent *membershipAgent) remove() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	err := agent.client.Remove(ctx, agent.registration.ID, agent.registration.InstanceID)
	var remote *registry.RemoteError
	if err != nil && (!errors.As(err, &remote) ||
		(remote.Code != "worker_not_found" && remote.Code != "lease_expired")) {
		log.Printf("swarmd membership removal: %v", err)
	}
}

func membershipStatus(
	state *workerproc.PersistentWorkerState,
	restartCount int,
	physicalMemoryBytes uint64,
) registry.Status {
	allocatable := physicalMemoryBytes
	if state.MLXMemoryLimitBytes > 0 && uint64(state.MLXMemoryLimitBytes) < allocatable {
		allocatable = uint64(state.MLXMemoryLimitBytes)
	}
	allocated := uint64(max(0, state.Memory.ActiveBytes+state.Memory.CacheBytes))
	available := uint64(0)
	if allocated < allocatable {
		available = allocatable - allocated
	}
	pressure := state.Memory.ProcessPhysicalBytes
	if pressure == 0 {
		pressure = allocated
	}
	shards := make([]registry.RetainedShard, len(state.LoadedShards))
	openSequences := 0
	for index, shard := range state.LoadedShards {
		memoryBytes := uint64(max(0, shard.LoadedMemory.ActiveBytes+shard.LoadedMemory.CacheBytes))
		shards[index] = registry.RetainedShard{
			ID: shard.ShardID, ModelID: shard.ModelID,
			CheckpointFingerprint: shard.CheckpointFingerprint,
			LayerStart:            shard.LayerStart, LayerEnd: shard.LayerEnd,
			OwnsInput: shard.OwnsInput, OwnsOutput: shard.OwnsOutput,
			MemoryBytes: memoryBytes, OpenSequenceCount: shard.OpenSequenceCount,
		}
		openSequences += shard.OpenSequenceCount
	}
	return registry.Status{
		Health:               registry.HealthHealthy,
		AvailableMemoryBytes: available, MemoryPressureBytes: pressure,
		OpenSequenceCount: openSequences, RetainedBytes: uint64(max(0, state.RetainedBytes)),
		RestartCount: restartCount, RetainedShards: shards,
	}
}

func randomInstanceID() (string, error) {
	bytes := make([]byte, 16)
	if _, err := rand.Read(bytes); err != nil {
		return "", fmt.Errorf("generate worker instance ID: %w", err)
	}
	return hex.EncodeToString(bytes), nil
}

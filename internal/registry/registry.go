package registry

import (
	"errors"
	"fmt"
	"net/url"
	"reflect"
	"slices"
	"strings"
	"sync"
	"time"
)

const DefaultLeaseTTL = 30 * time.Second

var (
	ErrDuplicateWorker   = errors.New("worker ID is already leased by another instance")
	ErrDuplicateInstance = errors.New("worker instance is already registered under another ID")
	ErrDuplicateEndpoint = errors.New("worker endpoint is already registered under another ID")
	ErrWorkerNotFound    = errors.New("worker is not registered")
	ErrStaleInstance     = errors.New("worker instance does not own the active lease")
	ErrLeaseExpired      = errors.New("worker lease has expired")
)

type Option func(*Registry)

func WithClock(now func() time.Time) Option {
	return func(registry *Registry) {
		if now != nil {
			registry.now = now
		}
	}
}

type Registry struct {
	mu       sync.RWMutex
	workers  map[string]Worker
	revision uint64
	ttl      time.Duration
	now      func() time.Time
}

func New(ttl time.Duration, options ...Option) *Registry {
	if ttl <= 0 {
		ttl = DefaultLeaseTTL
	}
	registry := &Registry{
		workers: make(map[string]Worker),
		ttl:     ttl,
		now:     time.Now,
	}
	for _, option := range options {
		option(registry)
	}
	return registry
}

func (r *Registry) Register(input Registration) (Mutation, error) {
	input = cloneRegistration(input)
	normalizeRegistration(&input)
	if err := validateRegistration(input); err != nil {
		return Mutation{}, err
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	now := r.now().UTC()
	r.expireLocked(now)
	for id, existing := range r.workers {
		if id == input.ID {
			continue
		}
		if existing.InstanceID == input.InstanceID {
			return Mutation{}, fmt.Errorf("%w: %q", ErrDuplicateInstance, input.InstanceID)
		}
		if existing.Endpoint == input.Endpoint {
			return Mutation{}, fmt.Errorf("%w: %q", ErrDuplicateEndpoint, input.Endpoint)
		}
	}
	if existing, ok := r.workers[input.ID]; ok {
		if existing.InstanceID != input.InstanceID {
			return Mutation{}, fmt.Errorf("%w: %q", ErrDuplicateWorker, input.ID)
		}
		worker := Worker{
			Registration: input,
			RegisteredAt: existing.RegisteredAt,
			LastSeen:     now,
			ExpiresAt:    now.Add(r.ttl),
		}
		r.workers[input.ID] = worker
		if !reflect.DeepEqual(existing.Registration, input) {
			r.revision++
		}
		return Mutation{InventoryRevision: r.revision, Worker: cloneWorker(worker)}, nil
	}

	worker := Worker{
		Registration: input,
		RegisteredAt: now,
		LastSeen:     now,
		ExpiresAt:    now.Add(r.ttl),
	}
	r.workers[input.ID] = worker
	r.revision++
	return Mutation{InventoryRevision: r.revision, Worker: cloneWorker(worker)}, nil
}

func (r *Registry) Heartbeat(id string, heartbeat Heartbeat) (Mutation, error) {
	normalizeStatus(&heartbeat.Status)
	if err := validateIdentity(id, heartbeat.SchemaVersion, heartbeat.InstanceID); err != nil {
		return Mutation{}, err
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	now := r.now().UTC()
	existing, ok := r.workers[id]
	if ok && !now.Before(existing.ExpiresAt) {
		delete(r.workers, id)
		r.revision++
		r.expireLocked(now)
		return Mutation{}, fmt.Errorf("%w: %q", ErrLeaseExpired, id)
	}
	r.expireLocked(now)
	if !ok {
		return Mutation{}, fmt.Errorf("%w: %q", ErrWorkerNotFound, id)
	}
	if existing.InstanceID != heartbeat.InstanceID {
		return Mutation{}, fmt.Errorf("%w: %q", ErrStaleInstance, id)
	}
	if err := validateStatus(heartbeat.Status, existing.Capabilities); err != nil {
		return Mutation{}, err
	}
	statusChanged := !reflect.DeepEqual(existing.Status, heartbeat.Status)
	existing.Status = cloneStatus(heartbeat.Status)
	existing.LastSeen = now
	existing.ExpiresAt = now.Add(r.ttl)
	r.workers[id] = existing
	if statusChanged {
		r.revision++
	}
	return Mutation{InventoryRevision: r.revision, Worker: cloneWorker(existing)}, nil
}

func (r *Registry) Remove(id, instanceID string) error {
	if !validIdentityPart(id) || !validIdentityPart(instanceID) {
		return errors.New("worker ID and instance ID are invalid")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	now := r.now().UTC()
	existing, ok := r.workers[id]
	if ok && !now.Before(existing.ExpiresAt) {
		delete(r.workers, id)
		r.revision++
		r.expireLocked(now)
		return fmt.Errorf("%w: %q", ErrLeaseExpired, id)
	}
	r.expireLocked(now)
	if !ok {
		return fmt.Errorf("%w: %q", ErrWorkerNotFound, id)
	}
	if existing.InstanceID != instanceID {
		return fmt.Errorf("%w: %q", ErrStaleInstance, id)
	}
	delete(r.workers, id)
	r.revision++
	return nil
}

func (r *Registry) Snapshot() Inventory {
	r.mu.Lock()
	defer r.mu.Unlock()
	now := r.now().UTC()
	r.expireLocked(now)

	workers := make([]Worker, 0, len(r.workers))
	for _, worker := range r.workers {
		workers = append(workers, cloneWorker(worker))
	}
	slices.SortFunc(workers, func(left, right Worker) int {
		return strings.Compare(left.ID, right.ID)
	})
	return Inventory{
		SchemaVersion: SchemaVersion,
		Revision:      r.revision, GeneratedAt: now,
		LeaseTTLMillis: r.ttl.Milliseconds(), Workers: workers,
	}
}

func (r *Registry) expireLocked(now time.Time) {
	removed := false
	for id, worker := range r.workers {
		if !now.Before(worker.ExpiresAt) {
			delete(r.workers, id)
			removed = true
		}
	}
	if removed {
		r.revision++
	}
}

func validateRegistration(input Registration) error {
	if err := validateIdentity(input.ID, input.SchemaVersion, input.InstanceID); err != nil {
		return err
	}
	endpoint, err := url.Parse(input.Endpoint)
	if err != nil || (endpoint.Scheme != "http" && endpoint.Scheme != "https") ||
		endpoint.Host == "" || endpoint.User != nil || endpoint.RawQuery != "" || endpoint.Fragment != "" {
		return errors.New("worker endpoint must be an http(s) URL without credentials, a query, or a fragment")
	}
	capabilities := input.Capabilities
	switch {
	case strings.TrimSpace(capabilities.Backend) == "":
		return errors.New("worker backend is required")
	case strings.TrimSpace(capabilities.Runtime) == "":
		return errors.New("worker runtime is required")
	case strings.TrimSpace(capabilities.OS) == "" || strings.TrimSpace(capabilities.Architecture) == "":
		return errors.New("worker OS and architecture are required")
	case strings.TrimSpace(capabilities.Device) == "":
		return errors.New("worker device is required")
	case capabilities.PhysicalMemoryBytes == 0:
		return errors.New("worker physical memory must be positive")
	case len(capabilities.Adapters) == 0:
		return errors.New("worker must advertise at least one adapter")
	case len(capabilities.Operations) == 0:
		return errors.New("worker must advertise at least one operation")
	case capabilities.Admission.MaxConcurrentRequests <= 0:
		return errors.New("worker max concurrent requests must be positive")
	case capabilities.Admission.MaxOpenSequencesPerShard <= 0:
		return errors.New("worker max open sequences must be positive")
	case capabilities.Admission.RetainedByteBudget == 0:
		return errors.New("worker retained byte budget must be positive")
	case len(capabilities.Transports) == 0:
		return errors.New("worker must advertise at least one transport")
	}
	for index, transport := range capabilities.Transports {
		if strings.TrimSpace(transport.Protocol) == "" || len(transport.TensorEncodings) == 0 ||
			transport.MaxRequestBytes <= 0 {
			return fmt.Errorf("worker transport %d is incomplete", index)
		}
		if index > 0 && capabilities.Transports[index-1].Protocol == transport.Protocol {
			return fmt.Errorf("worker transport protocol %q is duplicated", transport.Protocol)
		}
	}
	return validateStatus(input.Status, capabilities)
}

func validateIdentity(id string, schemaVersion int, instanceID string) error {
	switch {
	case schemaVersion != SchemaVersion:
		return fmt.Errorf("membership schema version is %d, want %d", schemaVersion, SchemaVersion)
	case !validIdentityPart(id):
		return errors.New("worker ID must use 1-128 ASCII letters, digits, dots, underscores, colons, or hyphens")
	case !validIdentityPart(instanceID):
		return errors.New("worker instance ID must use 1-128 ASCII letters, digits, dots, underscores, colons, or hyphens")
	}
	return nil
}

func validIdentityPart(value string) bool {
	if len(value) == 0 || len(value) > 128 {
		return false
	}
	for _, character := range value {
		if (character >= 'a' && character <= 'z') ||
			(character >= 'A' && character <= 'Z') ||
			(character >= '0' && character <= '9') ||
			character == '.' || character == '_' || character == ':' || character == '-' {
			continue
		}
		return false
	}
	return true
}

func validateStatus(status Status, capabilities Capabilities) error {
	switch status.Health {
	case HealthHealthy, HealthDegraded, HealthDraining:
	default:
		return fmt.Errorf("unsupported worker health %q", status.Health)
	}
	if status.AvailableMemoryBytes > capabilities.PhysicalMemoryBytes {
		return errors.New("available memory exceeds physical memory")
	}
	if status.OpenSequenceCount < 0 || status.RestartCount < 0 || status.RecentFailureCount < 0 {
		return errors.New("worker counters cannot be negative")
	}
	openSequences := 0
	for index, shard := range status.RetainedShards {
		if shard.ID == "" || shard.ModelID == "" || shard.CheckpointFingerprint == "" ||
			shard.LayerStart < 0 || shard.LayerEnd <= shard.LayerStart ||
			shard.OpenSequenceCount < 0 {
			return fmt.Errorf("retained shard %d is incomplete", index)
		}
		if index > 0 && status.RetainedShards[index-1].ID == shard.ID {
			return fmt.Errorf("retained shard ID %q is duplicated", shard.ID)
		}
		openSequences += shard.OpenSequenceCount
	}
	if status.OpenSequenceCount != openSequences {
		return fmt.Errorf(
			"worker open sequence count is %d, retained shards report %d",
			status.OpenSequenceCount, openSequences,
		)
	}
	return nil
}

func normalizeRegistration(input *Registration) {
	input.Endpoint = strings.TrimRight(input.Endpoint, "/")
	input.Capabilities.Adapters = normalizeStrings(input.Capabilities.Adapters)
	input.Capabilities.Operations = normalizeStrings(input.Capabilities.Operations)
	input.Capabilities.CheckpointFingerprints = normalizeStrings(input.Capabilities.CheckpointFingerprints)
	for index := range input.Capabilities.Transports {
		input.Capabilities.Transports[index].TensorEncodings = normalizeStrings(
			input.Capabilities.Transports[index].TensorEncodings,
		)
	}
	slices.SortFunc(input.Capabilities.Transports, func(left, right Transport) int {
		return strings.Compare(left.Protocol, right.Protocol)
	})
	normalizeStatus(&input.Status)
}

func normalizeStatus(status *Status) {
	status.RetainedShards = append([]RetainedShard(nil), status.RetainedShards...)
	slices.SortFunc(status.RetainedShards, func(left, right RetainedShard) int {
		return strings.Compare(left.ID, right.ID)
	})
}

func normalizeStrings(values []string) []string {
	result := append([]string(nil), values...)
	for index := range result {
		result[index] = strings.TrimSpace(result[index])
	}
	slices.Sort(result)
	result = slices.Compact(result)
	return slices.DeleteFunc(result, func(value string) bool { return value == "" })
}

func cloneRegistration(input Registration) Registration {
	input.Capabilities.Adapters = append([]string(nil), input.Capabilities.Adapters...)
	input.Capabilities.Operations = append([]string(nil), input.Capabilities.Operations...)
	input.Capabilities.CheckpointFingerprints = append(
		[]string(nil), input.Capabilities.CheckpointFingerprints...,
	)
	input.Capabilities.Transports = append([]Transport(nil), input.Capabilities.Transports...)
	for index := range input.Capabilities.Transports {
		input.Capabilities.Transports[index].TensorEncodings = append(
			[]string(nil), input.Capabilities.Transports[index].TensorEncodings...,
		)
	}
	input.Status = cloneStatus(input.Status)
	return input
}

func cloneStatus(status Status) Status {
	status.RetainedShards = append([]RetainedShard(nil), status.RetainedShards...)
	return status
}

func cloneWorker(worker Worker) Worker {
	worker.Registration = cloneRegistration(worker.Registration)
	return worker
}

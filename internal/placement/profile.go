package placement

import (
	"errors"
	"fmt"
	"math"
	"math/bits"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/generation"
)

const (
	DefaultProfileMaxAge              = 5 * time.Minute
	DefaultProfileMaxSamplesPerSeries = 64
	DefaultProfileMaxSeries           = 4096
	maxProfileLabelBytes              = 1024
)

// ProfileConfig bounds rolling topology and compute evidence. Zero values use
// conservative defaults so callers cannot accidentally create unbounded stores.
type ProfileConfig struct {
	MaxAge              time.Duration
	MaxSamplesPerSeries int
	MaxSeries           int
}

// LinkObservation is one directional topology measurement. RTTMicros is
// always required. PayloadBytes and ElapsedMicros are either both zero for an
// RTT-only probe or both positive for an effective transfer measurement.
type LinkObservation struct {
	SourceID         string    `json:"sourceID"`
	SourceInstanceID string    `json:"sourceInstanceID"`
	TargetID         string    `json:"targetID"`
	TargetInstanceID string    `json:"targetInstanceID"`
	Protocol         string    `json:"protocol"`
	TensorEncoding   string    `json:"tensorEncoding"`
	ObservedAt       time.Time `json:"observedAt"`
	RTTMicros        uint64    `json:"rttMicros"`
	PayloadBytes     uint64    `json:"payloadBytes,omitempty"`
	ElapsedMicros    uint64    `json:"elapsedMicros,omitempty"`
}

// ComputeObservation is one worker execution measurement. The key preserves
// backend, checkpoint, operation, and exact layer range so unlike work is not
// silently averaged together.
type ComputeObservation struct {
	WorkerID         string                    `json:"workerID"`
	WorkerInstanceID string                    `json:"workerInstanceID"`
	Backend          string                    `json:"backend"`
	Model            generation.ExecutionModel `json:"model"`
	Operation        string                    `json:"operation"`
	LayerStart       int                       `json:"layerStart"`
	LayerEnd         int                       `json:"layerEnd"`
	InputTokenCount  uint64                    `json:"inputTokenCount"`
	ComputeMicros    uint64                    `json:"computeMicros"`
	ObservedAt       time.Time                 `json:"observedAt"`
}

// ValueDistribution is a deterministic nearest-rank summary. The containing
// field supplies the unit, such as microseconds, bytes, or bytes per second.
type ValueDistribution struct {
	Count int    `json:"count"`
	Min   uint64 `json:"min"`
	P50   uint64 `json:"p50"`
	P95   uint64 `json:"p95"`
	Max   uint64 `json:"max"`
}

// LinkProfile summarizes one fresh directional transport series.
type LinkProfile struct {
	SourceID                string            `json:"sourceID"`
	SourceInstanceID        string            `json:"sourceInstanceID"`
	TargetID                string            `json:"targetID"`
	TargetInstanceID        string            `json:"targetInstanceID"`
	Protocol                string            `json:"protocol"`
	TensorEncoding          string            `json:"tensorEncoding"`
	LatestObservedAt        time.Time         `json:"latestObservedAt"`
	RTTMicros               ValueDistribution `json:"rttMicros"`
	EffectiveBytesPerSecond ValueDistribution `json:"effectiveBytesPerSecond"`
}

// ComputeProfile summarizes one fresh worker/model/range/operation series.
type ComputeProfile struct {
	WorkerID         string                    `json:"workerID"`
	WorkerInstanceID string                    `json:"workerInstanceID"`
	Backend          string                    `json:"backend"`
	Model            generation.ExecutionModel `json:"model"`
	Operation        string                    `json:"operation"`
	LayerStart       int                       `json:"layerStart"`
	LayerEnd         int                       `json:"layerEnd"`
	LatestObservedAt time.Time                 `json:"latestObservedAt"`
	InputTokenCount  ValueDistribution         `json:"inputTokenCount"`
	ComputeMicros    ValueDistribution         `json:"computeMicros"`
}

// ProfileSnapshot contains only evidence fresh at GeneratedAt. Link and
// compute profiles use stable key order so identical state produces identical
// machine-readable evidence.
type ProfileSnapshot struct {
	SchemaVersion       int              `json:"schemaVersion"`
	Revision            uint64           `json:"revision"`
	GeneratedAt         time.Time        `json:"generatedAt"`
	MaxAgeMillis        int64            `json:"maxAgeMillis"`
	MaxSamplesPerSeries int              `json:"maxSamplesPerSeries"`
	MaxSeries           int              `json:"maxSeries"`
	Links               []LinkProfile    `json:"links"`
	Compute             []ComputeProfile `json:"compute"`
}

type linkKey struct {
	sourceID, sourceInstanceID, targetID, targetInstanceID string
	protocol, tensorEncoding                               string
}

type computeKey struct {
	workerID, workerInstanceID, backend, modelID, fingerprint, operation string
	layerCount, layerStart, layerEnd                                     int
}

type linkSample struct {
	observedAt    time.Time
	rttMicros     uint64
	payloadBytes  uint64
	elapsedMicros uint64
}

type computeSample struct {
	observedAt      time.Time
	inputTokenCount uint64
	computeMicros   uint64
}

// ProfileStore is a bounded concurrency-safe rolling evidence store.
type ProfileStore struct {
	mu                  sync.Mutex
	maxAge              time.Duration
	maxSamplesPerSeries int
	maxSeries           int
	revision            uint64
	latestAcceptedAt    time.Time
	links               map[linkKey][]linkSample
	compute             map[computeKey][]computeSample
}

// NewProfileStore applies defaults, validates bounds, and creates an empty
// topology and compute profile.
func NewProfileStore(config ProfileConfig) (*ProfileStore, error) {
	if config.MaxAge == 0 {
		config.MaxAge = DefaultProfileMaxAge
	}
	if config.MaxSamplesPerSeries == 0 {
		config.MaxSamplesPerSeries = DefaultProfileMaxSamplesPerSeries
	}
	if config.MaxSeries == 0 {
		config.MaxSeries = DefaultProfileMaxSeries
	}
	if config.MaxAge < time.Millisecond || config.MaxAge%time.Millisecond != 0 {
		return nil, errors.New("profile maximum age must be a positive whole number of milliseconds")
	}
	if config.MaxSamplesPerSeries < 0 || config.MaxSeries < 0 {
		return nil, errors.New("profile sample and series limits must be positive")
	}
	return &ProfileStore{
		maxAge: config.MaxAge, maxSamplesPerSeries: config.MaxSamplesPerSeries,
		maxSeries: config.MaxSeries,
		links:     make(map[linkKey][]linkSample), compute: make(map[computeKey][]computeSample),
	}, nil
}

// ObserveLink accepts one measurement at the server-controlled reference time.
func (store *ProfileStore) ObserveLink(at time.Time, observation LinkObservation) error {
	normalized, key, err := normalizeLinkObservation(observation)
	if err != nil {
		return err
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	referenceAt, err := store.validateObservationTimeLocked(at, normalized.ObservedAt)
	if err != nil {
		return err
	}
	store.pruneExpiredLocked(referenceAt)
	if _, exists := store.links[key]; !exists && store.seriesCountLocked() >= store.maxSeries {
		return errors.New("profile series limit reached")
	}
	if store.revision == math.MaxUint64 {
		return errors.New("profile revision exhausted")
	}
	store.links[key] = append(store.links[key], linkSample{
		observedAt: normalized.ObservedAt, rttMicros: normalized.RTTMicros,
		payloadBytes: normalized.PayloadBytes, elapsedMicros: normalized.ElapsedMicros,
	})
	sortLinkObservations(store.links[key])
	store.links[key] = retainNewest(store.links[key], store.maxSamplesPerSeries)
	store.revision++
	store.acceptTimeLocked(at)
	return nil
}

// ObserveCompute accepts one execution measurement at the server-controlled
// reference time.
func (store *ProfileStore) ObserveCompute(at time.Time, observation ComputeObservation) error {
	return store.observeComputeBatch(at, []ComputeObservation{observation})
}

func (store *ProfileStore) observeComputeBatch(
	at time.Time,
	observations []ComputeObservation,
) error {
	if len(observations) == 0 {
		return errors.New("compute observation batch is empty")
	}
	normalized := make([]ComputeObservation, len(observations))
	keys := make([]computeKey, len(observations))
	for index, observation := range observations {
		var err error
		normalized[index], keys[index], err = normalizeComputeObservation(observation)
		if err != nil {
			return fmt.Errorf("compute observation %d: %w", index, err)
		}
	}

	store.mu.Lock()
	defer store.mu.Unlock()
	referenceAt := at.UTC()
	for index, observation := range normalized {
		validatedAt, err := store.validateObservationTimeLocked(at, observation.ObservedAt)
		if err != nil {
			return fmt.Errorf("compute observation %d: %w", index, err)
		}
		referenceAt = validatedAt
	}
	store.pruneExpiredLocked(referenceAt)
	newKeys := make(map[computeKey]struct{})
	for _, key := range keys {
		if _, exists := store.compute[key]; !exists {
			newKeys[key] = struct{}{}
		}
	}
	if store.seriesCountLocked()+len(newKeys) > store.maxSeries {
		return errors.New("profile series limit reached")
	}
	if store.revision == math.MaxUint64 {
		return errors.New("profile revision exhausted")
	}
	for index, observation := range normalized {
		key := keys[index]
		store.compute[key] = append(store.compute[key], computeSample{
			observedAt: observation.ObservedAt, inputTokenCount: observation.InputTokenCount,
			computeMicros: observation.ComputeMicros,
		})
		sortComputeObservations(store.compute[key])
		store.compute[key] = retainNewest(store.compute[key], store.maxSamplesPerSeries)
	}
	store.revision++
	store.acceptTimeLocked(at)
	return nil
}

// Snapshot drops expired evidence and returns only observations in the closed
// interval [at-MaxAge, at]. It rejects a reference time before the latest
// accepted update so callers cannot rewind profile freshness.
func (store *ProfileStore) Snapshot(at time.Time) (ProfileSnapshot, error) {
	if at.IsZero() {
		return ProfileSnapshot{}, errors.New("profile snapshot time is required")
	}
	at = at.UTC()
	cutoff := at.Add(-store.maxAge)
	store.mu.Lock()
	defer store.mu.Unlock()
	if at.Before(store.latestAcceptedAt) {
		return ProfileSnapshot{}, errors.New("profile snapshot time precedes the latest accepted update")
	}
	snapshot := ProfileSnapshot{
		SchemaVersion: SchemaVersion, Revision: store.revision, GeneratedAt: at,
		MaxAgeMillis:        store.maxAge.Milliseconds(),
		MaxSamplesPerSeries: store.maxSamplesPerSeries, MaxSeries: store.maxSeries,
		Links:   make([]LinkProfile, 0, len(store.links)),
		Compute: make([]ComputeProfile, 0, len(store.compute)),
	}
	for key, observations := range store.links {
		retained, fresh := currentLinkObservations(observations, cutoff, at)
		if len(retained) == 0 {
			delete(store.links, key)
		} else {
			store.links[key] = retained
		}
		if len(fresh) > 0 {
			snapshot.Links = append(snapshot.Links, summarizeLink(key, fresh))
		}
	}
	for key, observations := range store.compute {
		retained, fresh := currentComputeObservations(observations, cutoff, at)
		if len(retained) == 0 {
			delete(store.compute, key)
		} else {
			store.compute[key] = retained
		}
		if len(fresh) > 0 {
			snapshot.Compute = append(snapshot.Compute, summarizeCompute(key, fresh))
		}
	}
	slices.SortFunc(snapshot.Links, compareLinkProfiles)
	slices.SortFunc(snapshot.Compute, compareComputeProfiles)
	return snapshot, nil
}

func (store *ProfileStore) seriesCountLocked() int {
	return len(store.links) + len(store.compute)
}

func (store *ProfileStore) validateObservationTimeLocked(
	at, observedAt time.Time,
) (time.Time, error) {
	if at.IsZero() {
		return time.Time{}, errors.New("profile acceptance time is required")
	}
	at = at.UTC()
	if observedAt.After(at) {
		return time.Time{}, errors.New("profile observation time is after its acceptance time")
	}
	referenceAt := at
	if store.latestAcceptedAt.After(referenceAt) {
		referenceAt = store.latestAcceptedAt
	}
	if observedAt.Before(referenceAt.Add(-store.maxAge)) {
		return time.Time{}, errors.New("profile observation is already stale at acceptance")
	}
	return referenceAt, nil
}

func (store *ProfileStore) acceptTimeLocked(at time.Time) {
	at = at.UTC()
	if at.After(store.latestAcceptedAt) {
		store.latestAcceptedAt = at
	}
}

func (store *ProfileStore) pruneExpiredLocked(at time.Time) {
	cutoff := at.Add(-store.maxAge)
	for key, observations := range store.links {
		retained, _ := currentLinkObservations(observations, cutoff, at)
		if len(retained) == 0 {
			delete(store.links, key)
		} else {
			store.links[key] = retained
		}
	}
	for key, observations := range store.compute {
		retained, _ := currentComputeObservations(observations, cutoff, at)
		if len(retained) == 0 {
			delete(store.compute, key)
		} else {
			store.compute[key] = retained
		}
	}
}

func normalizeLinkObservation(observation LinkObservation) (LinkObservation, linkKey, error) {
	var err error
	if observation.SourceID, err = normalizeProfileLabel("source ID", observation.SourceID); err != nil {
		return LinkObservation{}, linkKey{}, err
	}
	if observation.TargetID, err = normalizeProfileLabel("target ID", observation.TargetID); err != nil {
		return LinkObservation{}, linkKey{}, err
	}
	if observation.SourceInstanceID, err = normalizeProfileLabel(
		"source instance ID", observation.SourceInstanceID,
	); err != nil {
		return LinkObservation{}, linkKey{}, err
	}
	if observation.TargetInstanceID, err = normalizeProfileLabel(
		"target instance ID", observation.TargetInstanceID,
	); err != nil {
		return LinkObservation{}, linkKey{}, err
	}
	if observation.SourceID == observation.TargetID {
		return LinkObservation{}, linkKey{}, errors.New("profile link endpoints must differ")
	}
	if observation.Protocol, err = normalizeProfileLabel("protocol", observation.Protocol); err != nil {
		return LinkObservation{}, linkKey{}, err
	}
	if observation.TensorEncoding, err = normalizeProfileLabel("tensor encoding", observation.TensorEncoding); err != nil {
		return LinkObservation{}, linkKey{}, err
	}
	if observation.ObservedAt.IsZero() || observation.RTTMicros == 0 {
		return LinkObservation{}, linkKey{}, errors.New("profile link observation requires time and positive RTT")
	}
	if (observation.PayloadBytes == 0) != (observation.ElapsedMicros == 0) {
		return LinkObservation{}, linkKey{}, errors.New("profile transfer bytes and elapsed time must both be zero or positive")
	}
	observation.ObservedAt = observation.ObservedAt.UTC()
	key := linkKey{
		sourceID: observation.SourceID, sourceInstanceID: observation.SourceInstanceID,
		targetID: observation.TargetID, targetInstanceID: observation.TargetInstanceID,
		protocol: observation.Protocol, tensorEncoding: observation.TensorEncoding,
	}
	return observation, key, nil
}

func normalizeComputeObservation(observation ComputeObservation) (ComputeObservation, computeKey, error) {
	var err error
	if observation.WorkerID, err = normalizeProfileLabel("worker ID", observation.WorkerID); err != nil {
		return ComputeObservation{}, computeKey{}, err
	}
	if observation.WorkerInstanceID, err = normalizeProfileLabel(
		"worker instance ID", observation.WorkerInstanceID,
	); err != nil {
		return ComputeObservation{}, computeKey{}, err
	}
	if observation.Backend, err = normalizeProfileLabel("backend", observation.Backend); err != nil {
		return ComputeObservation{}, computeKey{}, err
	}
	if observation.Model.ID, err = normalizeProfileLabel("model ID", observation.Model.ID); err != nil {
		return ComputeObservation{}, computeKey{}, err
	}
	if observation.Model.CheckpointFingerprint, err = normalizeProfileLabel(
		"checkpoint fingerprint", observation.Model.CheckpointFingerprint,
	); err != nil {
		return ComputeObservation{}, computeKey{}, err
	}
	if observation.Operation, err = normalizeProfileLabel("operation", observation.Operation); err != nil {
		return ComputeObservation{}, computeKey{}, err
	}
	if observation.Model.LayerCount <= 0 || observation.LayerStart < 0 ||
		observation.LayerEnd <= observation.LayerStart ||
		observation.LayerEnd > observation.Model.LayerCount {
		return ComputeObservation{}, computeKey{}, errors.New("profile compute observation has an invalid layer range")
	}
	if observation.ObservedAt.IsZero() || observation.InputTokenCount == 0 || observation.ComputeMicros == 0 {
		return ComputeObservation{}, computeKey{}, errors.New(
			"profile compute observation requires time, input tokens, and compute time",
		)
	}
	observation.ObservedAt = observation.ObservedAt.UTC()
	key := computeKey{
		workerID: observation.WorkerID, workerInstanceID: observation.WorkerInstanceID,
		backend: observation.Backend,
		modelID: observation.Model.ID, fingerprint: observation.Model.CheckpointFingerprint,
		layerCount: observation.Model.LayerCount, operation: observation.Operation,
		layerStart: observation.LayerStart, layerEnd: observation.LayerEnd,
	}
	return observation, key, nil
}

func normalizeProfileLabel(name, value string) (string, error) {
	if len(value) > maxProfileLabelBytes {
		return "", fmt.Errorf("profile %s exceeds %d bytes", name, maxProfileLabelBytes)
	}
	value = strings.TrimSpace(value)
	if value == "" {
		return "", fmt.Errorf("profile %s is required", name)
	}
	return strings.Clone(value), nil
}

func sortLinkObservations(observations []linkSample) {
	slices.SortFunc(observations, func(left, right linkSample) int {
		if order := left.observedAt.Compare(right.observedAt); order != 0 {
			return order
		}
		if order := compareUint64(left.rttMicros, right.rttMicros); order != 0 {
			return order
		}
		if order := compareUint64(left.payloadBytes, right.payloadBytes); order != 0 {
			return order
		}
		return compareUint64(left.elapsedMicros, right.elapsedMicros)
	})
}

func sortComputeObservations(observations []computeSample) {
	slices.SortFunc(observations, func(left, right computeSample) int {
		if order := left.observedAt.Compare(right.observedAt); order != 0 {
			return order
		}
		if order := compareUint64(left.inputTokenCount, right.inputTokenCount); order != 0 {
			return order
		}
		return compareUint64(left.computeMicros, right.computeMicros)
	})
}

func retainNewest[T any](values []T, limit int) []T {
	if len(values) <= limit {
		return values
	}
	return append([]T(nil), values[len(values)-limit:]...)
}

func currentLinkObservations(
	observations []linkSample,
	cutoff, at time.Time,
) ([]linkSample, []linkSample) {
	retained := make([]linkSample, 0, len(observations))
	fresh := make([]linkSample, 0, len(observations))
	for _, observation := range observations {
		if observation.observedAt.Before(cutoff) {
			continue
		}
		retained = append(retained, observation)
		if !observation.observedAt.After(at) {
			fresh = append(fresh, observation)
		}
	}
	return retained, fresh
}

func currentComputeObservations(
	observations []computeSample,
	cutoff, at time.Time,
) ([]computeSample, []computeSample) {
	retained := make([]computeSample, 0, len(observations))
	fresh := make([]computeSample, 0, len(observations))
	for _, observation := range observations {
		if observation.observedAt.Before(cutoff) {
			continue
		}
		retained = append(retained, observation)
		if !observation.observedAt.After(at) {
			fresh = append(fresh, observation)
		}
	}
	return retained, fresh
}

func summarizeLink(key linkKey, observations []linkSample) LinkProfile {
	rtt := make([]uint64, len(observations))
	throughput := make([]uint64, 0, len(observations))
	latest := observations[0].observedAt
	for index, observation := range observations {
		rtt[index] = observation.rttMicros
		if observation.payloadBytes > 0 {
			throughput = append(throughput, bytesPerSecond(observation.payloadBytes, observation.elapsedMicros))
		}
		if observation.observedAt.After(latest) {
			latest = observation.observedAt
		}
	}
	return LinkProfile{
		SourceID: key.sourceID, SourceInstanceID: key.sourceInstanceID,
		TargetID: key.targetID, TargetInstanceID: key.targetInstanceID, Protocol: key.protocol,
		TensorEncoding: key.tensorEncoding, LatestObservedAt: latest,
		RTTMicros: summarizeValues(rtt), EffectiveBytesPerSecond: summarizeValues(throughput),
	}
}

func summarizeCompute(key computeKey, observations []computeSample) ComputeProfile {
	tokens := make([]uint64, len(observations))
	compute := make([]uint64, len(observations))
	latest := observations[0].observedAt
	for index, observation := range observations {
		tokens[index] = observation.inputTokenCount
		compute[index] = observation.computeMicros
		if observation.observedAt.After(latest) {
			latest = observation.observedAt
		}
	}
	return ComputeProfile{
		WorkerID: key.workerID, WorkerInstanceID: key.workerInstanceID, Backend: key.backend,
		Model: generation.ExecutionModel{
			ID: key.modelID, CheckpointFingerprint: key.fingerprint, LayerCount: key.layerCount,
		},
		Operation: key.operation, LayerStart: key.layerStart, LayerEnd: key.layerEnd,
		LatestObservedAt: latest, InputTokenCount: summarizeValues(tokens),
		ComputeMicros: summarizeValues(compute),
	}
}

func summarizeValues(values []uint64) ValueDistribution {
	if len(values) == 0 {
		return ValueDistribution{}
	}
	values = append([]uint64(nil), values...)
	slices.Sort(values)
	return ValueDistribution{
		Count: len(values), Min: values[0], P50: nearestRank(values, 50),
		P95: nearestRank(values, 95), Max: values[len(values)-1],
	}
}

func nearestRank(sorted []uint64, percentile int) uint64 {
	index := (percentile*len(sorted) + 99) / 100
	if index < 1 {
		index = 1
	}
	return sorted[index-1]
}

func bytesPerSecond(payloadBytes, elapsedMicros uint64) uint64 {
	whole := payloadBytes / elapsedMicros
	if whole > math.MaxUint64/1_000_000 {
		return math.MaxUint64
	}
	result := whole * 1_000_000
	remainder := payloadBytes % elapsedMicros
	high, low := bits.Mul64(remainder, 1_000_000)
	partial, _ := bits.Div64(high, low, elapsedMicros)
	if result > math.MaxUint64-partial {
		return math.MaxUint64
	}
	return result + partial
}

func compareLinkProfiles(left, right LinkProfile) int {
	for _, pair := range [][2]string{
		{left.SourceID, right.SourceID}, {left.SourceInstanceID, right.SourceInstanceID},
		{left.TargetID, right.TargetID}, {left.TargetInstanceID, right.TargetInstanceID},
		{left.Protocol, right.Protocol}, {left.TensorEncoding, right.TensorEncoding},
	} {
		if order := strings.Compare(pair[0], pair[1]); order != 0 {
			return order
		}
	}
	return 0
}

func compareComputeProfiles(left, right ComputeProfile) int {
	for _, pair := range [][2]string{
		{left.WorkerID, right.WorkerID}, {left.WorkerInstanceID, right.WorkerInstanceID},
		{left.Backend, right.Backend},
		{left.Model.ID, right.Model.ID},
		{left.Model.CheckpointFingerprint, right.Model.CheckpointFingerprint},
		{left.Operation, right.Operation},
	} {
		if order := strings.Compare(pair[0], pair[1]); order != 0 {
			return order
		}
	}
	if left.Model.LayerCount != right.Model.LayerCount {
		return left.Model.LayerCount - right.Model.LayerCount
	}
	if left.LayerStart != right.LayerStart {
		return left.LayerStart - right.LayerStart
	}
	return left.LayerEnd - right.LayerEnd
}

func compareUint64(left, right uint64) int {
	if left < right {
		return -1
	}
	if left > right {
		return 1
	}
	return 0
}

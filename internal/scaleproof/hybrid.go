package scaleproof

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/benchmark"
	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/mesh"
	"github.com/fijimunkii/mlx-swarm/internal/meshstress"
	"github.com/fijimunkii/mlx-swarm/internal/placement"
	"github.com/fijimunkii/mlx-swarm/internal/pooledproof"
	"github.com/fijimunkii/mlx-swarm/internal/registry"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

const (
	hybridInventoryTimeout  = 30 * time.Second
	hybridControlRPCTimeout = 5 * time.Second
)

func runHybrid(
	ctx context.Context,
	config RunConfig,
	pooled RunEvidence,
) (result HybridEvidence, returnErr error) {
	client, err := registry.NewClient(config.ControlURL, config.HTTPClient)
	if err != nil {
		return result, fmt.Errorf("hybrid membership client: %w", err)
	}
	inventory, err := waitForRealInventory(ctx, client, config.Nodes)
	if err != nil {
		return result, err
	}
	specs := hybridSyntheticSpecs(config.Reference, config.SyntheticPeerCount)
	registered := make([]meshstress.WorkerSpec, 0, len(specs))
	defer func() {
		returnErr = errors.Join(returnErr, removeSyntheticPeers(client, registered))
	}()
	if err := registerSyntheticPeers(ctx, client, specs, &registered); err != nil {
		return result, err
	}
	inventoryContext, cancelInventory := context.WithTimeout(ctx, hybridControlRPCTimeout)
	inventory, err = client.Inventory(inventoryContext)
	cancelInventory()
	if err != nil {
		return result, fmt.Errorf("snapshot hybrid inventory: %w", err)
	}
	profiles, err := hybridProfiles(inventory, config.Coordinator, config.Nodes)
	if err != nil {
		return result, err
	}
	request, err := hybridRequest(config, pooled)
	if err != nil {
		return result, err
	}
	scheduler, err := mesh.NewSequenceScheduler(
		fixedInventory{inventory: inventory}, profiles,
		mesh.HTTPResolver{Client: config.HTTPClient},
	)
	if err != nil {
		return result, fmt.Errorf("create hybrid scheduler: %w", err)
	}
	var samples []generation.PlannedStageSample
	sequence, selection, err := scheduler.Prepare(
		ctx, request, nil, generation.PlannedSessionConfig{
			ForwardTimeout: config.ForwardTimeout,
			Observer: func(sample generation.PlannedStageSample) {
				samples = append(samples, sample)
			},
		},
	)
	if err != nil {
		return result, fmt.Errorf("prepare hybrid sequence: %w", err)
	}
	defer func() { returnErr = errors.Join(returnErr, sequence.Close()) }()
	run, err := runScheduledHybrid(
		ctx, sequence, &samples, config.Nodes, config.Reference,
	)
	if err != nil {
		return result, err
	}
	rejections, allSyntheticRejected := hybridSyntheticRejections(selection, specs)
	selectedReal := selectedTargetsAreReal(selection, config.Nodes)
	selectedFive := selectedTargetsAreDistinct(selection, RequiredNodeCount)
	postRunWorkers := hybridPostRunWorkers(run.Teardown, inventory)
	for _, rejection := range rejections {
		if !slices.Contains(rejection.Codes, placement.RejectionIncompatibleCheckpoint) {
			allSyntheticRejected = false
		}
	}
	result = HybridEvidence{
		InventoryRevision: inventory.Revision, InventoryGeneratedAt: inventory.GeneratedAt,
		InventoryWorkerCount: len(inventory.Workers), RealWorkerIDs: nodeIDs(config.Nodes),
		SyntheticWorkerIDs: syntheticSpecIDs(specs), Selection: selection,
		SyntheticRejections: rejections, Run: run,
		SelectedRealWorkersOnly: selectedReal, SelectedFiveDistinctWorkers: selectedFive,
		EverySyntheticWorkerRejected: allSyntheticRejected,
		GeneratedTokensMatchReference: slices.Equal(
			run.Generation.GeneratedTokenIDs, config.Reference.GeneratedTokenIDs,
		),
		SequenceStateReleased: run.SequenceStateReleased &&
			postRunObservationsOrdered(postRunWorkers),
		PostRunWorkers: postRunWorkers,
	}
	return result, nil
}

func registerSyntheticPeers(
	ctx context.Context,
	client *registry.Client,
	specs []meshstress.WorkerSpec,
	cleanup *[]meshstress.WorkerSpec,
) error {
	for _, spec := range specs {
		*cleanup = append(*cleanup, spec)
		registerContext, cancelRegister := context.WithTimeout(ctx, hybridControlRPCTimeout)
		_, err := client.Register(registerContext, spec.Registration(), true)
		cancelRegister()
		if err != nil {
			return fmt.Errorf("register hybrid peer %q: %w", spec.ID, err)
		}
	}
	return nil
}

func waitForRealInventory(
	ctx context.Context,
	client *registry.Client,
	nodes []Node,
) (registry.Inventory, error) {
	waitContext, cancel := context.WithTimeout(ctx, hybridInventoryTimeout)
	defer cancel()
	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()
	lastReason := "membership was not sampled"
	for {
		inventory, err := client.Inventory(waitContext)
		if err == nil {
			if ready, reason := realInventoryReady(inventory, nodes); ready {
				return inventory, nil
			} else {
				lastReason = reason
			}
		} else {
			lastReason = err.Error()
		}
		select {
		case <-waitContext.Done():
			return registry.Inventory{}, fmt.Errorf(
				"wait for fresh clean real-worker membership: %s: %w",
				lastReason, waitContext.Err(),
			)
		case <-ticker.C:
		}
	}
}

func realInventoryReady(inventory registry.Inventory, nodes []Node) (bool, string) {
	workers := make(map[string]registry.Worker, len(inventory.Workers))
	for _, worker := range inventory.Workers {
		workers[worker.ID] = worker
	}
	for _, node := range nodes {
		worker, found := workers[node.ID]
		switch {
		case !found:
			return false, fmt.Sprintf("worker %q is not registered", node.ID)
		case strings.TrimRight(worker.Endpoint, "/") != strings.TrimRight(node.Endpoint, "/"):
			return false, fmt.Sprintf("worker %q endpoint does not match the proof target", node.ID)
		case worker.InstanceID == "":
			return false, fmt.Sprintf("worker %q has no process instance", node.ID)
		case !inventory.WorkerStatusFresh(worker):
			return false, fmt.Sprintf("worker %q status is stale", node.ID)
		case worker.Status.Health != registry.HealthHealthy:
			return false, fmt.Sprintf("worker %q health is %q", node.ID, worker.Status.Health)
		case worker.Status.WorkerObservationSequence == 0:
			return false, fmt.Sprintf("worker %q has no ordered state observation", node.ID)
		case len(worker.Status.RetainedShards) != 0 || worker.Status.OpenSequenceCount != 0 ||
			worker.Status.RetainedBytes != 0:
			return false, fmt.Sprintf("worker %q has not published clean post-run state", node.ID)
		}
	}
	return true, ""
}

func hybridSyntheticSpecs(
	reference pooledproof.Reference,
	count int,
) []meshstress.WorkerSpec {
	specs := make([]meshstress.WorkerSpec, count)
	for index := range specs {
		id := fmt.Sprintf("synthetic-linux-%02d", index)
		specs[index] = meshstress.WorkerSpec{
			ID: id, InstanceID: id + "-instance",
			Endpoint: "http://" + id + ".invalid:8080",
			Backend:  "synthetic", Runtime: "mesh-hybrid-v1", OS: "linux",
			Architecture: "amd64", Device: "synthetic-cpu",
			PhysicalMemoryBytes: 64 << 30, AvailableMemoryBytes: 48 << 30,
			Adapters: []string{reference.ModelType}, Operations: []string{
				"closeSequence", "decode", "detokenize", "loadShard", "modelInfo",
				"openSequence", "prefill", "state", "tokenize",
			},
			CheckpointFingerprints: []string{"incompatible-" + reference.CheckpointFingerprint},
			Admission: registry.AdmissionLimits{
				MaxConcurrentRequests: 4, MaxOpenSequencesPerShard: 2,
				RetainedByteBudget: 1 << 30,
			},
			Transports: []registry.Transport{{
				Protocol:        workerproc.InstanceBoundHTTPProtocol,
				TensorEncodings: []string{workerproc.Base64JSONTensorEncoding},
				MaxRequestBytes: 64 << 20,
			}},
			Health: registry.HealthHealthy, RTTMicros: 100,
			BandwidthBytesPerSecond: 1_000_000_000,
			PrefillMicrosPerLayer:   1, DecodeMicrosPerLayer: 1,
		}
	}
	return specs
}

func hybridProfiles(
	inventory registry.Inventory,
	coordinator CoordinatorEvidence,
	nodes []Node,
) (*placement.ProfileStore, error) {
	profiles, err := placement.NewProfileStore(placement.ProfileConfig{
		MaxAge: time.Minute, MaxSamplesPerSeries: 4, MaxSeries: len(nodes) + 16,
	})
	if err != nil {
		return nil, err
	}
	workers := make(map[string]registry.Worker, len(inventory.Workers))
	for _, worker := range inventory.Workers {
		workers[worker.ID] = worker
	}
	for _, node := range nodes {
		worker, found := workers[node.ID]
		if !found {
			return nil, fmt.Errorf("profile real worker %q is absent from inventory", node.ID)
		}
		rtt := uint64(max(int64(1), node.ProbeMicros))
		if err := profiles.ObserveLink(inventory.GeneratedAt, placement.LinkObservation{
			SourceID: coordinator.ID, SourceInstanceID: coordinator.RunID,
			TargetID: worker.ID, TargetInstanceID: worker.InstanceID,
			Protocol:       workerproc.InstanceBoundHTTPProtocol,
			TensorEncoding: workerproc.Base64JSONTensorEncoding,
			ObservedAt:     inventory.GeneratedAt, RTTMicros: rtt,
		}); err != nil {
			return nil, fmt.Errorf("profile real worker %q: %w", node.ID, err)
		}
	}
	return profiles, nil
}

func hybridRequest(
	config RunConfig,
	pooled RunEvidence,
) (placement.PlanConstructionRequest, error) {
	if len(pooled.Plan.Stages) != RequiredNodeCount ||
		len(pooled.StageLoads) != RequiredNodeCount ||
		len(pooled.Prefill.Stages) != RequiredNodeCount ||
		len(pooled.Decode.Stages) != RequiredNodeCount ||
		len(pooled.All.Stages) != RequiredNodeCount {
		return placement.PlanConstructionRequest{}, errors.New(
			"hybrid range evidence does not contain five aligned stages",
		)
	}
	ranges := make([]placement.RangeCostEstimate, RequiredNodeCount)
	for index, stage := range pooled.Plan.Stages {
		load := pooled.StageLoads[index]
		if load.Index != index || load.Stage != stage ||
			pooled.Prefill.Stages[index].Stage != stage ||
			pooled.Decode.Stages[index].Stage != stage || pooled.All.Stages[index].Stage != stage {
			return placement.PlanConstructionRequest{}, fmt.Errorf(
				"hybrid range evidence stage %d is not aligned", index,
			)
		}
		loadedBytes := positiveIntSum(
			load.Snapshot.LoadedMemory.ActiveBytes, load.Snapshot.LoadedMemory.CacheBytes,
		)
		ranges[index] = placement.RangeCostEstimate{
			LayerStart: stage.LayerStart, LayerEnd: stage.LayerEnd,
			Estimate: placement.StageCostEstimate{
				LoadMemoryBytes:     loadedBytes,
				SequenceMemoryBytes: positiveInt(pooled.All.Stages[index].MaxKVCacheBytes),
				PrefillWireBytes: positiveInt64Sum(
					pooled.Prefill.Stages[index].InputWireBytes.P50Bytes,
					pooled.Prefill.Stages[index].ResponseWireBytes.P50Bytes,
				),
				DecodeWireBytesPerStep: positiveInt64Sum(
					pooled.Decode.Stages[index].InputWireBytes.P50Bytes,
					pooled.Decode.Stages[index].ResponseWireBytes.P50Bytes,
				),
				FallbackPrefillComputeMicros: positiveInt64(
					pooled.Prefill.Stages[index].ComputeMicros.P50Micros,
				),
				FallbackDecodeComputeMicrosPerStep: positiveInt64(
					pooled.Decode.Stages[index].ComputeMicros.P50Micros,
				),
			},
		}
	}
	return placement.PlanConstructionRequest{
		Model: pooled.Plan.Model,
		Scoring: placement.PlanScoringRequest{
			Adapter: config.Reference.ModelType,
			Transport: placement.TransportRequirement{
				Protocol:       workerproc.InstanceBoundHTTPProtocol,
				TensorEncoding: workerproc.Base64JSONTensorEncoding,
			},
			CoordinatorID: config.Coordinator.ID, CoordinatorInstanceID: config.Coordinator.RunID,
			PrefillInputTokens: uint64(len(config.Reference.PromptTokenIDs)),
			DecodeSteps:        uint64(len(config.Reference.GeneratedTokenIDs)),
			FallbackRTTMicros:  1000, FallbackBytesPerSecond: 100_000_000,
		},
		TerminalResponseMode: generation.StageResponseSampledToken,
		MaxStages:            RequiredNodeCount, MaxSearchOperations: 20_000, Ranges: ranges,
	}, nil
}

func runScheduledHybrid(
	ctx context.Context,
	sequence *mesh.ScheduledSequence,
	samples *[]generation.PlannedStageSample,
	nodes []Node,
	reference pooledproof.Reference,
) (RunEvidence, error) {
	generated, err := sequence.Generate(ctx, generation.Request{
		Prompt: reference.Prompt, MaxTokens: len(reference.GeneratedTokenIDs),
		SequenceID: "scale-hybrid-scheduled", IgnoreEOS: false,
	})
	if err != nil {
		return RunEvidence{}, fmt.Errorf("hybrid generation: %w", err)
	}
	all, err := benchmark.SummarizePlanned(*samples)
	if err != nil {
		return RunEvidence{}, err
	}
	prefill, err := benchmark.SummarizePlanned(filterSamples(*samples, "prefill"))
	if err != nil {
		return RunEvidence{}, err
	}
	decode, err := benchmark.SummarizePlanned(filterSamples(*samples, "decode"))
	if err != nil {
		return RunEvidence{}, err
	}
	teardown, released, err := teardownEvidence(ctx, nodes)
	if err != nil {
		return RunEvidence{}, err
	}
	info := sequence.Info()
	return RunEvidence{
		Name: "hybrid-scheduler-12b", StageCount: len(info.ExecutionPlan.Stages),
		CriticalPathWorkers:  len(info.ExecutionPlan.Stages),
		NetworkBoundaryCount: len(info.ExecutionPlan.Stages) - 1,
		Plan:                 info.ExecutionPlan, StageLoads: info.StageLoads, Generation: generated,
		Prefill: prefill, Decode: decode, All: all,
		TokensMatchExpectation: slices.Equal(
			generated.GeneratedTokenIDs, reference.GeneratedTokenIDs,
		),
		SequenceStateReleased: released, Teardown: teardown,
	}, nil
}

func hybridSyntheticRejections(
	selection mesh.SequenceSelection,
	specs []meshstress.WorkerSpec,
) ([]HybridSyntheticRejection, bool) {
	wanted := make(map[string]struct{}, len(specs))
	for _, spec := range specs {
		wanted[spec.ID] = struct{}{}
	}
	codes := make(map[string]map[placement.RejectionCode]struct{}, len(specs))
	eligible := make(map[string]bool, len(specs))
	for _, candidateRange := range selection.Construction.Ranges {
		for _, candidate := range candidateRange.Eligibility.Candidates {
			if _, found := wanted[candidate.WorkerID]; !found {
				continue
			}
			if candidate.Eligible {
				eligible[candidate.WorkerID] = true
				continue
			}
			if codes[candidate.WorkerID] == nil {
				codes[candidate.WorkerID] = make(map[placement.RejectionCode]struct{})
			}
			for _, rejection := range candidate.Rejections {
				codes[candidate.WorkerID][rejection.Code] = struct{}{}
			}
		}
	}
	ids := make([]string, 0, len(codes))
	for id := range codes {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	result := make([]HybridSyntheticRejection, len(ids))
	for index, id := range ids {
		workerCodes := make([]placement.RejectionCode, 0, len(codes[id]))
		for code := range codes[id] {
			workerCodes = append(workerCodes, code)
		}
		slices.Sort(workerCodes)
		result[index] = HybridSyntheticRejection{WorkerID: id, Codes: workerCodes}
	}
	allRejected := len(result) == len(specs)
	for id := range wanted {
		if eligible[id] {
			allRejected = false
		}
	}
	return result, allRejected
}

func selectedTargetsAreReal(selection mesh.SequenceSelection, nodes []Node) bool {
	real := make(map[string]struct{}, len(nodes))
	for _, node := range nodes {
		real[node.ID] = struct{}{}
	}
	if len(selection.Targets) == 0 {
		return false
	}
	for _, target := range selection.Targets {
		if _, found := real[target.WorkerID]; !found {
			return false
		}
	}
	return true
}

func selectedTargetsAreDistinct(selection mesh.SequenceSelection, count int) bool {
	if len(selection.Targets) != count {
		return false
	}
	ids := make(map[string]struct{}, count)
	for _, target := range selection.Targets {
		ids[target.WorkerID] = struct{}{}
	}
	return len(ids) == count
}

func syntheticSpecIDs(specs []meshstress.WorkerSpec) []string {
	ids := make([]string, len(specs))
	for index, spec := range specs {
		ids[index] = spec.ID
	}
	return ids
}

func removeSyntheticPeers(client *registry.Client, specs []meshstress.WorkerSpec) error {
	return removeSyntheticPeersWithTimeout(client, specs, hybridControlRPCTimeout)
}

func removeSyntheticPeersWithTimeout(
	client *registry.Client,
	specs []meshstress.WorkerSpec,
	timeout time.Duration,
) error {
	var cleanupErr error
	for _, spec := range specs {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		err := client.Remove(ctx, spec.ID, spec.InstanceID)
		cancel()
		var remote *registry.RemoteError
		if err != nil && (!errors.As(err, &remote) ||
			(remote.Code != "worker_not_found" && remote.Code != "lease_expired")) {
			cleanupErr = errors.Join(cleanupErr, fmt.Errorf("remove %s: %w", spec.ID, err))
		}
	}
	return cleanupErr
}

func hybridPostRunWorkers(
	teardown []TeardownEvidence,
	inventory registry.Inventory,
) []HybridWorkerObservation {
	sequences := make(map[string]uint64, len(inventory.Workers))
	for _, worker := range inventory.Workers {
		sequences[worker.ID] = worker.Status.WorkerObservationSequence
	}
	result := make([]HybridWorkerObservation, len(teardown))
	for index, worker := range teardown {
		result[index] = HybridWorkerObservation{
			WorkerID:                     worker.NodeID,
			InventoryObservationSequence: sequences[worker.NodeID],
			WorkerObservationSequence:    worker.WorkerObservationSequence,
			LoadedShardCount:             worker.LoadedShardCount,
			OpenSequenceCount:            worker.OpenSequenceCount, KVCacheBytes: worker.KVCacheBytes,
			RetainedBytes: worker.RetainedBytes,
		}
	}
	return result
}

func postRunObservationsOrdered(workers []HybridWorkerObservation) bool {
	if len(workers) != RequiredNodeCount {
		return false
	}
	for _, worker := range workers {
		if worker.InventoryObservationSequence == 0 ||
			worker.WorkerObservationSequence <= worker.InventoryObservationSequence ||
			worker.OpenSequenceCount != 0 ||
			worker.KVCacheBytes != 0 || worker.RetainedBytes != 0 {
			return false
		}
	}
	return true
}

func positiveInt(value int) uint64 {
	if value <= 0 {
		return 1
	}
	return uint64(value)
}

func positiveInt64(value int64) uint64 {
	if value <= 0 {
		return 1
	}
	return uint64(value)
}

func positiveIntSum(values ...int) uint64 {
	total := uint64(0)
	for _, value := range values {
		if value > 0 {
			converted := uint64(value)
			if total > ^uint64(0)-converted {
				return ^uint64(0)
			}
			total += converted
		}
	}
	if total == 0 {
		return 1
	}
	return total
}

func positiveInt64Sum(values ...int64) uint64 {
	total := uint64(0)
	for _, value := range values {
		if value > 0 {
			converted := uint64(value)
			if total > ^uint64(0)-converted {
				return ^uint64(0)
			}
			total += converted
		}
	}
	if total == 0 {
		return 1
	}
	return total
}

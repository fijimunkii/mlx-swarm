package scaleproof

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/benchmark"
	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/pooledproof"
	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

const (
	cleanupShardPrefix = "generate-"
	cleanupTimeout     = 10 * time.Second
)

// Run proves five-host correctness, compares 2/3/4/5-stage performance on the
// same persistent worker pool, and finally runs the checked-in pooled-memory
// model across all five workers.
func Run(ctx context.Context, config RunConfig) (result Result, returnErr error) {
	result = Result{SchemaVersion: SchemaVersion, Coordinator: config.Coordinator}
	if err := normalizeConfig(&config); err != nil {
		return result, err
	}
	result.Nodes = nodeEvidence(config.Nodes)
	result.Checks.FiveDistinctNodes = true

	initial := make(map[string]*workerproc.PersistentWorkerState, len(config.Nodes))
	for _, node := range config.Nodes {
		state, err := workerproc.State(ctx, node.Caller)
		if err != nil {
			return result, fmt.Errorf("%s initial state: %w", node.ID, err)
		}
		initial[node.ID] = state
		if !cleanWorker(state) {
			return result, fmt.Errorf("worker %s must be clean before scale proof", node.ID)
		}
	}
	result.Checks.CleanWorkersAtStart = true
	defer func() {
		if err := cleanupNodes(config.Nodes); err != nil {
			returnErr = errors.Join(returnErr, fmt.Errorf("final cleanup: %w", err))
		}
	}()

	smallModel, err := discoverModel(ctx, config.Nodes[0].Caller, config.SmallModel)
	if err != nil {
		return result, fmt.Errorf("small model: %w", err)
	}
	verificationPlan, err := balancedPlan(
		*smallModel, config.InventoryRevision+"/correctness", config.Nodes,
		generation.StageResponseTensor,
	)
	if err != nil {
		return result, err
	}
	var logits []workerproc.WireTensor
	correctnessRun, err := runPlan(
		ctx, "five-node-logits", verificationPlan, config.Nodes,
		config.Prompt, config.TokenCount, nil, true, config.ForwardTimeout,
		func(_ int, value workerproc.WireTensor) { logits = append(logits, value) },
	)
	if err != nil {
		return result, fmt.Errorf("five-node correctness generation: %w", err)
	}
	if err := cleanupNodes(config.Nodes); err != nil {
		return result, fmt.Errorf("unload correctness plan: %w", err)
	}
	result.Correctness.Run = correctnessRun
	verification, err := generation.VerifyTrace(
		ctx, config.Nodes[0].Caller, *smallModel,
		generation.TraceVerificationConfig{
			RTol: config.RTol, ATol: config.ATol, ForwardTimeout: config.ForwardTimeout,
		},
		correctnessRun.Generation.Prompt,
		correctnessRun.Generation.PromptTokenIDs,
		correctnessRun.Generation.GeneratedTokenIDs,
		logits,
	)
	if err != nil {
		return result, fmt.Errorf("independent small-model oracle: %w", err)
	}
	if err := cleanupNodes(config.Nodes); err != nil {
		return result, fmt.Errorf("unload correctness oracle: %w", err)
	}
	correctnessRun.TokensMatchExpectation = verification.GreedyTokenIDsMatch &&
		verification.ComparedTokens == config.TokenCount
	result.Correctness = CorrectnessEvidence{
		Run: correctnessRun, Verification: verification,
		RTol: config.RTol, ATol: config.ATol,
		AllLogitsMatch: verification.ComparedTokens == config.TokenCount,
	}
	result.Checks.SmallModelLogitsMatch = result.Correctness.AllLogitsMatch
	result.Checks.SmallModelTokensMatch = correctnessRun.TokensMatchExpectation

	expectedSmallTokens := correctnessRun.Generation.GeneratedTokenIDs
	for stageCount := 2; stageCount <= RequiredNodeCount; stageCount++ {
		nodes := config.Nodes[:stageCount]
		plan, err := balancedPlan(
			*smallModel,
			fmt.Sprintf("%s/scaling/%d", config.InventoryRevision, stageCount),
			nodes, generation.StageResponseSampledToken,
		)
		if err != nil {
			return result, err
		}
		run, err := runPlan(
			ctx, fmt.Sprintf("small-model-%d-stage", stageCount), plan, nodes,
			config.Prompt, config.TokenCount, expectedSmallTokens, true,
			config.ForwardTimeout, nil,
		)
		if err != nil {
			return result, fmt.Errorf("%d-stage scaling run: %w", stageCount, err)
		}
		result.Scaling = append(result.Scaling, run)
		if err := cleanupNodes(config.Nodes); err != nil {
			return result, fmt.Errorf("unload %d-stage scaling plan: %w", stageCount, err)
		}
	}
	result.Checks.ScalingRunsTwoThroughFive = len(result.Scaling) == 4
	result.Checks.ScalingTokensMatch = allRunTokensMatch(result.Scaling)

	pooledModel := generation.ExecutionModel{
		ID:                    config.Reference.Model,
		CheckpointFingerprint: config.Reference.CheckpointFingerprint,
		LayerCount:            config.Reference.LayerCount,
	}
	pooledPlan, err := OwnershipAwarePlan(
		pooledModel, config.InventoryRevision+"/pooled-memory", nodeIDs(config.Nodes),
		config.EdgeReserveLayers,
	)
	if err != nil {
		return result, fmt.Errorf("pooled-memory plan: %w", err)
	}
	pooledRun, err := runPlan(
		ctx, "pooled-memory-12b", pooledPlan, config.Nodes,
		config.Reference.Prompt, len(config.Reference.GeneratedTokenIDs),
		config.Reference.GeneratedTokenIDs, false, config.ForwardTimeout, nil,
	)
	if err != nil {
		return result, fmt.Errorf("pooled-memory run: %w", err)
	}
	pooled := PooledEvidence{
		Run: pooledRun, ReferenceModel: config.Reference.Model,
		ReferenceCheckpointBytes: config.Reference.CheckpointBytes,
		ReferenceFullProcessPeak: config.Reference.FullCheckpointMemory.MaxProcessPhysicalBytes,
		CheckpointMatchesReference: pooledRun.Generation.Model == config.Reference.Model &&
			pooledRun.Generation.ModelType == config.Reference.ModelType &&
			pooledRun.Generation.CheckpointFingerprint == config.Reference.CheckpointFingerprint &&
			pooledRun.Generation.CheckpointBytes == config.Reference.CheckpointBytes,
		PromptTokensMatchReference: slices.Equal(
			pooledRun.Generation.PromptTokenIDs, config.Reference.PromptTokenIDs,
		),
		GeneratedTokensMatchReference: slices.Equal(
			pooledRun.Generation.GeneratedTokenIDs, config.Reference.GeneratedTokenIDs,
		),
	}
	pooled.Nodes = pooledNodeEvidence(config, initial, pooledRun)
	pooled.ComplementaryShardsOnly = complementaryLoads(pooledPlan, pooledRun.StageLoads)
	pooled.NoServingFullModelOracle = pooled.ComplementaryShardsOnly
	result.PooledMemory = pooled
	result.Checks.PooledCheckpointMatches = pooled.CheckpointMatchesReference
	result.Checks.PooledPromptTokensMatch = pooled.PromptTokensMatchReference
	result.Checks.PooledGeneratedTokensMatch = pooled.GeneratedTokensMatchReference
	result.Checks.PooledWorkersWithinMemory = pooledNodesWithinMemory(pooled.Nodes)
	result.Checks.NoServingFullModelOracle = pooled.NoServingFullModelOracle

	if err := cleanupNodes(config.Nodes); err != nil {
		return result, fmt.Errorf("unload pooled-memory plan: %w", err)
	}
	result.Checks.SequenceStateReleased = correctnessRun.SequenceStateReleased &&
		allRunSequencesReleased(result.Scaling) && pooledRun.SequenceStateReleased
	result.Checks.WorkersCleanAfterProof = nodesClean(config.Nodes)
	result.Checks.AllPassed = allChecksPassed(result.Checks)
	if !result.Checks.AllPassed {
		return result, errors.New("five-node scale proof did not satisfy every check")
	}
	return result, nil
}

func normalizeConfig(config *RunConfig) error {
	if len(config.Nodes) != RequiredNodeCount {
		return fmt.Errorf("scale proof requires exactly %d nodes", RequiredNodeCount)
	}
	if config.SmallModel == "" {
		config.SmallModel = DefaultSmallModelID
	}
	if config.Prompt == "" {
		config.Prompt = DefaultPrompt
	}
	if config.TokenCount < DefaultTokenCount {
		return fmt.Errorf("token count must be at least %d", DefaultTokenCount)
	}
	if config.ForwardTimeout <= 0 {
		return errors.New("forward timeout must be positive")
	}
	if config.ExpectedMemoryThresholdBytes <= 0 {
		return errors.New("expected memory threshold must be positive")
	}
	if config.RTol < 0 || config.ATol < 0 || math.IsNaN(config.RTol) ||
		math.IsNaN(config.ATol) || math.IsInf(config.RTol, 0) || math.IsInf(config.ATol, 0) {
		return errors.New("numeric tolerances must be finite and non-negative")
	}
	if config.Coordinator.ID == "" || config.Coordinator.RunID == "" ||
		config.Coordinator.OS == "" || config.Coordinator.Architecture == "" {
		return errors.New("coordinator identity is incomplete")
	}
	if err := pooledproof.ValidateReference(config.Reference); err != nil {
		return fmt.Errorf("pooled-memory reference: %w", err)
	}
	if err := validateNodes(config.Nodes); err != nil {
		return err
	}
	return nil
}

func validateNodes(nodes []Node) error {
	ids, endpoints := map[string]struct{}{}, map[string]struct{}{}
	for index, node := range nodes {
		if node.ID == "" || node.Endpoint == "" || node.Caller == nil {
			return fmt.Errorf("node %d is incomplete", index)
		}
		if _, exists := ids[node.ID]; exists {
			return fmt.Errorf("node ID %q is duplicated", node.ID)
		}
		ids[node.ID] = struct{}{}
		endpoint := strings.TrimRight(node.Endpoint, "/")
		if _, exists := endpoints[endpoint]; exists {
			return fmt.Errorf("node endpoint %q is duplicated", endpoint)
		}
		endpoints[endpoint] = struct{}{}
		if node.Capabilities.PhysicalMemoryBytes == 0 {
			return fmt.Errorf("node %q has incomplete capabilities", node.ID)
		}
	}
	return nil
}

func runPlan(
	ctx context.Context,
	name string,
	plan generation.ExecutionPlan,
	nodes []Node,
	prompt string,
	tokens int,
	expectedTokens []int32,
	ignoreEOS bool,
	forwardTimeout time.Duration,
	logitsObserver generation.LogitsObserver,
) (RunEvidence, error) {
	targets := executionTargets(nodes)
	var samples []generation.PlannedStageSample
	session, err := generation.NewPlannedSession(ctx, plan, targets, nil, generation.PlannedSessionConfig{
		ForwardTimeout: forwardTimeout, LogitsObserver: logitsObserver,
		Observer: func(sample generation.PlannedStageSample) { samples = append(samples, sample) },
	})
	if err != nil {
		return RunEvidence{}, err
	}
	generated, err := session.Generate(ctx, generation.Request{
		Prompt: prompt, MaxTokens: tokens,
		SequenceID: fmt.Sprintf("scale-%s", name), IgnoreEOS: ignoreEOS,
	})
	if err != nil {
		return RunEvidence{}, err
	}
	if len(generated.GeneratedTokenIDs) != tokens {
		return RunEvidence{}, fmt.Errorf("generated %d tokens, want %d", len(generated.GeneratedTokenIDs), tokens)
	}
	all, err := benchmark.SummarizePlanned(samples)
	if err != nil {
		return RunEvidence{}, err
	}
	prefill, err := benchmark.SummarizePlanned(filterSamples(samples, "prefill"))
	if err != nil {
		return RunEvidence{}, err
	}
	decode, err := benchmark.SummarizePlanned(filterSamples(samples, "decode"))
	if err != nil {
		return RunEvidence{}, err
	}
	teardown, released, err := teardownEvidence(ctx, nodes)
	if err != nil {
		return RunEvidence{}, err
	}
	match := len(expectedTokens) == 0 || slices.Equal(generated.GeneratedTokenIDs, expectedTokens)
	return RunEvidence{
		Name: name, StageCount: len(plan.Stages),
		CriticalPathWorkers: len(plan.Stages), NetworkBoundaryCount: len(plan.Stages) - 1,
		Plan: plan, StageLoads: session.Info().StageLoads, Generation: generated,
		Prefill: prefill, Decode: decode, All: all,
		TokensMatchExpectation: match, SequenceStateReleased: released, Teardown: teardown,
	}, nil
}

func balancedPlan(
	model workerproc.PersistentModelResult,
	inventoryRevision string,
	nodes []Node,
	mode generation.StageResponseMode,
) (generation.ExecutionPlan, error) {
	balanced, err := generation.BuildBalancedExecutionPlan(generation.ExecutionModel{
		ID: model.ModelID, CheckpointFingerprint: model.CheckpointFingerprint,
		LayerCount: model.LayerCount,
	}, nodeIDs(nodes), mode)
	if err != nil {
		return generation.ExecutionPlan{}, err
	}
	return generation.BuildExecutionPlan(balanced.Model, inventoryRevision, balanced.Stages)
}

func discoverModel(
	ctx context.Context,
	caller workerproc.PersistentCaller,
	modelID string,
) (*workerproc.PersistentModelResult, error) {
	response, err := caller.Call(ctx, workerproc.PersistentRequest{
		Command: "modelInfo", Model: &workerproc.PersistentModelRequest{ModelID: modelID},
	})
	if err != nil {
		return nil, err
	}
	if response.Result == nil || response.Result.Model == nil {
		return nil, errors.New("modelInfo returned no model metadata")
	}
	return response.Result.Model, nil
}

func filterSamples(samples []generation.PlannedStageSample, operation string) []generation.PlannedStageSample {
	result := make([]generation.PlannedStageSample, 0, len(samples))
	for _, sample := range samples {
		if sample.Operation == operation {
			result = append(result, sample)
		}
	}
	return result
}

func teardownEvidence(ctx context.Context, nodes []Node) ([]TeardownEvidence, bool, error) {
	evidence := make([]TeardownEvidence, len(nodes))
	released := true
	for index, node := range nodes {
		state, err := workerproc.State(ctx, node.Caller)
		if err != nil {
			return nil, false, fmt.Errorf("%s teardown state: %w", node.ID, err)
		}
		open := 0
		for _, shard := range state.LoadedShards {
			open += shard.OpenSequenceCount
		}
		evidence[index] = TeardownEvidence{
			NodeID: node.ID, LoadedShardCount: len(state.LoadedShards),
			OpenSequenceCount: open, KVCacheBytes: state.KVCacheBytes,
			RetainedBytes: state.RetainedBytes,
		}
		if open != 0 || state.KVCacheBytes != 0 || state.RetainedBytes != 0 {
			released = false
		}
	}
	return evidence, released, nil
}

func cleanupNodes(nodes []Node) error {
	var cleanupErr error
	for _, node := range nodes {
		ctx, cancel := context.WithTimeout(context.Background(), cleanupTimeout)
		state, err := workerproc.State(ctx, node.Caller)
		cancel()
		if err != nil {
			cleanupErr = errors.Join(cleanupErr, fmt.Errorf("%s state: %w", node.ID, err))
			continue
		}
		for _, shard := range state.LoadedShards {
			if !strings.HasPrefix(shard.ShardID, cleanupShardPrefix) {
				cleanupErr = errors.Join(cleanupErr, fmt.Errorf(
					"%s has unexpected shard %q", node.ID, shard.ShardID,
				))
				continue
			}
			ctx, cancel := context.WithTimeout(context.Background(), cleanupTimeout)
			_, err := node.Caller.Call(ctx, workerproc.PersistentRequest{
				Command: "unloadShard",
				Shard:   &workerproc.PersistentShardRequest{ShardID: shard.ShardID},
			})
			cancel()
			if err != nil {
				cleanupErr = errors.Join(cleanupErr, fmt.Errorf(
					"%s unload %s: %w", node.ID, shard.ShardID, err,
				))
			}
		}
	}
	return cleanupErr
}

func nodeEvidence(nodes []Node) []NodeEvidence {
	result := make([]NodeEvidence, len(nodes))
	for index, node := range nodes {
		result[index] = NodeEvidence{
			ID: node.ID, Endpoint: node.Endpoint, ProbeMicros: node.ProbeMicros,
			Capabilities: node.Capabilities,
		}
	}
	return result
}

func nodeIDs(nodes []Node) []string {
	result := make([]string, len(nodes))
	for index, node := range nodes {
		result[index] = node.ID
	}
	return result
}

func executionTargets(nodes []Node) []generation.ExecutionTarget {
	result := make([]generation.ExecutionTarget, len(nodes))
	for index, node := range nodes {
		result[index] = generation.ExecutionTarget{TargetID: node.ID, Caller: node.Caller}
	}
	return result
}

func cleanWorker(state *workerproc.PersistentWorkerState) bool {
	return state != nil && len(state.LoadedShards) == 0 &&
		state.KVCacheBytes == 0 && state.RetainedBytes == 0
}

func nodesClean(nodes []Node) bool {
	for _, node := range nodes {
		ctx, cancel := context.WithTimeout(context.Background(), cleanupTimeout)
		state, err := workerproc.State(ctx, node.Caller)
		cancel()
		if err != nil || !cleanWorker(state) {
			return false
		}
	}
	return true
}

func allRunTokensMatch(runs []RunEvidence) bool {
	for _, run := range runs {
		if !run.TokensMatchExpectation {
			return false
		}
	}
	return len(runs) > 0
}

func allRunSequencesReleased(runs []RunEvidence) bool {
	for _, run := range runs {
		if !run.SequenceStateReleased {
			return false
		}
	}
	return len(runs) > 0
}

func complementaryPlan(plan generation.ExecutionPlan) bool {
	if len(plan.Stages) != RequiredNodeCount {
		return false
	}
	expectedStart := 0
	for index, stage := range plan.Stages {
		if stage.LayerStart != expectedStart || stage.LayerEnd <= stage.LayerStart ||
			stage.OwnsInput != (index == 0) || stage.OwnsOutput != (index == len(plan.Stages)-1) ||
			stage.LayerStart == 0 && stage.LayerEnd == plan.Model.LayerCount {
			return false
		}
		expectedStart = stage.LayerEnd
	}
	return expectedStart == plan.Model.LayerCount
}

func complementaryLoads(plan generation.ExecutionPlan, loads []generation.StageLoad) bool {
	if !complementaryPlan(plan) || len(loads) != len(plan.Stages) {
		return false
	}
	for index, load := range loads {
		stage := plan.Stages[index]
		snapshot := load.Snapshot
		if load.Index != index || load.Stage != stage || load.Reused ||
			snapshot.ShardID != stage.ShardID || snapshot.ModelID != plan.Model.ID ||
			snapshot.CheckpointFingerprint != plan.Model.CheckpointFingerprint ||
			snapshot.LayerStart != stage.LayerStart || snapshot.LayerEnd != stage.LayerEnd ||
			snapshot.OwnsInput != stage.OwnsInput || snapshot.OwnsOutput != stage.OwnsOutput ||
			(snapshot.LayerStart == 0 && snapshot.LayerEnd == plan.Model.LayerCount) {
			return false
		}
	}
	return true
}

func pooledNodeEvidence(
	config RunConfig,
	initial map[string]*workerproc.PersistentWorkerState,
	run RunEvidence,
) []PooledNodeEvidence {
	result := make([]PooledNodeEvidence, len(config.Nodes))
	for index, node := range config.Nodes {
		load := run.StageLoads[index].Snapshot
		peak := max(
			load.LoadedMemory.ProcessPhysicalBytes,
			load.LoadedMemory.ProcessPeakPhysicalBytes,
			run.All.Stages[index].MemoryHighWater.ProcessPhysicalBytes,
			run.All.Stages[index].MemoryHighWater.ProcessPeakPhysicalBytes,
		)
		state := initial[node.ID]
		configured := node.Capabilities.MLXMemoryLimitBytes == config.ExpectedMemoryThresholdBytes &&
			uint64(config.ExpectedMemoryThresholdBytes) <= node.Capabilities.PhysicalMemoryBytes &&
			state != nil && state.MLXMemoryLimitBytes == config.ExpectedMemoryThresholdBytes &&
			state.PhysicalMemoryBytes == node.Capabilities.PhysicalMemoryBytes &&
			state.MLXCacheLimitBytes == node.Capabilities.MLXCacheLimitBytes
		result[index] = PooledNodeEvidence{
			NodeID: node.ID, Stage: run.Plan.Stages[index], Load: load,
			MaxProcessPhysicalBytes:    peak,
			WithinPhysicalMemory:       peak > 0 && peak <= node.Capabilities.PhysicalMemoryBytes,
			UsesConfiguredMLXThreshold: configured,
		}
	}
	return result
}

func pooledNodesWithinMemory(nodes []PooledNodeEvidence) bool {
	if len(nodes) != RequiredNodeCount {
		return false
	}
	for _, node := range nodes {
		if !node.WithinPhysicalMemory || !node.UsesConfiguredMLXThreshold {
			return false
		}
	}
	return true
}

func allChecksPassed(checks Checks) bool {
	return checks.FiveDistinctNodes && checks.CleanWorkersAtStart &&
		checks.SmallModelLogitsMatch && checks.SmallModelTokensMatch &&
		checks.ScalingRunsTwoThroughFive && checks.ScalingTokensMatch &&
		checks.PooledCheckpointMatches &&
		checks.PooledPromptTokensMatch && checks.PooledGeneratedTokensMatch &&
		checks.PooledWorkersWithinMemory && checks.NoServingFullModelOracle &&
		checks.SequenceStateReleased && checks.WorkersCleanAfterProof
}

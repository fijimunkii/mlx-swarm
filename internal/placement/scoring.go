package placement

import (
	"errors"
	"fmt"
	"math"
	"math/bits"
	"strconv"
	"strings"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/registry"
)

// StageCostEstimate supplies the request-specific resource and fallback cost
// model for one plan stage. Wire byte estimates include request and response
// bytes for one traversal.
type StageCostEstimate struct {
	LoadMemoryBytes                    uint64 `json:"loadMemoryBytes"`
	SequenceMemoryBytes                uint64 `json:"sequenceMemoryBytes"`
	PrefillWireBytes                   uint64 `json:"prefillWireBytes"`
	DecodeWireBytesPerStep             uint64 `json:"decodeWireBytesPerStep"`
	FallbackPrefillComputeMicros       uint64 `json:"fallbackPrefillComputeMicros"`
	FallbackDecodeComputeMicrosPerStep uint64 `json:"fallbackDecodeComputeMicrosPerStep"`
}

// PlanScoringRequest describes one workload and the conservative estimates to
// use wherever fresh exact profile evidence is unavailable.
type PlanScoringRequest struct {
	Adapter                string               `json:"adapter"`
	Transport              TransportRequirement `json:"transport"`
	CoordinatorID          string               `json:"coordinatorID"`
	CoordinatorInstanceID  string               `json:"coordinatorInstanceID"`
	StatusMaxAgeMillis     int64                `json:"statusMaxAgeMillis,omitempty"`
	PrefillInputTokens     uint64               `json:"prefillInputTokens"`
	DecodeSteps            uint64               `json:"decodeSteps"`
	FallbackRTTMicros      uint64               `json:"fallbackRTTMicros"`
	FallbackBytesPerSecond uint64               `json:"fallbackBytesPerSecond"`
	Stages                 []StageCostEstimate  `json:"stages"`
}

// StageComputeScore records the request-scaled compute estimate and any exact
// process/model/range profiles that supplied it.
type StageComputeScore struct {
	PrefillMicros   uint64          `json:"prefillMicros"`
	DecodeMicros    uint64          `json:"decodeMicros"`
	TotalMicros     uint64          `json:"totalMicros"`
	PrefillProfiled bool            `json:"prefillProfiled"`
	DecodeProfiled  bool            `json:"decodeProfiled"`
	PrefillProfile  *ComputeProfile `json:"prefillProfile,omitempty"`
	DecodeProfile   *ComputeProfile `json:"decodeProfile,omitempty"`
}

// StageTransferScore records coordinator-relayed transfer cost and the exact
// process-to-process link profile used when one was available.
type StageTransferScore struct {
	RTTMicros         uint64       `json:"rttMicros"`
	BytesPerSecond    uint64       `json:"bytesPerSecond"`
	PrefillMicros     uint64       `json:"prefillMicros"`
	DecodeMicros      uint64       `json:"decodeMicros"`
	TotalMicros       uint64       `json:"totalMicros"`
	RTTProfiled       bool         `json:"rttProfiled"`
	BandwidthProfiled bool         `json:"bandwidthProfiled"`
	LinkProfile       *LinkProfile `json:"linkProfile,omitempty"`
}

// StagePlanEvaluation preserves hard-constraint evidence for every inventory
// worker and the cost assigned to the plan's selected worker.
type StagePlanEvaluation struct {
	Index             int                       `json:"index"`
	Stage             generation.ExecutionStage `json:"stage"`
	Eligibility       Evaluation                `json:"eligibility"`
	SelectedCandidate Candidate                 `json:"selectedCandidate"`
	Compute           StageComputeScore         `json:"compute"`
	Transfer          StageTransferScore        `json:"transfer"`
}

// PlanScore keeps unlike cost signals separate. EstimatedMicros is the
// sequential compute-plus-transfer estimate; the remaining fields are stable
// risk, pressure, and reuse tie-breakers.
type PlanScore struct {
	StageCount                    int    `json:"stageCount"`
	EstimatedMicros               uint64 `json:"estimatedMicros"`
	ComputeMicros                 uint64 `json:"computeMicros"`
	TransferMicros                uint64 `json:"transferMicros"`
	RecentFailureCount            uint64 `json:"recentFailureCount"`
	RestartCount                  uint64 `json:"restartCount"`
	MemoryPressureBytes           uint64 `json:"memoryPressureBytes"`
	RequiredAdditionalMemoryBytes uint64 `json:"requiredAdditionalMemoryBytes"`
	NewLoadMemoryBytes            uint64 `json:"newLoadMemoryBytes"`
	ReusedStageCount              int    `json:"reusedStageCount"`
	ProfiledComputeOperationCount int    `json:"profiledComputeOperationCount"`
	ProfiledRTTStageCount         int    `json:"profiledRTTStageCount"`
	ProfiledBandwidthStageCount   int    `json:"profiledBandwidthStageCount"`
}

// PlanEvaluation is versioned machine-readable evidence for one complete
// caller-proposed execution plan.
type PlanEvaluation struct {
	SchemaVersion       int                      `json:"schemaVersion"`
	InventoryRevision   uint64                   `json:"inventoryRevision"`
	ProfileRevision     uint64                   `json:"profileRevision"`
	ProfileMaxAgeMillis int64                    `json:"profileMaxAgeMillis"`
	GeneratedAt         time.Time                `json:"generatedAt"`
	Plan                generation.ExecutionPlan `json:"plan"`
	Request             PlanScoringRequest       `json:"request"`
	Eligible            bool                     `json:"eligible"`
	Stages              []StagePlanEvaluation    `json:"stages"`
	Score               PlanScore                `json:"score"`
}

// ScorePlan validates and scores one complete plan. Hard-ineligible plans are
// returned with full stage rejection evidence and zero cost/risk fields rather
// than an error; malformed or incoherent inputs return an error.
func ScorePlan(
	inventory registry.Inventory,
	profile ProfileSnapshot,
	plan generation.ExecutionPlan,
	request PlanScoringRequest,
) (PlanEvaluation, error) {
	request.Adapter = strings.TrimSpace(request.Adapter)
	request.Transport.Protocol = strings.TrimSpace(request.Transport.Protocol)
	request.Transport.TensorEncoding = strings.TrimSpace(request.Transport.TensorEncoding)
	request.CoordinatorID = strings.TrimSpace(request.CoordinatorID)
	request.CoordinatorInstanceID = strings.TrimSpace(request.CoordinatorInstanceID)
	if err := validateScoringInputs(inventory, profile, plan, request); err != nil {
		return PlanEvaluation{}, err
	}

	evaluation := PlanEvaluation{
		SchemaVersion: SchemaVersion, InventoryRevision: inventory.Revision,
		ProfileRevision: profile.Revision, ProfileMaxAgeMillis: profile.MaxAgeMillis,
		GeneratedAt: inventory.GeneratedAt, Plan: plan, Request: request,
		Eligible: true, Stages: make([]StagePlanEvaluation, len(plan.Stages)),
		Score: PlanScore{StageCount: len(plan.Stages)},
	}
	evaluation.Plan.Stages = append([]generation.ExecutionStage(nil), plan.Stages...)
	evaluation.Request.Stages = append([]StageCostEstimate(nil), request.Stages...)
	workers := make(map[string]registry.Worker, len(inventory.Workers))
	for _, worker := range inventory.Workers {
		workers[worker.ID] = worker
	}

	for index, stage := range plan.Stages {
		estimate := request.Stages[index]
		requirement := StageRequirement{
			Model: plan.Model, Adapter: request.Adapter, ShardID: stage.ShardID,
			LayerStart: stage.LayerStart, LayerEnd: stage.LayerEnd,
			OwnsInput: stage.OwnsInput, OwnsOutput: stage.OwnsOutput,
			LoadMemoryBytes:     estimate.LoadMemoryBytes,
			SequenceMemoryBytes: estimate.SequenceMemoryBytes,
			Transport:           request.Transport, StatusMaxAgeMillis: request.StatusMaxAgeMillis,
		}
		eligibility, err := EvaluateCandidates(inventory, requirement)
		if err != nil {
			return PlanEvaluation{}, fmt.Errorf("score plan stage %d eligibility: %w", index, err)
		}
		selected, found := findCandidate(eligibility.Candidates, stage.TargetID)
		if !found {
			return PlanEvaluation{}, fmt.Errorf(
				"score plan stage %d target %q is not in inventory", index, stage.TargetID,
			)
		}
		evaluation.Stages[index] = StagePlanEvaluation{
			Index: index, Stage: stage, Eligibility: eligibility, SelectedCandidate: selected,
		}
		if !selected.Eligible {
			evaluation.Eligible = false
		}
	}
	if !evaluation.Eligible {
		return evaluation, nil
	}

	for index := range evaluation.Stages {
		stageEvaluation := &evaluation.Stages[index]
		stage := stageEvaluation.Stage
		estimate := request.Stages[index]
		worker := workers[stage.TargetID]
		compute, err := scoreStageCompute(profile, plan.Model, worker, stage, request, estimate)
		if err != nil {
			return PlanEvaluation{}, fmt.Errorf("score plan stage %d compute: %w", index, err)
		}
		transfer, err := scoreStageTransfer(profile, worker, request, estimate)
		if err != nil {
			return PlanEvaluation{}, fmt.Errorf("score plan stage %d transfer: %w", index, err)
		}
		stageEvaluation.Compute = compute
		stageEvaluation.Transfer = transfer
		if err := mergeStageScore(
			&evaluation.Score, worker, stageEvaluation.SelectedCandidate,
			estimate, compute, transfer,
		); err != nil {
			return PlanEvaluation{}, fmt.Errorf("score plan stage %d: %w", index, err)
		}
	}
	return evaluation, nil
}

// ComparePlanEvaluations returns a negative value when left is preferred, a
// positive value when right is preferred, and zero when all stable score and
// plan-identity fields are equal.
func ComparePlanEvaluations(left, right PlanEvaluation) int {
	if left.Eligible != right.Eligible {
		if left.Eligible {
			return -1
		}
		return 1
	}
	for _, pair := range [][2]uint64{
		{left.Score.EstimatedMicros, right.Score.EstimatedMicros},
		{left.Score.RecentFailureCount, right.Score.RecentFailureCount},
		{left.Score.MemoryPressureBytes, right.Score.MemoryPressureBytes},
		{left.Score.NewLoadMemoryBytes, right.Score.NewLoadMemoryBytes},
		{left.Score.RequiredAdditionalMemoryBytes, right.Score.RequiredAdditionalMemoryBytes},
		{left.Score.RestartCount, right.Score.RestartCount},
	} {
		if order := compareUint64(pair[0], pair[1]); order != 0 {
			return order
		}
	}
	if left.Score.StageCount != right.Score.StageCount {
		if left.Score.StageCount < right.Score.StageCount {
			return -1
		}
		return 1
	}
	return strings.Compare(left.Plan.Revision, right.Plan.Revision)
}

func validateScoringInputs(
	inventory registry.Inventory,
	profile ProfileSnapshot,
	plan generation.ExecutionPlan,
	request PlanScoringRequest,
) error {
	if err := generation.ValidateExecutionPlan(plan); err != nil {
		return fmt.Errorf("score execution plan: %w", err)
	}
	if plan.InventoryRevision == "" ||
		plan.InventoryRevision != strconv.FormatUint(inventory.Revision, 10) {
		return fmt.Errorf(
			"score plan inventory revision is %q; current inventory revision is %d",
			plan.InventoryRevision, inventory.Revision,
		)
	}
	workerIDs := make(map[string]struct{}, len(inventory.Workers))
	for index, worker := range inventory.Workers {
		if _, duplicate := workerIDs[worker.ID]; duplicate {
			return fmt.Errorf("inventory worker %d has duplicated ID %q", index, worker.ID)
		}
		workerIDs[worker.ID] = struct{}{}
	}
	if profile.SchemaVersion != SchemaVersion {
		return fmt.Errorf("profile schema version is %d, want %d", profile.SchemaVersion, SchemaVersion)
	}
	if !profile.GeneratedAt.Equal(inventory.GeneratedAt) {
		return errors.New("profile and inventory generation times must match")
	}
	if err := validateProfileSnapshot(profile); err != nil {
		return err
	}
	if request.Adapter == "" || request.CoordinatorID == "" || request.CoordinatorInstanceID == "" {
		return errors.New("score adapter and coordinator identity are required")
	}
	if request.Transport.Protocol == "" || request.Transport.TensorEncoding == "" {
		return errors.New("score transport protocol and tensor encoding are required")
	}
	if request.StatusMaxAgeMillis < 0 {
		return errors.New("score status maximum age cannot be negative")
	}
	if request.PrefillInputTokens == 0 || request.DecodeSteps == 0 {
		return errors.New("score prefill tokens and decode steps must be positive")
	}
	if request.FallbackRTTMicros == 0 || request.FallbackBytesPerSecond == 0 {
		return errors.New("score fallback RTT and bandwidth must be positive")
	}
	if len(request.Stages) != len(plan.Stages) {
		return fmt.Errorf("score has %d stage estimates; plan has %d stages", len(request.Stages), len(plan.Stages))
	}
	for index, estimate := range request.Stages {
		if estimate.LoadMemoryBytes == 0 || estimate.SequenceMemoryBytes == 0 ||
			estimate.PrefillWireBytes == 0 || estimate.DecodeWireBytesPerStep == 0 ||
			estimate.FallbackPrefillComputeMicros == 0 ||
			estimate.FallbackDecodeComputeMicrosPerStep == 0 {
			return fmt.Errorf("score stage estimate %d is incomplete", index)
		}
		if _, overflow := addUint64(estimate.LoadMemoryBytes, estimate.SequenceMemoryBytes); overflow {
			return fmt.Errorf("score stage estimate %d memory overflows uint64", index)
		}
		if _, overflow := multiplyUint64(estimate.DecodeWireBytesPerStep, request.DecodeSteps); overflow {
			return fmt.Errorf("score stage estimate %d decode bytes overflow uint64", index)
		}
		if _, overflow := multiplyUint64(
			estimate.FallbackDecodeComputeMicrosPerStep, request.DecodeSteps,
		); overflow {
			return fmt.Errorf("score stage estimate %d fallback compute overflows uint64", index)
		}
	}
	return nil
}

func validateProfileSnapshot(profile ProfileSnapshot) error {
	if profile.GeneratedAt.IsZero() || profile.MaxAgeMillis <= 0 ||
		profile.MaxSamplesPerSeries <= 0 || profile.MaxSeries <= 0 {
		return errors.New("profile snapshot metadata is incomplete")
	}
	const maxDurationMillis = int64(^uint64(0)>>1) / int64(time.Millisecond)
	if profile.MaxAgeMillis > maxDurationMillis ||
		len(profile.Links)+len(profile.Compute) > profile.MaxSeries {
		return errors.New("profile snapshot bounds are invalid")
	}
	cutoff := profile.GeneratedAt.Add(-time.Duration(profile.MaxAgeMillis) * time.Millisecond)
	links := make(map[linkKey]struct{}, len(profile.Links))
	for index, link := range profile.Links {
		key := linkKey{
			sourceID: link.SourceID, sourceInstanceID: link.SourceInstanceID,
			targetID: link.TargetID, targetInstanceID: link.TargetInstanceID,
			protocol: link.Protocol, tensorEncoding: link.TensorEncoding,
		}
		if err := validateSnapshotLabels(
			link.SourceID, link.SourceInstanceID, link.TargetID, link.TargetInstanceID,
			link.Protocol, link.TensorEncoding,
		); err != nil {
			return fmt.Errorf("profile link %d: %w", index, err)
		}
		if link.SourceID == link.TargetID || link.LatestObservedAt.Before(cutoff) ||
			link.LatestObservedAt.After(profile.GeneratedAt) {
			return fmt.Errorf("profile link %d identity or freshness is invalid", index)
		}
		if err := validateDistribution(link.RTTMicros, false, profile.MaxSamplesPerSeries); err != nil {
			return fmt.Errorf("profile link %d RTT: %w", index, err)
		}
		if err := validateDistribution(
			link.EffectiveBytesPerSecond, true, profile.MaxSamplesPerSeries,
		); err != nil {
			return fmt.Errorf("profile link %d bandwidth: %w", index, err)
		}
		if _, duplicate := links[key]; duplicate {
			return fmt.Errorf("profile link %d is duplicated", index)
		}
		links[key] = struct{}{}
	}
	compute := make(map[computeKey]struct{}, len(profile.Compute))
	for index, item := range profile.Compute {
		key := computeKey{
			workerID: item.WorkerID, workerInstanceID: item.WorkerInstanceID,
			backend: item.Backend, modelID: item.Model.ID,
			fingerprint: item.Model.CheckpointFingerprint, operation: item.Operation,
			layerCount: item.Model.LayerCount, layerStart: item.LayerStart, layerEnd: item.LayerEnd,
		}
		if err := validateSnapshotLabels(
			item.WorkerID, item.WorkerInstanceID, item.Backend, item.Model.ID,
			item.Model.CheckpointFingerprint, item.Operation,
		); err != nil {
			return fmt.Errorf("profile compute %d: %w", index, err)
		}
		if item.Model.LayerCount <= 0 || item.LayerStart < 0 || item.LayerEnd <= item.LayerStart ||
			item.LayerEnd > item.Model.LayerCount || item.LatestObservedAt.Before(cutoff) ||
			item.LatestObservedAt.After(profile.GeneratedAt) {
			return fmt.Errorf("profile compute %d range or freshness is invalid", index)
		}
		if err := validateDistribution(item.InputTokenCount, false, profile.MaxSamplesPerSeries); err != nil {
			return fmt.Errorf("profile compute %d input tokens: %w", index, err)
		}
		if err := validateDistribution(item.ComputeMicros, false, profile.MaxSamplesPerSeries); err != nil {
			return fmt.Errorf("profile compute %d time: %w", index, err)
		}
		if _, duplicate := compute[key]; duplicate {
			return fmt.Errorf("profile compute %d is duplicated", index)
		}
		compute[key] = struct{}{}
	}
	return nil
}

func validateSnapshotLabels(values ...string) error {
	for _, value := range values {
		if value == "" || len(value) > maxProfileLabelBytes || strings.TrimSpace(value) != value {
			return errors.New("profile identity label is invalid")
		}
	}
	return nil
}

func validateDistribution(
	distribution ValueDistribution,
	allowEmpty bool,
	maxCount int,
) error {
	if distribution.Count == 0 {
		if allowEmpty && distribution == (ValueDistribution{}) {
			return nil
		}
		return errors.New("distribution is empty")
	}
	if distribution.Count < 0 || distribution.Count > maxCount || distribution.Min == 0 ||
		distribution.Min > distribution.P50 || distribution.P50 > distribution.P95 ||
		distribution.P95 > distribution.Max {
		return errors.New("distribution is inconsistent")
	}
	return nil
}

func scoreStageCompute(
	profile ProfileSnapshot,
	model generation.ExecutionModel,
	worker registry.Worker,
	stage generation.ExecutionStage,
	request PlanScoringRequest,
	estimate StageCostEstimate,
) (StageComputeScore, error) {
	prefillProfile := findComputeProfile(profile.Compute, worker, model, stage, "prefill")
	prefill, prefillProfiled, err := estimateCompute(
		prefillProfile, request.PrefillInputTokens, 1, estimate.FallbackPrefillComputeMicros,
	)
	if err != nil {
		return StageComputeScore{}, err
	}
	decodeProfile := findComputeProfile(profile.Compute, worker, model, stage, "decode")
	decode, decodeProfiled, err := estimateCompute(
		decodeProfile, 1, request.DecodeSteps, estimate.FallbackDecodeComputeMicrosPerStep,
	)
	if err != nil {
		return StageComputeScore{}, err
	}
	total, overflow := addUint64(prefill, decode)
	if overflow {
		return StageComputeScore{}, errors.New("compute estimate overflows uint64")
	}
	return StageComputeScore{
		PrefillMicros: prefill, DecodeMicros: decode, TotalMicros: total,
		PrefillProfiled: prefillProfiled, DecodeProfiled: decodeProfiled,
		PrefillProfile: cloneComputeProfile(prefillProfile),
		DecodeProfile:  cloneComputeProfile(decodeProfile),
	}, nil
}

func estimateCompute(
	profile *ComputeProfile,
	inputTokens, repetitions, fallbackMicros uint64,
) (uint64, bool, error) {
	perCall := fallbackMicros
	profiled := profile != nil
	if profile != nil {
		var err error
		perCall, err = multiplyDivideCeil(
			profile.ComputeMicros.P50, inputTokens, profile.InputTokenCount.P50,
		)
		if err != nil {
			return 0, false, err
		}
	}
	total, overflow := multiplyUint64(perCall, repetitions)
	if overflow {
		return 0, false, errors.New("compute repetition estimate overflows uint64")
	}
	return total, profiled, nil
}

func scoreStageTransfer(
	profile ProfileSnapshot,
	worker registry.Worker,
	request PlanScoringRequest,
	estimate StageCostEstimate,
) (StageTransferScore, error) {
	rtt := request.FallbackRTTMicros
	bandwidth := request.FallbackBytesPerSecond
	link := findLinkProfile(profile.Links, worker, request)
	rttProfiled := link != nil
	bandwidthProfiled := link != nil && link.EffectiveBytesPerSecond.Count > 0
	if link != nil {
		rtt = link.RTTMicros.P50
		if bandwidthProfiled {
			bandwidth = link.EffectiveBytesPerSecond.P50
		}
	}
	prefillPayload, err := multiplyDivideCeil(estimate.PrefillWireBytes, 1_000_000, bandwidth)
	if err != nil {
		return StageTransferScore{}, err
	}
	prefill, overflow := addUint64(rtt, prefillPayload)
	if overflow {
		return StageTransferScore{}, errors.New("prefill transfer estimate overflows uint64")
	}
	decodePayload, err := multiplyDivideCeil(estimate.DecodeWireBytesPerStep, 1_000_000, bandwidth)
	if err != nil {
		return StageTransferScore{}, err
	}
	decodePerStep, overflow := addUint64(rtt, decodePayload)
	if overflow {
		return StageTransferScore{}, errors.New("decode transfer estimate overflows uint64")
	}
	decode, overflow := multiplyUint64(decodePerStep, request.DecodeSteps)
	if overflow {
		return StageTransferScore{}, errors.New("decode transfer estimate overflows uint64")
	}
	total, overflow := addUint64(prefill, decode)
	if overflow {
		return StageTransferScore{}, errors.New("transfer estimate overflows uint64")
	}
	return StageTransferScore{
		RTTMicros: rtt, BytesPerSecond: bandwidth,
		PrefillMicros: prefill, DecodeMicros: decode, TotalMicros: total,
		RTTProfiled: rttProfiled, BandwidthProfiled: bandwidthProfiled,
		LinkProfile: cloneLinkProfile(link),
	}, nil
}

func mergeStageScore(
	score *PlanScore,
	worker registry.Worker,
	candidate Candidate,
	estimate StageCostEstimate,
	compute StageComputeScore,
	transfer StageTransferScore,
) error {
	var err error
	if score.ComputeMicros, err = checkedScoreAdd(score.ComputeMicros, compute.TotalMicros); err != nil {
		return err
	}
	if score.TransferMicros, err = checkedScoreAdd(score.TransferMicros, transfer.TotalMicros); err != nil {
		return err
	}
	if score.EstimatedMicros, err = checkedScoreAdd(score.ComputeMicros, score.TransferMicros); err != nil {
		return err
	}
	if worker.Status.RecentFailureCount < 0 || worker.Status.RestartCount < 0 {
		return errors.New("worker failure counters cannot be negative")
	}
	if score.RecentFailureCount, err = checkedScoreAdd(
		score.RecentFailureCount, uint64(worker.Status.RecentFailureCount),
	); err != nil {
		return err
	}
	if score.RestartCount, err = checkedScoreAdd(
		score.RestartCount, uint64(worker.Status.RestartCount),
	); err != nil {
		return err
	}
	if score.MemoryPressureBytes, err = checkedScoreAdd(
		score.MemoryPressureBytes, worker.Status.MemoryPressureBytes,
	); err != nil {
		return err
	}
	if score.RequiredAdditionalMemoryBytes, err = checkedScoreAdd(
		score.RequiredAdditionalMemoryBytes, candidate.RequiredAdditionalMemoryBytes,
	); err != nil {
		return err
	}
	if candidate.ReusesRetainedShard {
		score.ReusedStageCount++
	} else if score.NewLoadMemoryBytes, err = checkedScoreAdd(
		score.NewLoadMemoryBytes, estimate.LoadMemoryBytes,
	); err != nil {
		return err
	}
	if compute.PrefillProfiled {
		score.ProfiledComputeOperationCount++
	}
	if compute.DecodeProfiled {
		score.ProfiledComputeOperationCount++
	}
	if transfer.RTTProfiled {
		score.ProfiledRTTStageCount++
	}
	if transfer.BandwidthProfiled {
		score.ProfiledBandwidthStageCount++
	}
	return nil
}

func findCandidate(candidates []Candidate, workerID string) (Candidate, bool) {
	for _, candidate := range candidates {
		if candidate.WorkerID == workerID {
			return candidate, true
		}
	}
	return Candidate{}, false
}

func findComputeProfile(
	profiles []ComputeProfile,
	worker registry.Worker,
	model generation.ExecutionModel,
	stage generation.ExecutionStage,
	operation string,
) *ComputeProfile {
	for index := range profiles {
		profile := &profiles[index]
		if profile.WorkerID == worker.ID && profile.WorkerInstanceID == worker.InstanceID &&
			profile.Backend == worker.Capabilities.Backend && profile.Model == model &&
			profile.Operation == operation && profile.LayerStart == stage.LayerStart &&
			profile.LayerEnd == stage.LayerEnd {
			return profile
		}
	}
	return nil
}

func findLinkProfile(
	profiles []LinkProfile,
	worker registry.Worker,
	request PlanScoringRequest,
) *LinkProfile {
	for index := range profiles {
		profile := &profiles[index]
		if profile.SourceID == request.CoordinatorID &&
			profile.SourceInstanceID == request.CoordinatorInstanceID &&
			profile.TargetID == worker.ID && profile.TargetInstanceID == worker.InstanceID &&
			profile.Protocol == request.Transport.Protocol &&
			profile.TensorEncoding == request.Transport.TensorEncoding {
			return profile
		}
	}
	return nil
}

func cloneComputeProfile(profile *ComputeProfile) *ComputeProfile {
	if profile == nil {
		return nil
	}
	cloned := *profile
	return &cloned
}

func cloneLinkProfile(profile *LinkProfile) *LinkProfile {
	if profile == nil {
		return nil
	}
	cloned := *profile
	return &cloned
}

func multiplyUint64(left, right uint64) (uint64, bool) {
	if left != 0 && right > math.MaxUint64/left {
		return 0, true
	}
	return left * right, false
}

func multiplyDivideCeil(left, right, denominator uint64) (uint64, error) {
	if denominator == 0 {
		return 0, errors.New("estimate denominator is zero")
	}
	high, low := bits.Mul64(left, right)
	if high >= denominator {
		return 0, errors.New("estimate overflows uint64")
	}
	quotient, remainder := bits.Div64(high, low, denominator)
	if remainder > 0 {
		if quotient == math.MaxUint64 {
			return 0, errors.New("rounded estimate overflows uint64")
		}
		quotient++
	}
	return quotient, nil
}

func checkedScoreAdd(left, right uint64) (uint64, error) {
	value, overflow := addUint64(left, right)
	if overflow {
		return 0, errors.New("plan score overflows uint64")
	}
	return value, nil
}

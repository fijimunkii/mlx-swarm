package placement

import (
	"cmp"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/fijimunkii/mlx-swarm/internal/generation"
	"github.com/fijimunkii/mlx-swarm/internal/registry"
)

const (
	// DefaultMaxPlanStages matches the current real-mesh validation ceiling.
	DefaultMaxPlanStages = 5
	// DefaultMaxSearchOperations bounds range/worker checks plus DP transitions.
	DefaultMaxSearchOperations = uint64(100_000)
)

// ErrPlanSearchLimit reports that the configured deterministic work budget was
// insufficient. No partially searched plan is returned as selected.
var ErrPlanSearchLimit = errors.New("plan search operation limit reached")

// RangeCostEstimate supplies the complete resource and fallback cost model for
// one allowed contiguous layer range. Omitting a range removes that edge from
// the search graph, allowing model adapters to bound meaningful split points.
type RangeCostEstimate struct {
	LayerStart int               `json:"layerStart"`
	LayerEnd   int               `json:"layerEnd"`
	Estimate   StageCostEstimate `json:"estimate"`
}

// PlanConstructionRequest defines one bounded deterministic search over
// caller-supplied range estimates. Scoring.Stages must be empty because the
// constructor supplies the estimates aligned with each candidate plan.
type PlanConstructionRequest struct {
	Model                generation.ExecutionModel    `json:"model"`
	Scoring              PlanScoringRequest           `json:"scoring"`
	TerminalResponseMode generation.StageResponseMode `json:"terminalResponseMode"`
	MaxStages            int                          `json:"maxStages"`
	MaxSearchOperations  uint64                       `json:"maxSearchOperations"`
	Ranges               []RangeCostEstimate          `json:"ranges"`
}

// RangePlanEvaluation preserves hard-constraint evidence for every worker on
// one allowed layer range, including ranges that are not in the selected plan.
type RangePlanEvaluation struct {
	LayerStart  int               `json:"layerStart"`
	LayerEnd    int               `json:"layerEnd"`
	Estimate    StageCostEstimate `json:"estimate"`
	Eligibility Evaluation        `json:"eligibility"`
}

// PlanConstructionResult records the complete normalized search input,
// bounded-work counters, range/worker evidence, and selected scored plan. A nil
// SelectedPlan means the current mesh has no valid plan under these constraints.
type PlanConstructionResult struct {
	SchemaVersion            int                     `json:"schemaVersion"`
	InventoryRevision        uint64                  `json:"inventoryRevision"`
	ProfileRevision          uint64                  `json:"profileRevision"`
	GeneratedAt              time.Time               `json:"generatedAt"`
	Request                  PlanConstructionRequest `json:"request"`
	CandidateWorkerCount     int                     `json:"candidateWorkerCount"`
	SearchOperationCount     uint64                  `json:"searchOperationCount"`
	RetainedSearchStateCount uint64                  `json:"retainedSearchStateCount"`
	CompletePlanCount        uint64                  `json:"completePlanCount"`
	SearchLimitReached       bool                    `json:"searchLimitReached"`
	Ranges                   []RangePlanEvaluation   `json:"ranges"`
	SelectedPlan             *PlanEvaluation         `json:"selectedPlan,omitempty"`
}

type planRangeSearch struct {
	rangeEstimate RangeCostEstimate
	evaluation    Evaluation
	options       []planStageOption
}

type planStageOption struct {
	worker    registry.Worker
	candidate Candidate
	compute   StageComputeScore
	transfer  StageTransferScore
}

type partialPlan struct {
	end       int
	stages    []generation.ExecutionStage
	estimates []StageCostEstimate
	score     PlanScore
}

// ConstructPlan selects the lowest-scoring valid complete plan reachable
// through the supplied range graph. It uses dynamic programming to retain one
// dominant partial plan per layer boundary and selected-worker set.
func ConstructPlan(
	inventory registry.Inventory,
	profile ProfileSnapshot,
	request PlanConstructionRequest,
) (PlanConstructionResult, error) {
	normalized, err := normalizePlanConstructionRequest(inventory, profile, request)
	if err != nil {
		return PlanConstructionResult{}, err
	}
	result := PlanConstructionResult{
		SchemaVersion: SchemaVersion, InventoryRevision: inventory.Revision,
		ProfileRevision: profile.Revision, GeneratedAt: inventory.GeneratedAt,
		Request: normalized, Ranges: make([]RangePlanEvaluation, len(normalized.Ranges)),
	}
	workerByID := make(map[string]registry.Worker, len(inventory.Workers))
	for _, worker := range inventory.Workers {
		workerByID[worker.ID] = worker
	}

	initialOperations, overflow := multiplyUint64(
		uint64(len(normalized.Ranges)), uint64(len(inventory.Workers)),
	)
	if overflow || initialOperations > normalized.MaxSearchOperations {
		result.SearchLimitReached = true
		return result, fmt.Errorf(
			"%w before range evaluation: need %d operations; limit is %d",
			ErrPlanSearchLimit, initialOperations, normalized.MaxSearchOperations,
		)
	}
	result.SearchOperationCount = initialOperations

	ranges := make([]planRangeSearch, len(normalized.Ranges))
	rangesByStart := make(map[int][]int)
	candidateWorkers := make(map[string]struct{})
	for index, rangeEstimate := range normalized.Ranges {
		search, err := preparePlanRange(
			inventory, profile, normalized, rangeEstimate, workerByID,
		)
		if err != nil {
			return PlanConstructionResult{}, fmt.Errorf(
				"construct plan range [%d,%d): %w",
				rangeEstimate.LayerStart, rangeEstimate.LayerEnd, err,
			)
		}
		ranges[index] = search
		rangesByStart[rangeEstimate.LayerStart] = append(
			rangesByStart[rangeEstimate.LayerStart], index,
		)
		result.Ranges[index] = RangePlanEvaluation{
			LayerStart: rangeEstimate.LayerStart, LayerEnd: rangeEstimate.LayerEnd,
			Estimate: rangeEstimate.Estimate, Eligibility: search.evaluation,
		}
		for _, option := range search.options {
			candidateWorkers[option.worker.ID] = struct{}{}
		}
	}
	result.CandidateWorkerCount = len(candidateWorkers)

	states := map[int]map[string]partialPlan{0: {"": {}}}
	for stageCount := 1; stageCount <= normalized.MaxStages && len(states) > 0; stageCount++ {
		next := make(map[int]map[string]partialPlan)
		for _, state := range orderedPartialPlans(states) {
			for _, rangeIndex := range rangesByStart[state.end] {
				search := ranges[rangeIndex]
				if search.rangeEstimate.LayerEnd < normalized.Model.LayerCount &&
					stageCount == normalized.MaxStages {
					continue
				}
				for _, option := range search.options {
					if partialUsesWorker(state, option.worker.ID) {
						continue
					}
					if result.SearchOperationCount == normalized.MaxSearchOperations {
						result.SearchLimitReached = true
						result.SelectedPlan = nil
						return result, fmt.Errorf(
							"%w after %d operations",
							ErrPlanSearchLimit, result.SearchOperationCount,
						)
					}
					result.SearchOperationCount++
					candidate, err := extendPartialPlan(
						state, normalized, search.rangeEstimate, option,
					)
					if err != nil {
						return PlanConstructionResult{}, err
					}
					key := partialWorkerSetKey(candidate)
					bucket := next[candidate.end]
					if bucket == nil {
						bucket = make(map[string]partialPlan)
						next[candidate.end] = bucket
					}
					if existing, found := bucket[key]; !found ||
						comparePartialPlans(candidate, existing) < 0 {
						bucket[key] = candidate
					}
				}
			}
		}

		for _, bucket := range next {
			result.RetainedSearchStateCount += uint64(len(bucket))
		}
		if complete := next[normalized.Model.LayerCount]; len(complete) > 0 {
			for _, candidate := range orderedPartialPlanBucket(complete) {
				evaluation, err := evaluateCompletePlan(
					inventory, profile, normalized, candidate,
				)
				if err != nil {
					return PlanConstructionResult{}, err
				}
				result.CompletePlanCount++
				if result.SelectedPlan == nil ||
					ComparePlanEvaluations(evaluation, *result.SelectedPlan) < 0 {
					selected := evaluation
					result.SelectedPlan = &selected
				}
			}
			delete(next, normalized.Model.LayerCount)
		}
		states = next
	}
	return result, nil
}

func normalizePlanConstructionRequest(
	inventory registry.Inventory,
	profile ProfileSnapshot,
	request PlanConstructionRequest,
) (PlanConstructionRequest, error) {
	request.Scoring = normalizeScoringRequest(request.Scoring)
	request.Scoring.Stages = append([]StageCostEstimate(nil), request.Scoring.Stages...)
	request.Ranges = append([]RangeCostEstimate(nil), request.Ranges...)
	if request.MaxStages == 0 {
		request.MaxStages = DefaultMaxPlanStages
		if request.Model.LayerCount > 0 && request.MaxStages > request.Model.LayerCount {
			request.MaxStages = request.Model.LayerCount
		}
	}
	if request.MaxSearchOperations == 0 {
		request.MaxSearchOperations = DefaultMaxSearchOperations
	}
	if err := validateScoringSnapshots(inventory, profile); err != nil {
		return PlanConstructionRequest{}, err
	}
	if err := validateScoringContext(request.Scoring); err != nil {
		return PlanConstructionRequest{}, err
	}
	if len(request.Scoring.Stages) != 0 {
		return PlanConstructionRequest{}, errors.New("construct plan scoring stages must be empty")
	}
	if request.Model.ID == "" || request.Model.CheckpointFingerprint == "" ||
		request.Model.LayerCount <= 0 {
		return PlanConstructionRequest{}, errors.New("construct plan model identity is incomplete")
	}
	if request.TerminalResponseMode != generation.StageResponseTensor &&
		request.TerminalResponseMode != generation.StageResponseSampledToken {
		return PlanConstructionRequest{}, fmt.Errorf(
			"construct plan terminal response mode %q is invalid",
			request.TerminalResponseMode,
		)
	}
	if request.MaxStages < 1 || request.MaxStages > request.Model.LayerCount {
		return PlanConstructionRequest{}, fmt.Errorf(
			"construct plan maximum stage count %d is invalid for %d layers",
			request.MaxStages, request.Model.LayerCount,
		)
	}
	if len(request.Ranges) == 0 {
		return PlanConstructionRequest{}, errors.New("construct plan requires range estimates")
	}
	slices.SortFunc(request.Ranges, func(left, right RangeCostEstimate) int {
		if order := cmp.Compare(left.LayerStart, right.LayerStart); order != 0 {
			return order
		}
		return cmp.Compare(left.LayerEnd, right.LayerEnd)
	})
	for index, rangeEstimate := range request.Ranges {
		ownsInput := rangeEstimate.LayerStart == 0
		ownsOutput := rangeEstimate.LayerEnd == request.Model.LayerCount
		if _, err := generation.DeriveExecutionShardID(
			request.Model, rangeEstimate.LayerStart, rangeEstimate.LayerEnd,
			ownsInput, ownsOutput,
		); err != nil {
			return PlanConstructionRequest{}, fmt.Errorf(
				"construct plan range estimate %d: %w", index, err,
			)
		}
		if err := validateStageCostEstimate(
			rangeEstimate.Estimate, request.Scoring.DecodeSteps,
		); err != nil {
			return PlanConstructionRequest{}, fmt.Errorf(
				"construct plan range estimate %d: %w", index, err,
			)
		}
		if index > 0 && request.Ranges[index-1].LayerStart == rangeEstimate.LayerStart &&
			request.Ranges[index-1].LayerEnd == rangeEstimate.LayerEnd {
			return PlanConstructionRequest{}, fmt.Errorf(
				"construct plan range [%d,%d) is duplicated",
				rangeEstimate.LayerStart, rangeEstimate.LayerEnd,
			)
		}
	}
	return request, nil
}

func preparePlanRange(
	inventory registry.Inventory,
	profile ProfileSnapshot,
	request PlanConstructionRequest,
	rangeEstimate RangeCostEstimate,
	workerByID map[string]registry.Worker,
) (planRangeSearch, error) {
	ownsInput := rangeEstimate.LayerStart == 0
	ownsOutput := rangeEstimate.LayerEnd == request.Model.LayerCount
	shardID, err := generation.DeriveExecutionShardID(
		request.Model, rangeEstimate.LayerStart, rangeEstimate.LayerEnd,
		ownsInput, ownsOutput,
	)
	if err != nil {
		return planRangeSearch{}, err
	}
	requirement := StageRequirement{
		Model: request.Model, Adapter: request.Scoring.Adapter, ShardID: shardID,
		LayerStart: rangeEstimate.LayerStart, LayerEnd: rangeEstimate.LayerEnd,
		OwnsInput: ownsInput, OwnsOutput: ownsOutput,
		LoadMemoryBytes:     rangeEstimate.Estimate.LoadMemoryBytes,
		SequenceMemoryBytes: rangeEstimate.Estimate.SequenceMemoryBytes,
		Transport:           request.Scoring.Transport,
		StatusMaxAgeMillis:  request.Scoring.StatusMaxAgeMillis,
	}
	evaluation, err := EvaluateCandidates(inventory, requirement)
	if err != nil {
		return planRangeSearch{}, err
	}
	search := planRangeSearch{
		rangeEstimate: rangeEstimate, evaluation: evaluation,
		options: make([]planStageOption, 0, len(evaluation.Candidates)),
	}
	for _, candidate := range evaluation.Candidates {
		if !candidate.Eligible {
			continue
		}
		worker, found := workerByID[candidate.WorkerID]
		if !found {
			return planRangeSearch{}, fmt.Errorf(
				"eligible worker %q is not in inventory", candidate.WorkerID,
			)
		}
		stage := generation.ExecutionStage{
			TargetID: worker.ID, LayerStart: rangeEstimate.LayerStart,
			LayerEnd: rangeEstimate.LayerEnd, OwnsInput: ownsInput, OwnsOutput: ownsOutput,
		}
		compute, err := scoreStageCompute(
			profile, request.Model, worker, stage, request.Scoring, rangeEstimate.Estimate,
		)
		if err != nil {
			return planRangeSearch{}, err
		}
		transfer, err := scoreStageTransfer(
			profile, worker, request.Scoring, rangeEstimate.Estimate,
		)
		if err != nil {
			return planRangeSearch{}, err
		}
		search.options = append(search.options, planStageOption{
			worker: worker, candidate: candidate, compute: compute, transfer: transfer,
		})
	}
	return search, nil
}

func extendPartialPlan(
	state partialPlan,
	request PlanConstructionRequest,
	rangeEstimate RangeCostEstimate,
	option planStageOption,
) (partialPlan, error) {
	stageIndex := len(state.stages)
	stage := generation.ExecutionStage{
		Name: fmt.Sprintf("stage-%d", stageIndex), TargetID: option.worker.ID,
		LayerStart: rangeEstimate.LayerStart, LayerEnd: rangeEstimate.LayerEnd,
		OwnsInput:    rangeEstimate.LayerStart == 0,
		OwnsOutput:   rangeEstimate.LayerEnd == request.Model.LayerCount,
		ResponseMode: generation.StageResponseTensor,
	}
	if stage.OwnsOutput {
		stage.ResponseMode = request.TerminalResponseMode
	}
	candidate := partialPlan{
		end: rangeEstimate.LayerEnd,
		stages: append(append(
			make([]generation.ExecutionStage, 0, len(state.stages)+1), state.stages...), stage,
		),
		estimates: append(append(
			make([]StageCostEstimate, 0, len(state.estimates)+1), state.estimates...),
			rangeEstimate.Estimate,
		),
		score: state.score,
	}
	if err := mergeStageScore(
		&candidate.score, option.worker, option.candidate, rangeEstimate.Estimate,
		option.compute, option.transfer,
	); err != nil {
		return partialPlan{}, err
	}
	candidate.score.StageCount = len(candidate.stages)
	return candidate, nil
}

func evaluateCompletePlan(
	inventory registry.Inventory,
	profile ProfileSnapshot,
	request PlanConstructionRequest,
	candidate partialPlan,
) (PlanEvaluation, error) {
	plan, err := generation.BuildExecutionPlan(
		request.Model, strconv.FormatUint(inventory.Revision, 10), candidate.stages,
	)
	if err != nil {
		return PlanEvaluation{}, fmt.Errorf("construct execution plan: %w", err)
	}
	scoring := request.Scoring
	scoring.Stages = append([]StageCostEstimate(nil), candidate.estimates...)
	evaluation, err := ScorePlan(inventory, profile, plan, scoring)
	if err != nil {
		return PlanEvaluation{}, err
	}
	if !evaluation.Eligible || evaluation.Score != candidate.score {
		return PlanEvaluation{}, errors.New("complete plan score disagrees with range search")
	}
	return evaluation, nil
}

func partialUsesWorker(state partialPlan, workerID string) bool {
	for _, stage := range state.stages {
		if stage.TargetID == workerID {
			return true
		}
	}
	return false
}

func partialWorkerSetKey(state partialPlan) string {
	workers := make([]string, len(state.stages))
	for index, stage := range state.stages {
		workers[index] = stage.TargetID
	}
	slices.Sort(workers)
	return strings.Join(workers, "\x00")
}

func comparePartialPlans(left, right partialPlan) int {
	if order := comparePlanScores(left.score, right.score); order != 0 {
		return order
	}
	return comparePlanIdentity(
		generation.ExecutionPlan{Stages: left.stages},
		generation.ExecutionPlan{Stages: right.stages},
	)
}

func orderedPartialPlans(states map[int]map[string]partialPlan) []partialPlan {
	result := make([]partialPlan, 0)
	for _, bucket := range states {
		for _, state := range bucket {
			result = append(result, state)
		}
	}
	slices.SortFunc(result, func(left, right partialPlan) int {
		if order := cmp.Compare(left.end, right.end); order != 0 {
			return order
		}
		return comparePartialPlans(left, right)
	})
	return result
}

func orderedPartialPlanBucket(bucket map[string]partialPlan) []partialPlan {
	result := make([]partialPlan, 0, len(bucket))
	for _, state := range bucket {
		result = append(result, state)
	}
	slices.SortFunc(result, comparePartialPlans)
	return result
}

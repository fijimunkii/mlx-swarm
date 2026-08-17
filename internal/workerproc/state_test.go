package workerproc

import (
	"context"
	"testing"
)

type stateObservationCaller struct {
	sequence uint64
}

func (caller stateObservationCaller) Call(
	context.Context,
	PersistentRequest,
) (PersistentResponse, error) {
	return PersistentResponse{
		OK: true, WorkerObservationSequence: caller.sequence,
		Result: &PersistentWorkerResult{State: &PersistentWorkerState{}},
	}, nil
}

func TestObserveStatePreservesWorkerObservationSequence(t *testing.T) {
	const sequence = uint64(42)
	observation, err := ObserveState(
		context.Background(), stateObservationCaller{sequence: sequence},
	)
	if err != nil {
		t.Fatal(err)
	}
	if observation.ObservationSequence != sequence {
		t.Fatalf("observation sequence = %d, want %d", observation.ObservationSequence, sequence)
	}
}

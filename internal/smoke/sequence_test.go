package smoke

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

type sequenceCaller struct {
	open               map[string]bool
	owners             map[string]string
	failOpen           string
	applyBeforeFailure bool
	definitiveFailure  bool
	closeCalls         []string
}

func (caller *sequenceCaller) Call(
	_ context.Context,
	request workerproc.PersistentRequest,
) (workerproc.PersistentResponse, error) {
	key := request.Sequence.ShardID + "/" + request.Sequence.SequenceID
	switch request.Command {
	case "openSequence":
		if key == caller.failOpen {
			if caller.applyBeforeFailure {
				caller.open[key] = true
				caller.owners[key] = request.Sequence.OwnerID
			}
			if caller.definitiveFailure {
				return workerproc.PersistentResponse{}, &workerproc.WorkerResponseError{
					RequestID: "test", Message: "sequence is already open",
				}
			}
			return workerproc.PersistentResponse{}, errors.New("injected open failure")
		}
		caller.open[key] = true
		caller.owners[key] = request.Sequence.OwnerID
	case "closeSequence":
		delete(caller.open, key)
		delete(caller.owners, key)
		caller.closeCalls = append(caller.closeCalls, key)
	default:
		return workerproc.PersistentResponse{}, fmt.Errorf("unexpected command %q", request.Command)
	}
	return workerproc.PersistentResponse{OK: true}, nil
}

func TestOpenSequencesRollsBackPartialMatrix(t *testing.T) {
	first := &sequenceCaller{open: make(map[string]bool), owners: make(map[string]string)}
	second := &sequenceCaller{open: make(map[string]bool), owners: make(map[string]string), failOpen: "consumer/b"}
	targets := []SequenceTarget{
		{Name: "producer", Caller: first, ShardID: "producer"},
		{Name: "consumer", Caller: second, ShardID: "consumer"},
	}

	if _, err := OpenSequences(context.Background(), targets, "a", "b"); err == nil {
		t.Fatal("expected open failure")
	}
	if len(first.open) != 0 || len(second.open) != 0 {
		t.Fatalf("rollback left sequences open: first=%v second=%v", first.open, second.open)
	}
}

func TestOpenSequencesTracksAmbiguousOpenForRollback(t *testing.T) {
	caller := &sequenceCaller{
		open: make(map[string]bool), owners: make(map[string]string),
		failOpen: "producer/a", applyBeforeFailure: true,
	}
	if _, err := OpenSequences(context.Background(), []SequenceTarget{
		{Caller: caller, ShardID: "producer"},
	}, "a"); err == nil {
		t.Fatal("expected ambiguous open failure")
	}
	if len(caller.open) != 0 {
		t.Fatalf("rollback left ambiguous open behind: %v", caller.open)
	}
}

func TestOpenSequencesDoesNotCloseDefinitivelyRejectedOpen(t *testing.T) {
	first := &sequenceCaller{open: make(map[string]bool), owners: make(map[string]string)}
	second := &sequenceCaller{
		open:     map[string]bool{"consumer/a": true},
		owners:   map[string]string{"consumer/a": "another-owner"},
		failOpen: "consumer/a", definitiveFailure: true,
	}
	if _, err := OpenSequences(context.Background(), []SequenceTarget{
		{Caller: first, ShardID: "producer"},
		{Caller: second, ShardID: "consumer"},
	}, "a"); err == nil {
		t.Fatal("expected definitive open rejection")
	}
	if !second.open["consumer/a"] || second.owners["consumer/a"] != "another-owner" {
		t.Fatalf("rollback changed another owner's sequence: %v", second.open)
	}
}

func TestOpenSequencesAssignsOnePrivateOwner(t *testing.T) {
	first := &sequenceCaller{open: make(map[string]bool), owners: make(map[string]string)}
	second := &sequenceCaller{open: make(map[string]bool), owners: make(map[string]string)}
	set, err := OpenSequences(context.Background(), []SequenceTarget{
		{Caller: first, ShardID: "producer"},
		{Caller: second, ShardID: "consumer"},
	}, "a")
	if err != nil {
		t.Fatal(err)
	}
	defer set.Cleanup()
	firstOwner := first.owners["producer/a"]
	secondOwner := second.owners["consumer/a"]
	if firstOwner == "" || firstOwner != secondOwner {
		t.Fatalf("owners = %q and %q", firstOwner, secondOwner)
	}
}

func TestOpenSequencesValidatesConfigurationBeforeCallingWorkers(t *testing.T) {
	caller := &sequenceCaller{open: make(map[string]bool), owners: make(map[string]string)}
	tests := []struct {
		name    string
		targets []SequenceTarget
		ids     []string
	}{
		{name: "no targets", ids: []string{"a"}},
		{name: "no caller", targets: []SequenceTarget{{ShardID: "producer"}}, ids: []string{"a"}},
		{name: "no shard", targets: []SequenceTarget{{Caller: caller}}, ids: []string{"a"}},
		{name: "no IDs", targets: []SequenceTarget{{Caller: caller, ShardID: "producer"}}},
		{name: "duplicate ID", targets: []SequenceTarget{{Caller: caller, ShardID: "producer"}}, ids: []string{"a", "a"}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := OpenSequences(context.Background(), test.targets, test.ids...); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
	if len(caller.open) != 0 {
		t.Fatalf("validation called worker: %v", caller.open)
	}
}

func TestSequenceSetClosesOneSequenceAcrossTargets(t *testing.T) {
	first := &sequenceCaller{open: make(map[string]bool), owners: make(map[string]string)}
	second := &sequenceCaller{open: make(map[string]bool), owners: make(map[string]string)}
	set, err := OpenSequences(context.Background(), []SequenceTarget{
		{Caller: first, ShardID: "producer"},
		{Caller: second, ShardID: "consumer"},
	}, "a", "b")
	if err != nil {
		t.Fatal(err)
	}
	if err := set.CloseSequence(context.Background(), "b"); err != nil {
		t.Fatal(err)
	}
	if first.open["producer/b"] || second.open["consumer/b"] ||
		!first.open["producer/a"] || !second.open["consumer/a"] {
		t.Fatalf("unexpected open sequences: first=%v second=%v", first.open, second.open)
	}
	if err := set.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(first.open) != 0 || len(second.open) != 0 {
		t.Fatalf("close left sequences open: first=%v second=%v", first.open, second.open)
	}
}

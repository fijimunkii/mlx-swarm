package failureharness

import (
	"context"
	"os"
	"testing"
	"time"
)

func TestMain(m *testing.M) {
	if len(os.Args) > 1 && os.Args[1] == "fault-worker" {
		if err := ServeFaultWorker(os.Args[2:], os.Stdin, os.Stdout); err != nil {
			os.Exit(2)
		}
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func TestRunCharacterizesBoundedFailuresAndRecovery(t *testing.T) {
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	summary, err := Run(ctx, executable, 150*time.Millisecond, 2*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if len(summary.Scenarios) != 5 || !summary.AllTerminatedBounded ||
		!summary.AllCleanupReleased || !summary.AllRecoveryReady {
		t.Fatalf("unexpected summary: %+v", summary)
	}
	if summary.FailedTokens != 4 || summary.AcceptedTokens != 8 || summary.FailedTokenRate != 1.0/3.0 {
		t.Fatalf("unexpected token accounting: %+v", summary)
	}
	for _, scenario := range summary.Scenarios {
		if scenario.ExpectedFailure {
			if !scenario.ObservedFailure || scenario.Failure == nil ||
				scenario.Failure.SequenceID == "" || scenario.Failure.ShardID == "" ||
				scenario.Failure.Phase != "consumer_decode" ||
				scenario.Failure.LastAcceptedTokenID == nil {
				t.Fatalf("scenario lacks structured failure metadata: %+v", scenario)
			}
		}
	}
}

package smoke

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

// InferenceRequest constructs one persistent forward request.
func InferenceRequest(
	command string,
	shardID string,
	sequenceID string,
	position uint64,
	inputKind string,
	input workerproc.WireTensor,
) workerproc.PersistentRequest {
	return workerproc.PersistentRequest{
		Command: command,
		Forward: &workerproc.PersistentForwardRequest{
			ShardID: shardID, SequenceID: sequenceID, Position: position,
			InputKind: inputKind, Input: input,
		},
	}
}

// Infer executes a cache-aware inference command and validates its metadata.
func Infer(
	ctx context.Context,
	caller workerproc.PersistentCaller,
	command string,
	shardID string,
	sequenceID string,
	position uint64,
	inputKind string,
	input workerproc.WireTensor,
) (*workerproc.PersistentForwardResult, error) {
	response, err := Call(ctx, caller, InferenceRequest(
		command, shardID, sequenceID, position, inputKind, input,
	))
	if err != nil {
		return nil, err
	}
	if response.Result == nil || response.Result.Forward == nil {
		return nil, fmt.Errorf("%s returned no tensor", command)
	}
	result := response.Result.Forward
	if result.Operation != command || result.Position != position || result.KVCacheBytes == 0 {
		return nil, fmt.Errorf(
			"%s returned inconsistent metadata: operation=%s position=%d kv=%d",
			command, result.Operation, result.Position, result.KVCacheBytes,
		)
	}
	return result, nil
}

// ExpectWorkerError reports whether a request failed with the expected worker
// response text.
func ExpectWorkerError(
	ctx context.Context,
	caller workerproc.PersistentCaller,
	request workerproc.PersistentRequest,
	want string,
) bool {
	_, err := caller.Call(ctx, request)
	var responseError *workerproc.WorkerResponseError
	return errors.As(err, &responseError) && strings.Contains(responseError.Message, want)
}

package smoke

import (
	"encoding/binary"

	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

// TokenTensor encodes one batch of int32 token IDs for a worker request.
func TokenTensor(tokens []int32) workerproc.WireTensor {
	data := make([]byte, len(tokens)*4)
	for index, token := range tokens {
		binary.LittleEndian.PutUint32(data[index*4:], uint32(token))
	}
	return workerproc.WireTensor{
		Shape: []int{1, len(tokens)}, DType: "int32", Data: data,
	}
}

package generation

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

func greedyToken(tensor workerproc.WireTensor) (int32, error) {
	if len(tensor.Shape) == 0 {
		return 0, errors.New("logit tensor has no dimensions")
	}
	vocabulary := tensor.Shape[len(tensor.Shape)-1]
	values, err := finalValues(tensor, vocabulary)
	if err != nil {
		return 0, err
	}
	bestIndex := 0
	bestValue := values[0]
	if math.IsNaN(bestValue) {
		return 0, errors.New("NaN at vocabulary index 0")
	}
	for index := 1; index < len(values); index++ {
		if math.IsNaN(values[index]) {
			return 0, fmt.Errorf("NaN at vocabulary index %d", index)
		}
		if values[index] > bestValue {
			bestIndex = index
			bestValue = values[index]
		}
	}
	if bestIndex > math.MaxInt32 {
		return 0, fmt.Errorf("greedy token index %d exceeds int32", bestIndex)
	}
	return int32(bestIndex), nil
}

func compareFinalLogits(
	got workerproc.WireTensor,
	want workerproc.WireTensor,
	rtol float64,
	atol float64,
) (float64, float64, error) {
	if len(got.Shape) == 0 || len(want.Shape) == 0 {
		return 0, 0, errors.New("logit tensor has no dimensions")
	}
	if len(got.Shape) != len(want.Shape) {
		return 0, 0, fmt.Errorf("logit rank mismatch: got %v want %v", got.Shape, want.Shape)
	}
	for index := range got.Shape {
		if got.Shape[index] != want.Shape[index] {
			return 0, 0, fmt.Errorf("logit shape mismatch: got %v want %v", got.Shape, want.Shape)
		}
	}
	vocabulary := got.Shape[len(got.Shape)-1]
	gotValues, err := finalValues(got, vocabulary)
	if err != nil {
		return 0, 0, fmt.Errorf("decode distributed logits: %w", err)
	}
	wantValues, err := finalValues(want, vocabulary)
	if err != nil {
		return 0, 0, fmt.Errorf("decode reference logits: %w", err)
	}
	maxAbsolute := 0.0
	maxRelative := 0.0
	for index := range gotValues {
		actual := gotValues[index]
		expected := wantValues[index]
		if math.IsNaN(actual) || math.IsNaN(expected) {
			return 0, 0, fmt.Errorf("NaN at vocabulary index %d", index)
		}
		if math.IsInf(actual, 0) || math.IsInf(expected, 0) {
			if actual == expected {
				continue
			}
			return 0, 0, fmt.Errorf("non-matching infinity at vocabulary index %d", index)
		}
		absolute := math.Abs(actual - expected)
		// Logits are at most float32 precision, so its smallest representable
		// nonzero magnitude keeps the zero-reference diagnostic finite without
		// changing relative errors for any representable nonzero reference.
		relative := absolute / math.Max(math.Abs(expected), math.SmallestNonzeroFloat32)
		maxAbsolute = math.Max(maxAbsolute, absolute)
		maxRelative = math.Max(maxRelative, relative)
		if absolute > atol+rtol*math.Abs(expected) {
			return maxAbsolute, maxRelative, fmt.Errorf(
				"vocabulary index %d differs: got=%g want=%g abs=%g tolerance=%g",
				index,
				actual,
				expected,
				absolute,
				atol+rtol*math.Abs(expected),
			)
		}
	}
	return maxAbsolute, maxRelative, nil
}

func finalValues(tensor workerproc.WireTensor, count int) ([]float64, error) {
	var elementBytes int
	switch tensor.DType {
	case "bfloat16", "float16":
		elementBytes = 2
	case "float32":
		elementBytes = 4
	default:
		return nil, fmt.Errorf("unsupported logit dtype %q", tensor.DType)
	}
	required := count * elementBytes
	if count <= 0 || required/elementBytes != count || len(tensor.Data) < required || len(tensor.Data)%elementBytes != 0 {
		return nil, fmt.Errorf("invalid %s logit payload size %d", tensor.DType, len(tensor.Data))
	}
	data := tensor.Data[len(tensor.Data)-required:]
	values := make([]float64, count)
	for index := range values {
		offset := index * elementBytes
		switch tensor.DType {
		case "bfloat16":
			bits := uint32(binary.LittleEndian.Uint16(data[offset:])) << 16
			values[index] = float64(math.Float32frombits(bits))
		case "float16":
			values[index] = float64(float16(binary.LittleEndian.Uint16(data[offset:])))
		case "float32":
			values[index] = float64(math.Float32frombits(binary.LittleEndian.Uint32(data[offset:])))
		}
	}
	return values, nil
}

func float16(bits uint16) float32 {
	sign := 1.0
	if bits&0x8000 != 0 {
		sign = -1
	}
	exponent := int(bits>>10) & 0x1f
	fraction := int(bits & 0x03ff)
	switch exponent {
	case 0:
		if fraction == 0 {
			return float32(math.Copysign(0, sign))
		}
		return float32(sign * math.Ldexp(float64(fraction), -24))
	case 0x1f:
		if fraction == 0 {
			return float32(math.Inf(int(sign)))
		}
		return float32(math.NaN())
	}
	value := math.Ldexp(1+float64(fraction)/1024, exponent-15)
	return float32(sign * value)
}

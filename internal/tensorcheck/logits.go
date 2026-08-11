// Package tensorcheck provides reusable correctness checks for worker tensors.
package tensorcheck

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

// Difference describes the largest element-wise difference in a comparison.
type Difference struct {
	Absolute float64
	Relative float64
}

// Metrics accumulates final-logit comparison results across a smoke proof.
type Metrics struct {
	Comparisons           int
	MaxAbsoluteDifference float64
	MaxRelativeDifference float64
}

// CompareFinalLogits compares the final sequence position in two equally
// shaped logit tensors using absolute and relative tolerances.
func CompareFinalLogits(
	got workerproc.WireTensor,
	want workerproc.WireTensor,
	rtol float64,
	atol float64,
) (Difference, error) {
	var difference Difference
	if rtol < 0 || atol < 0 || math.IsNaN(rtol) || math.IsNaN(atol) ||
		math.IsInf(rtol, 0) || math.IsInf(atol, 0) {
		return difference, errors.New("logit tolerances must be finite and non-negative")
	}
	if len(got.Shape) == 0 || len(want.Shape) == 0 {
		return difference, errors.New("logit tensor has no dimensions")
	}
	if len(got.Shape) != len(want.Shape) {
		return difference, fmt.Errorf("logit rank mismatch: got %v want %v", got.Shape, want.Shape)
	}
	for index := range got.Shape {
		if got.Shape[index] != want.Shape[index] {
			return difference, fmt.Errorf("logit shape mismatch: got %v want %v", got.Shape, want.Shape)
		}
	}
	vocabulary := got.Shape[len(got.Shape)-1]
	gotValues, err := FinalValues(got, vocabulary)
	if err != nil {
		return difference, fmt.Errorf("decode distributed logits: %w", err)
	}
	wantValues, err := FinalValues(want, vocabulary)
	if err != nil {
		return difference, fmt.Errorf("decode reference logits: %w", err)
	}
	for index := range gotValues {
		actual := gotValues[index]
		expected := wantValues[index]
		if math.IsNaN(actual) || math.IsNaN(expected) {
			return difference, fmt.Errorf("NaN at vocabulary index %d", index)
		}
		if math.IsInf(actual, 0) || math.IsInf(expected, 0) {
			if actual == expected {
				continue
			}
			return difference, fmt.Errorf("non-matching infinity at vocabulary index %d", index)
		}
		absolute := math.Abs(actual - expected)
		// Logits are at most float32 precision. This floor keeps the
		// zero-reference diagnostic finite without changing representable
		// nonzero relative errors.
		relative := absolute / math.Max(math.Abs(expected), math.SmallestNonzeroFloat32)
		difference.Absolute = math.Max(difference.Absolute, absolute)
		difference.Relative = math.Max(difference.Relative, relative)
		if absolute > atol+rtol*math.Abs(expected) {
			return difference, fmt.Errorf(
				"vocabulary index %d differs: got=%g want=%g abs=%g tolerance=%g",
				index,
				actual,
				expected,
				absolute,
				atol+rtol*math.Abs(expected),
			)
		}
	}
	return difference, nil
}

// Compare checks one final-logit pair and incorporates its differences.
func (metrics *Metrics) Compare(
	got workerproc.WireTensor,
	want workerproc.WireTensor,
	rtol float64,
	atol float64,
) error {
	difference, err := CompareFinalLogits(got, want, rtol, atol)
	metrics.MaxAbsoluteDifference = math.Max(metrics.MaxAbsoluteDifference, difference.Absolute)
	metrics.MaxRelativeDifference = math.Max(metrics.MaxRelativeDifference, difference.Relative)
	if err != nil {
		return err
	}
	metrics.Comparisons++
	return nil
}

// FinalValues decodes count values from the final position of a worker tensor.
func FinalValues(tensor workerproc.WireTensor, count int) ([]float64, error) {
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
			values[index] = float64(Float16(binary.LittleEndian.Uint16(data[offset:])))
		case "float32":
			values[index] = float64(math.Float32frombits(binary.LittleEndian.Uint32(data[offset:])))
		}
	}
	return values, nil
}

// Float16 decodes one IEEE 754 binary16 value.
func Float16(bits uint16) float32 {
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

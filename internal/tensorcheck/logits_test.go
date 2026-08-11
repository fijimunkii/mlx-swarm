package tensorcheck

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

func TestCompareFinalLogitsUsesLastPosition(t *testing.T) {
	got := float32Tensor([]int{1, 2, 2}, []float32{99, 99, 1, -2})
	want := float32Tensor([]int{1, 2, 2}, []float32{-99, -99, 1, -2})

	difference, err := CompareFinalLogits(got, want, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if difference.Absolute != 0 || difference.Relative != 0 {
		t.Fatalf("difference = %+v, want zero", difference)
	}
}

func TestMetricsAccumulatesComparisons(t *testing.T) {
	var metrics Metrics
	want := float32Tensor([]int{1, 1, 2}, []float32{1, 2})
	for _, values := range [][]float32{{1.01, 2}, {1, 2.02}} {
		if err := metrics.Compare(
			float32Tensor([]int{1, 1, 2}, values), want, 0, 0.03,
		); err != nil {
			t.Fatal(err)
		}
	}
	if metrics.Comparisons != 2 || metrics.MaxAbsoluteDifference < 0.019 {
		t.Fatalf("unexpected metrics: %+v", metrics)
	}
}

func TestCompareFinalLogitsRejectsMismatch(t *testing.T) {
	got := float32Tensor([]int{1, 1, 2}, []float32{1, 2.01})
	want := float32Tensor([]int{1, 1, 2}, []float32{1, 2})

	if _, err := CompareFinalLogits(got, want, 1e-4, 1e-4); err == nil {
		t.Fatal("accepted an out-of-tolerance value")
	}
}

func TestCompareFinalLogitsHandlesNonFiniteValues(t *testing.T) {
	infinity := float32Tensor([]int{1, 1, 1}, []float32{float32(math.Inf(1))})
	if _, err := CompareFinalLogits(infinity, infinity, 0, 0); err != nil {
		t.Fatalf("matching infinities: %v", err)
	}
	negativeInfinity := float32Tensor([]int{1, 1, 1}, []float32{float32(math.Inf(-1))})
	if _, err := CompareFinalLogits(infinity, negativeInfinity, 0, 0); err == nil {
		t.Fatal("accepted non-matching infinities")
	}
}

func TestCompareFinalLogitsRejectsInvalidTolerances(t *testing.T) {
	tensor := float32Tensor([]int{1, 1, 1}, []float32{1})
	for _, tolerance := range []float64{-1, math.NaN(), math.Inf(1)} {
		if _, err := CompareFinalLogits(tensor, tensor, tolerance, 0); err == nil {
			t.Fatalf("accepted rtol %g", tolerance)
		}
		if _, err := CompareFinalLogits(tensor, tensor, 0, tolerance); err == nil {
			t.Fatalf("accepted atol %g", tolerance)
		}
	}
}

func TestCompareFinalLogitsKeepsZeroRelativeDifferenceFinite(t *testing.T) {
	got := float32Tensor([]int{1, 1, 1}, []float32{math.MaxFloat32})
	want := float32Tensor([]int{1, 1, 1}, []float32{0})
	difference, err := CompareFinalLogits(got, want, 0, math.MaxFloat64)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(difference.Relative) || math.IsInf(difference.Relative, 0) {
		t.Fatalf("non-finite difference: %+v", difference)
	}
}

func TestFloat16(t *testing.T) {
	tests := map[uint16]float32{
		0x0000: 0,
		0x0001: float32(math.Ldexp(1, -24)),
		0x3c00: 1,
		0xc000: -2,
		0x7c00: float32(math.Inf(1)),
	}
	for bits, want := range tests {
		if got := Float16(bits); got != want {
			t.Errorf("Float16(%#x) = %g, want %g", bits, got, want)
		}
	}
}

func float32Tensor(shape []int, values []float32) workerproc.WireTensor {
	data := make([]byte, len(values)*4)
	for index, value := range values {
		binary.LittleEndian.PutUint32(data[index*4:], math.Float32bits(value))
	}
	return workerproc.WireTensor{Shape: shape, DType: "float32", Data: data}
}

package main

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/fijimunkii/mlx-swarm/internal/workerproc"
)

func TestCompareFinalLogitsUsesLastPosition(t *testing.T) {
	got := float32Tensor([]int{1, 2, 2}, []float32{99, 99, 1, -2})
	want := float32Tensor([]int{1, 2, 2}, []float32{-99, -99, 1, -2})

	absolute, relative, err := compareFinalLogits(got, want, 0, 0)
	if err != nil {
		t.Fatalf("compareFinalLogits: %v", err)
	}
	if absolute != 0 || relative != 0 {
		t.Fatalf("differences = %g, %g; want zero", absolute, relative)
	}
}

func TestCompareFinalLogitsRejectsMismatch(t *testing.T) {
	got := float32Tensor([]int{1, 1, 2}, []float32{1, 2.01})
	want := float32Tensor([]int{1, 1, 2}, []float32{1, 2})

	if _, _, err := compareFinalLogits(got, want, 1e-4, 1e-4); err == nil {
		t.Fatal("compareFinalLogits accepted an out-of-tolerance value")
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
		if got := float16(bits); got != want {
			t.Errorf("float16(%#x) = %g, want %g", bits, got, want)
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

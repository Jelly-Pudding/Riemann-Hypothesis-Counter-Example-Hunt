package main

import (
	"math"
	"testing"
)

// Benchmark the production base pass: dip-collecting scan at the base
// lattice over a representative stretch.
func BenchmarkBasePass(b *testing.B) {
	a := 3e12
	e := a + 30000
	bg, _ := buildBlockGrid(a, e, 16)
	h := math.Ldexp(1, -8)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bg.scan(a, e, h, 16, true)
	}
}

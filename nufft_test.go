package main

import (
	"math"
	"math/cmplx"
	"math/rand"
	"testing"
)

// The NUFFT must match brute-force direct summation to near machine
// precision (relative to sum|c|), across the whole output range
// including the edge modes where deconvolution amplifies most.
func TestNUFFTMatchesDirect(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, cfg := range []struct{ m, R int }{
		{3000, 1024},
		{500, 4096},
		{20000, 512},
	} {
		x := make([]float64, cfg.m)
		c := make([]complex128, cfg.m)
		sumAbs := 0.0
		for n := range x {
			x[n] = rng.Float64() * 2 * math.Pi
			c[n] = complex(rng.NormFloat64(), rng.NormFloat64())
			sumAbs += cmplx.Abs(c[n])
		}
		got := nufft1(x, c, cfg.R, 4)
		worst := 0.0
		for j := 0; j < cfg.R; j++ {
			k := float64(j - cfg.R/2)
			var direct complex128
			for n := range x {
				s, cs := math.Sincos(k * x[n])
				direct += c[n] * complex(cs, s)
			}
			if d := cmplx.Abs(got[j] - direct); d > worst {
				worst = d
			}
		}
		if worst > 1e-11*sumAbs {
			t.Errorf("m=%d R=%d: worst abs err %.3e (%.3e relative to sum|c|)",
				cfg.m, cfg.R, worst, worst/sumAbs)
		}
	}
}

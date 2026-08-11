package main

import (
	"math"
	"math/cmplx"
	"testing"
)

// Adjacent scans must tile exactly: splitting a range at any point --
// lattice-aligned or not -- must preserve the total zero count, with
// every zero attributed to exactly one side.
func TestScanTilingInvariant(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a 690k-entry table")
	}
	a, b := 3.0e12+100.25, 3.0e12+1300.75
	h := math.Ldexp(1, -8)
	whole, _ := scanRange(a, b, h, 8)
	for _, mid := range []float64{3.0e12 + 656, 3.0e12 + 656.3} { // aligned, unaligned
		p1, _ := scanRange(a, mid, h, 8)
		p2, _ := scanRange(mid, b, h, 8)
		if len(p1)+len(p2) != len(whole) {
			t.Errorf("split at %.2f: %d + %d != %d", mid, len(p1), len(p2), len(whole))
		}
		for _, z := range p1 {
			if z > mid {
				t.Errorf("split at %.2f: left side reported zero at %.6f beyond the split", mid, z)
			}
		}
		for _, z := range p2 {
			if z <= mid-h {
				t.Errorf("split at %.2f: right side reported zero at %.6f too far left", mid, z)
			}
		}
	}
}

// The tapered-sinc kernel must reconstruct pure tones across the whole
// design band (|freq| <= 0.2 cycles/sample for interpSigma = 2.5) to
// near the kernel's theoretical error.
func TestInterpKernelTone(t *testing.T) {
	si := &sInterp{hi: 1}
	si.initKernel()
	const N = 256
	si.B = make([]complex128, N)
	for _, freq := range []float64{0, 0.05, -0.1, 0.15, 0.2, -0.2} {
		om := 2 * math.Pi * freq
		for j := 0; j < N; j++ {
			s, c := math.Sincos(om * float64(j))
			si.B[j] = complex(c, s)
		}
		worst := 0.0
		for i := 0; i < 40; i++ {
			u := 60 + 130*float64(i)/40 + 0.017
			s, c := math.Sincos(om * u)
			want := complex(c, s)
			if d := cmplx.Abs(si.evalB(u) - want); d > worst {
				worst = d
			}
		}
		if worst > 5e-8 {
			t.Errorf("freq=%+.2f cyc/sample: worst reconstruction error %.3e", freq, worst)
		}
	}
}

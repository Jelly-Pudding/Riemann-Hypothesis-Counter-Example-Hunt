package main

import (
	"math"
	"testing"
)

// Production regression: the block at 3002139e6 leaked a pair with gap
// of exactly one ulp past every window-based rescan on the server.
// dipHunt must recover it from the finest-lattice zero list.
func TestDipHuntRecovers2139(t *testing.T) {
	if testing.Short() {
		t.Skip("frontier-height scan")
	}
	a, b := 3002139000000.0, 3002140000000.0
	exp := nDiff(a, b)
	bg, _ := buildBlockGrid(a, b, 16)
	mids, _ := bg.scanZ(a, b, math.Ldexp(1, -11), 16)
	short := exp - float64(len(mids))
	if short < 1.5 {
		t.Fatalf("expected the lattice scan to be ~2 short, got %.3f", short)
	}
	mids2, _, rec := dipHunt(bg, a, b, mids, 16)
	if rec != 2 {
		t.Fatalf("dipHunt recovered %d zeros, want 2", rec)
	}
	if d := float64(len(mids2)) - exp; math.Abs(d) > 1.5 {
		t.Errorf("after dipHunt count is off by %+.3f", d)
	}
}

package main

import (
	"math"
	"testing"
)

// The production pair at 3002139944606.3530/.3535 has a gap of exactly
// one float64 ulp. Before u-space scanning the interpolation lattice
// could not go below one ulp, so this pair was only reachable via dip
// fingerprints plus direct dd-grid probes. A sub-ulp lattice must now
// resolve it as two plain sign changes.
func TestSubUlpScanResolvesPair(t *testing.T) {
	const z1, z2 = 3002139944606.3530, 3002139944606.3535
	a, b := z1-2.0, z2+2.0
	bg, _ := buildBlockGrid(a, b, 8)
	ulp := math.Nextafter(b, math.Inf(1)) - b
	mids, _ := bg.scanZ(z1-0.01, z2+0.01, ulp/8, 8)
	if len(mids) != 2 {
		t.Fatalf("sub-ulp scan found %d crossings in the pair window, want 2 (mids %v)", len(mids), mids)
	}
	if math.Abs(mids[0]-z1) > 2*ulp || math.Abs(mids[1]-z2) > 2*ulp {
		t.Fatalf("crossings at %.4f, %.4f; want near %.4f, %.4f", mids[0], mids[1], z1, z2)
	}
}

// A sub-ulp lattice must agree with the one-ulp lattice zero for zero
// over a stretch with no sub-ulp structure.
func TestSubUlpMatchesUlpLattice(t *testing.T) {
	a := 3e12
	b := a + 30
	bg, _ := buildBlockGrid(a, b, 8)
	ulp := math.Nextafter(b, math.Inf(1)) - b
	m1, _ := bg.scanZ(a, b, ulp, 8)
	m2, _ := bg.scanZ(a, b, ulp/4, 8)
	if len(m1) != len(m2) {
		t.Fatalf("ulp lattice found %d zeros, ulp/4 lattice found %d", len(m1), len(m2))
	}
	for i := range m1 {
		if math.Abs(m1[i]-m2[i]) > ulp {
			t.Fatalf("zero %d moved from %.6f to %.6f", i, m1[i], m2[i])
		}
	}
}

// scanFull (the heavy-pass list builder) must produce complete lists:
// on the tight-pair-rich storm block its 8x dip-first pass must land on
// the independently established truth exactly, proving replacement
// lists never lose probe-recovered pairs.
func TestScanFullStormBlock(t *testing.T) {
	if testing.Short() {
		t.Skip()
	}
	a, b := 3001575e6, 3001576e6
	bg, _ := buildBlockGrid(a, b, 16)
	mids, _ := scanFull(bg, a, b, math.Ldexp(1, -11), 16)
	if len(mids) != 4280041 {
		t.Fatalf("scanFull found %d zeros in the storm block, truth is 4280041", len(mids))
	}
}

// The fast base-pass kernel must stay within its ~1.2e-5 error budget
// of the full kernel and find the identical crossing set.
func TestFastKernelMatchesFull(t *testing.T) {
	a := 3e12
	b := a + 30
	bg, _ := buildBlockGrid(a, b, 8)
	h := math.Ldexp(1, -8)
	n := int((b - a) / h)
	zf := make([]float64, n)
	za := make([]float64, n)
	bg.evalRange(a, h, 0, n, 8, zf, true)
	bg.evalRange(a, h, 0, n, 8, za, false)
	worst := 0.0
	cf, ca := 0, 0
	for j := 1; j < n; j++ {
		if d := math.Abs(zf[j] - za[j]); d > worst {
			worst = d
		}
		if (zf[j-1] < 0) != (zf[j] < 0) {
			cf++
		}
		if (za[j-1] < 0) != (za[j] < 0) {
			ca++
		}
	}
	if worst > 5e-5 {
		t.Fatalf("fast kernel deviates by %g from full kernel, budget 5e-5", worst)
	}
	if cf != ca {
		t.Fatalf("fast kernel found %d crossings, full kernel %d", cf, ca)
	}
}

// The fast base pass re-checks hairline crossings with the full kernel
// before counting them. On a clean stretch the veto must never fire and
// the fast scan's crossing count must match the full-kernel scan's.
func TestFastScanPhantomVeto(t *testing.T) {
	a := 3e12
	b := a + 30
	bg, _ := buildBlockGrid(a, b, 8)
	phantomDrops.Store(0)
	h := math.Ldexp(1, -8)
	m1, _, _ := bg.scan(a, b, h, 8, true, true)
	m2, _, _ := bg.scan(a, b, h, 8, true, false)
	if len(m1) != len(m2) {
		t.Fatalf("fast scan found %d crossings, full scan %d", len(m1), len(m2))
	}
	if n := phantomDrops.Swap(0); n != 0 {
		t.Fatalf("full-kernel veto dropped %d crossings on a clean stretch", n)
	}
}

// At t = 2.9e13 one ulp is 2^-7 and the base lattice (2^-9 here) sits
// below it: the old engine could not scan at all at this height. The
// count over a 50-unit window must match the argument principle within
// normal S wobble.
func TestHighHeightSubUlpBaseScan(t *testing.T) {
	if testing.Short() {
		t.Skip()
	}
	a := 2.9e13
	b := a + 50
	bg, _ := buildBlockGrid(a, b, 8)
	mids, _ := bg.scanZ(a, b, math.Ldexp(1, -9), 8)
	exp := nDiff(a, b)
	if d := float64(len(mids)) - exp; math.Abs(d) > 3 {
		t.Fatalf("found %d zeros, expected %.3f (drift %+.1f)", len(mids), exp, d)
	}
}

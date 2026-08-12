package main

import (
	"math"
	"testing"
)

// Independent truth for the disputed storm block: crossings at the
// finest lattice plus dipHunt for anything below it.
func TestTruthStormBlock1(t *testing.T) {
	if testing.Short() {
		t.Skip()
	}
	a, b := 3001575000000.0, 3001576000000.0
	exp := nDiff(a, b)
	bg, _ := buildBlockGrid(a, b, 16)
	mids, _ := bg.scanZ(a, b, math.Ldexp(1, -11), 16)
	t.Logf("ulp-lattice crossings: %d, expected %.3f", len(mids), exp)
	mids2, _, rec := dipHunt(bg, a, b, mids, 16)
	t.Logf("dipHunt recovered %d more; TRUE TOTAL %d (expected %.3f)", rec, len(mids2), exp)
}

package main

import (
	"math"
	"sort"
	"testing"
)

// The certification machinery end-to-end at the frontier: two anchors
// bracketing a stretch must certify an absolute-count difference that
// matches the found zeros between them exactly (integer equality, no
// S-wobble), a corrupted anchor window must refuse to certify, and a
// pair deleted between the anchors must surface as a deficit of exactly
// 2 -- the jackpot signal the whole hunt is built around.
func TestTuringAnchorCertifies(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a 690k-entry table")
	}
	a, b := 3.0e12, 3.0e12+80
	bg, _ := buildBlockGrid(a, b, 8)
	mids, _ := bg.scanZ(a, b, math.Ldexp(1, -11), 8) // dense: complete list
	tA, tB := a+2, b-turingL-2
	n0, r0, ok0 := turingAnchor(tA, mids)
	n1, r1, ok1 := turingAnchor(tB, mids)
	if !ok0 || !ok1 {
		t.Fatalf("anchors failed on a clean dense scan (residuals %+.2f, %+.2f)", r0, r1)
	}
	lo := sort.SearchFloat64s(mids, tA)
	hi := sort.SearchFloat64s(mids, tB)
	if certified, found := n1-n0, int64(hi-lo); certified != found {
		t.Fatalf("certified %d zeros between anchors, found %d", certified, found)
	}

	// Fail-safe: delete a zero near the middle of the first anchor window
	// (shifts the S-integral by ~L/2, far past the Trudgian bound); the
	// anchor must refuse rather than certify a corrupt window.
	winMid := sort.SearchFloat64s(mids, tA+turingL/2)
	cut := append(append([]float64{}, mids[:winMid]...), mids[winMid+1:]...)
	if _, res, ok := turingAnchor(tA, cut); ok {
		t.Fatalf("anchor certified a window missing a zero (residual %+.2f)", res)
	}

	// Jackpot shape: delete a pair BETWEEN the windows. Both anchor
	// windows stay intact, so both certifications are unchanged and the
	// found count drops by exactly 2.
	victim := sort.SearchFloat64s(mids, tA+turingL+20)
	cut = append(append([]float64{}, mids[:victim]...), mids[victim+2:]...)
	m0, _, okA := turingAnchor(tA, cut)
	m1, _, okB := turingAnchor(tB, cut)
	if !okA || !okB {
		t.Fatal("anchors failed after deleting zeros outside their windows")
	}
	if m0 != n0 || m1 != n1 {
		t.Fatalf("certified counts moved (%d->%d, %d->%d) though windows are intact", n0, m0, n1, m1)
	}
	lo = sort.SearchFloat64s(cut, tA)
	hi = sort.SearchFloat64s(cut, tB)
	if deficit := (m1 - m0) - int64(hi-lo); deficit != 2 {
		t.Fatalf("deleted pair shows as deficit %d, want exactly 2", deficit)
	}
}

package main

import (
	"math"
	"sort"
)

// Turing's method with Trudgian's explicit constants (Improvements to
// Turing's Method II, Rocky Mountain J. Math. 46 (2016), arXiv:1406.3416,
// Theorem 1): for t2 > t1 > 1e5,
//
//	|integral_{t1}^{t2} S(t) dt| <= 1.698 + 0.183 loglog t2 + 0.049 log t2
//
// (Constants verified against the paper. They are tuned for t1 ~ 1e10 but
// the paper states the whole table is valid for all t1 >= 1e5; at the
// hunt's heights ~3e12 the bound evaluates to ~3.7, and uniqueness needs
// only L > 2*(bound + slack) = ~7.7 < 12.)
//
// where N(t) = theta(t)/pi + 1 + S(t). Over a window [T, T+L] the
// integral I(M) = M*L + sum_{z in window} (T+L-z) - integral(theta/pi+1)
// equals integral of S for the TRUE absolute count M = N(T), and shifts
// by exactly k*L for a count off by k. With L > 2B only one M fits the
// bound, so the absolute count at T is CERTIFIED, not estimated. The
// found count between consecutive certified anchors must then match the
// anchor difference exactly: any discrepancy is a proven integer number
// of missing (or phantom) zeros, free of S-wobble ambiguity.

const turingL = 12.0

// turingBound is Trudgian's explicit bound on |int S| for windows
// ending at t2.
func turingBound(t2 float64) float64 {
	return 1.698 + 0.183*math.Log(math.Log(t2)) + 0.049*math.Log(t2)
}

// turingAnchor certifies the absolute zero count N(T) using the found
// zeros in (T, T+L]. Returns the certified count and the residual
// integral (for logging). ok is false when no count fits the bound,
// which happens when the window itself is missing a zero: the method
// is fail-safe and refuses to certify a corrupt window.
func turingAnchor(T float64, mids []float64) (int64, float64, bool) {
	t2 := T + turingL
	// zSum = integral of the found-count step function above its value at T.
	lo := sort.SearchFloat64s(mids, T)
	hi := sort.SearchFloat64s(mids, t2)
	zSum := 0.0
	for _, z := range mids[lo:hi] {
		zSum += t2 - z
	}
	inWindow := hi - lo
	// G = integral over [T, t2] of (theta(t)/pi + 1) dt, composite
	// Simpson in double-double (the integrand is ~1e13; dd accumulation
	// avoids cancellation when subtracting M*L + zSum).
	const nS = 64
	hS := turingL / nS
	var acc dd
	for i := 0; i <= nS; i++ {
		w := 2.0
		if i == 0 || i == nS {
			w = 1
		} else if i%2 == 1 {
			w = 4
		}
		f := ddAddD(ddDivD(thetaDDt(dd{T + float64(i)*hS, 0}), math.Pi), 1)
		acc = ddAdd(acc, ddMulD(f, w))
	}
	G := ddMulD(acc, hS/3)

	// The unique admissible absolute count.
	mReal := ddMulD(ddAddD(G, -zSum), 1/turingL)
	M := math.Round(mReal.hi + mReal.lo)
	// I(M) = M*L + zSum - G, computed in dd.
	iM := ddAdd(twoProd(M, turingL), dd{zSum, 0})
	iM = ddSub(iM, G)
	res := iM.hi + iM.lo

	// Measurement slack: found positions are accurate to half a lattice
	// step; ~50 zeros in the window contribute under 0.15 total.
	b := turingBound(t2) + 0.15
	if math.Abs(res) > b || inWindow == 0 {
		return 0, res, false
	}
	return int64(M), res, true
}

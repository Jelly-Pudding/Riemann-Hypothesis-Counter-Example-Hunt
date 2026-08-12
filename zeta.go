package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"sync"
)

// Bernoulli numbers B_{2k}, k = 1..14, used in the Euler-Maclaurin tail.
var bern = []float64{
	1.0 / 6.0,
	-1.0 / 30.0,
	1.0 / 42.0,
	-1.0 / 30.0,
	5.0 / 66.0,
	-691.0 / 2730.0,
	7.0 / 6.0,
	-3617.0 / 510.0,
	43867.0 / 798.0,
	-174611.0 / 330.0,
	854513.0 / 138.0,
	-236364091.0 / 2730.0,
	8553103.0 / 6.0,
	-23749461029.0 / 870.0,
}

// Zeta computes the Riemann zeta function for any s != 1.
// For Re(s) >= 0 it uses Euler-Maclaurin summation; for Re(s) < 0 it
// reflects with the functional equation. Accuracy is roughly 1e-12 at
// moderate heights; cost grows linearly with |Im(s)|, so this is the
// slow-but-trusted evaluator (practical up to |Im(s)| ~ 1e6). Use ZFast
// for scanning at large t.
func Zeta(s complex128) complex128 {
	if real(s) < 0 {
		// zeta(s) = 2^s pi^(s-1) sin(pi s/2) Gamma(1-s) zeta(1-s)
		return cmplx.Pow(2, s) * cmplx.Pow(math.Pi, s-1) *
			cmplx.Sin(s*complex(math.Pi/2, 0)) * gammaC(1-s) * Zeta(1-s)
	}

	t := math.Abs(imag(s))
	// N > |s|/(2*pi) makes successive tail terms shrink by ~(|s|/(2*pi*N))^2;
	// 1.3*t gives a per-term ratio < 0.015, so 14 terms reach ~1e-13.
	N := int(math.Max(30, 1.3*t))
	Nc := complex(float64(N), 0)

	sum := complex(0, 0)
	for n := 1; n < N; n++ {
		sum += cmplx.Pow(complex(float64(n), 0), -s)
	}
	sum += cmplx.Pow(Nc, 1-s) / (s - 1)
	sum += cmplx.Pow(Nc, -s) / 2

	// Tail: sum_k B_{2k}/(2k)! * N^(1-s-2k) * s(s+1)...(s+2k-2)
	term := cmplx.Pow(Nc, -s-1) * s
	fact := 2.0 // (2k)!
	sum += complex(bern[0]/fact, 0) * term
	for k := 2; k <= len(bern); k++ {
		term *= (s + complex(float64(2*k-3), 0)) * (s + complex(float64(2*k-2), 0)) / (Nc * Nc)
		fact *= float64(2*k-1) * float64(2*k)
		dt := complex(bern[k-1]/fact, 0) * term
		sum += dt
		if cmplx.Abs(dt) < 1e-17*cmplx.Abs(sum) {
			break
		}
	}
	return sum
}

// Lanczos approximation coefficients (g = 7).
var lanczos = []float64{
	0.99999999999980993,
	676.5203681218851,
	-1259.1392167224028,
	771.32342877765313,
	-176.61502916214059,
	12.507343278686905,
	-0.13857109526572012,
	9.9843695780195716e-6,
	1.5056327351493116e-7,
}

func gammaC(z complex128) complex128 {
	if real(z) < 0.5 {
		return complex(math.Pi, 0) / (cmplx.Sin(complex(math.Pi, 0)*z) * gammaC(1-z))
	}
	z -= 1
	x := complex(lanczos[0], 0)
	for i := 1; i < len(lanczos); i++ {
		x += complex(lanczos[i], 0) / (z + complex(float64(i), 0))
	}
	tt := z + complex(7.5, 0)
	return complex(math.Sqrt(2*math.Pi), 0) * cmplx.Pow(tt, z+0.5) * cmplx.Exp(-tt) * x
}

// Theta is the Riemann-Siegel theta function collapsed to float64.
func Theta(t float64) float64 {
	th := thetaDD(t)
	return th.hi + th.lo
}

// Z is the Hardy Z function computed from the accurate Zeta:
// Z(t) = exp(i*theta(t)) * zeta(1/2 + it), which is real for real t.
// Zeros of Z on the real axis are exactly the zeros of zeta on the
// critical line. Inherits Zeta's O(t) cost so low/moderate t only.
func Z(t float64) float64 {
	th := ddMod2Pi(thetaDD(t))
	return real(cmplx.Exp(complex(0, th)) * Zeta(complex(0.5, t)))
}

// Shared read-only tables for the Riemann-Siegel main sum: ln(n) in
// double-double and 1/sqrt(n). Grown under a mutex; callers receive
// slice headers that stay valid even if the tables are later regrown.
var (
	tabMu    sync.Mutex
	lnTab    []dd
	rsqrtTab []float64
)

const maxTableN = 200_000_000 // ~4.8 GB of tables; reached near t ~ 2.5e17

func getTables(m int) ([]dd, []float64) {
	tabMu.Lock()
	defer tabMu.Unlock()
	if len(lnTab) < m+1 {
		if m > maxTableN {
			panic(fmt.Sprintf("Riemann-Siegel sum needs %d terms; table cap is %d", m, maxTableN))
		}
		newLen := m + 1
		if newLen < 2*len(lnTab) {
			newLen = 2 * len(lnTab)
		}
		nl := make([]dd, newLen)
		nr := make([]float64, newLen)
		copy(nl, lnTab)
		copy(nr, rsqrtTab)
		start := len(lnTab)
		if start < 1 {
			start = 1
		}
		for n := start; n < newLen; n++ {
			nl[n] = lnDD(float64(n))
			nr[n] = 1 / math.Sqrt(float64(n))
		}
		lnTab, rsqrtTab = nl, nr
	}
	return lnTab, rsqrtTab
}

// ZFast is the Riemann-Siegel main sum with the first (C0) correction
// term. Cost is O(sqrt(t)) per call. Formula error is O((t/2pi)^(-3/4))
// which shrinks with height and is ~1e-9 at t = 3e12, meaning C0 alone
// suffices there. The phase theta(t) - t*ln(n) is carried in
// double-double, keeping it exact far beyond t = 1e20.
func ZFast(t float64) float64 {
	a := math.Sqrt(t / (2 * math.Pi))
	m := int(a)
	ln, rs := getTables(m)
	th := thetaDD(t)

	var sum, comp float64 // Kahan-compensated main sum
	for n := 1; n <= m; n++ {
		ph := ddMod2Pi(ddSub(th, ddMulD(ln[n], t)))
		term := math.Cos(ph) * rs[n]
		y := term - comp
		s := sum + y
		comp = (s - sum) - y
		sum = s
	}
	sum *= 2
	return sum + rsRemainder(t, m)
}

// rsRemainder is the Riemann-Siegel C0 correction term:
// (-1)^(m-1) (t/2pi)^(-1/4) * C0(p), where
// C0(p) = cos(2pi(p^2 - p - 1/16)) / cos(2pi*p), with removable
// singularities at p = 1/4, 3/4 handled by L'Hopital.
func rsRemainder(t float64, m int) float64 {
	a := math.Sqrt(t / (2 * math.Pi))
	p := a - float64(m)
	num := 2 * math.Pi * (p*p - p - 1.0/16.0)
	den := 2 * math.Pi * p
	var c0 float64
	if math.Abs(math.Cos(den)) < 1e-4 {
		c0 = (2*p - 1) * math.Sin(num) / math.Sin(den)
	} else {
		c0 = math.Cos(num) / math.Cos(den)
	}
	r := c0 / math.Sqrt(a)
	if m%2 == 0 {
		r = -r
	}
	return r
}

// NApprox is the Riemann-von Mangoldt zero count N(T) ~= theta(T)/pi + 1.
// The exact theorem is N(T) = theta(T)/pi + 1 + S(T) with S(T) a
// zero-mean fluctuation, almost always |S| < 2 (max ever observed ~3.2).
// So this is exact up to that small wobble. Off-line zeros come in
// mirror pairs (sigma, 1-sigma), so a genuine counterexample shows up as
// a persistent deficit of at least 2 sign changes against this count.
func NApprox(T float64) float64 {
	return Theta(T)/math.Pi + 1
}

// nDiff is the expected number of zeros with height in (t0, t1),
// computed as (theta(t1)-theta(t0))/pi in double-double so the
// difference is exact even when theta itself is ~4e13.
func nDiff(t0, t1 float64) float64 {
	d := ddSub(thetaDD(t1), thetaDD(t0))
	return (d.hi + d.lo) / math.Pi
}

// FindZeros locates zeros of zeta on the critical line with imaginary
// part in (t0, t1): scan ZFast for sign changes, then (for t up to 1e6,
// where the Euler-Maclaurin Z is affordable) bisect down to ~1e-10.
// Above that the sign-change midpoint is returned.
func FindZeros(t0, t1 float64) []float64 {
	var zeros []float64
	// Mean zero spacing near height t is 2pi/log(t/2pi); step at a
	// quarter of that to make missed pairs unlikely.
	step := func(t float64) float64 {
		d := 2 * math.Pi / math.Log(math.Max(t, 10)/(2*math.Pi))
		return math.Max(d/4, 1e-3)
	}
	t := math.Max(t0, 10) // R-S expansion needs t/2pi > 1
	prev := ZFast(t)
	for t < t1 {
		next := t + step(t)
		if next > t1 {
			next = t1
		}
		cur := ZFast(next)
		if (prev < 0) != (cur < 0) {
			if next <= 1e6 {
				zeros = append(zeros, bisect(t, next))
			} else {
				zeros = append(zeros, (t+next)/2)
			}
		}
		t, prev = next, cur
	}
	return zeros
}

func bisect(lo, hi float64) float64 {
	flo := Z(lo)
	for i := 0; i < 100 && hi-lo > 1e-11; i++ {
		mid := (lo + hi) / 2
		fm := Z(mid)
		if (flo < 0) != (fm < 0) {
			hi = mid
		} else {
			lo, flo = mid, fm
		}
	}
	return (lo + hi) / 2
}

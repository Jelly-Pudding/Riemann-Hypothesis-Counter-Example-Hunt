package main

import (
	"fmt"
	"math"
)

// Band-limited interpolation of the Riemann-Siegel main sum -- the
// second half of the Odlyzko-Schonhage method (the half Gourdon's
// record verification used to make rescans free).
//
// The main sum S(t) = sum_{n<=m} n^{-1/2} e^{-i t ln n} is band-limited:
// its spectrum lives in [-ln m, 0]. Sampled at interpSigma times the
// complex Nyquist rate (spacing 2pi/(sigma*ln m) -- about 1.25 samples
// per zero spacing), S is reconstructed anywhere by a Gaussian-tapered
// sinc kernel over 2*interpL neighboring samples. One cheap NUFFT pass
// then supports unlimited re-evaluation at ~100ns per point, so dense
// sign scans and close-pair hunts no longer touch the 690k-term sum.

const (
	interpSigma = 2.5 // oversampling vs complex Nyquist
	interpL     = 20  // kernel taps per side
)

type sInterp struct {
	tA   float64 // t of sample 0 (exact double anchor)
	hi   float64 // sample spacing
	wc   float64 // demodulation frequency ln(m)/2
	m    int     // main-sum length for this segment
	segA float64 // Z-domain this grid serves: [segA, segB)
	segB float64
	B    []complex128 // demodulated samples: B[j] = S(t_j) e^{+i j hi wc}
	a2   float64      // Gaussian taper width^2
	e3   [interpL + 1]float64
}

func buildSInterp(segA, segB float64, m, workers int) (*sInterp, int64) {
	W := math.Log(float64(m))
	hi := 2 * math.Pi / W / interpSigma
	pad := float64(interpL+8) * hi
	tA := segA - pad
	R := int(math.Ceil((segB+pad-tA)/hi)) + 2

	ln, rs := getTables(m)
	x := make([]float64, m)
	parallelChunks(m, workers, func(_, nLo, nHi int) {
		for n := nLo; n <= nHi; n++ {
			x[n-1] = ddMod2Pi(ddMulD(ln[n], -hi))
		}
	})
	S := make([]complex128, R)
	nChunks := (R + maxNUFFTR - 1) / maxNUFFTR
	chunkLen := (R + nChunks - 1) / nChunks
	for cs := 0; cs < R; cs += chunkLen {
		ce := min(R, cs+chunkLen)
		computeS(tA, hi, cs, ce-cs, workers, x, ln, rs, S[cs:ce])
	}

	si := &sInterp{tA: tA, hi: hi, wc: W / 2, m: m, segA: segA, segB: segB, B: S}
	si.initKernel()
	hw := hi * si.wc
	parallelRange(R, workers, func(lo, hi2 int) {
		for j := lo; j < hi2; j++ {
			s, c := math.Sincos(float64(j) * hw)
			S[j] *= complex(c, s)
		}
	})
	return si, int64(R)
}

// initKernel sets the Gaussian taper width balancing truncation error
// e^(-L^2/a2) against passband response error e^(-pi^2 a2 d^2), where
// d = (1-1/sigma)/2 is the guard band between the signal edge and the
// sinc cutoff. Both land at e^(-pi L (1-1/sigma)/2) ~ 6e-9 for L=20,
// sigma=2.5.
func (si *sInterp) initKernel() {
	si.a2 = 2 * float64(interpL) / (math.Pi * (1 - 1/interpSigma))
	for l := 0; l <= interpL; l++ {
		si.e3[l] = math.Exp(-float64(l*l) / si.a2)
	}
}

// evalB reconstructs the demodulated (baseband) sum at fractional
// sample coordinate u. Gaussian-tapered sinc; the Gaussian is split as
// exp(-(f-l)^2/a2) = E1 * E2^l * e3[|l|] so the loop is two exps total.
func (si *sInterp) evalB(u float64) complex128 {
	j0 := int(u)
	f := u - float64(j0)
	if f < 1e-9 {
		return si.B[j0]
	}
	if f > 1-1e-9 {
		return si.B[j0+1]
	}
	s0 := math.Sin(math.Pi*f) / math.Pi
	e1 := math.Exp(-f * f / si.a2)
	e2 := math.Exp(2 * f / si.a2)
	p := e1 * s0
	var acc complex128
	// l = 0, -1, ..., -(interpL-1)
	q, sign := p, 1.0
	e2inv := 1 / e2
	for k := 0; k < interpL; k++ {
		acc += si.B[j0-k] * complex(sign*q*si.e3[k]/(f+float64(k)), 0)
		q *= e2inv
		sign = -sign
	}
	// l = 1, ..., interpL
	q, sign = p, -1.0
	for l := 1; l <= interpL; l++ {
		q *= e2
		acc += si.B[j0+l] * complex(sign*q*si.e3[l]/(f-float64(l)), 0)
		sign = -sign
	}
	return acc
}

// zAt is the single-point evaluation (tests, spot checks): full-dd
// theta, explicit demodulation. The bulk path in scanZ folds the
// demodulation into its windowed theta expansion instead.
func (si *sInterp) zAt(t float64) float64 {
	u := (t - si.tA) / si.hi
	b := si.evalB(u)
	s, c := math.Sincos(math.Mod((t-si.tA)*si.wc, 2*math.Pi))
	S := b * complex(c, -s) // e^{-i wc (t-tA)}
	th := ddMod2Pi(thetaDDt(dd{t, 0}))
	wi, wr := math.Sincos(th)
	return 2*(wr*real(S)-wi*imag(S)) + rsRemainder(t, si.m)
}

// blockGrid covers [a, b] with one sInterp per constant-m segment.
type blockGrid struct {
	segs []*sInterp
}

func buildBlockGrid(a, b float64, workers int) (*blockGrid, int64) {
	bg := &blockGrid{}
	var evals int64
	segA := a
	for segA < b {
		m := int(math.Sqrt(segA / (2 * math.Pi)))
		segB := math.Min(b, 2*math.Pi*float64(m+1)*float64(m+1))
		if segB <= segA {
			segB = math.Min(b, math.Nextafter(segA, math.MaxFloat64))
		}
		si, e := buildSInterp(segA, segB, m, workers)
		bg.segs = append(bg.segs, si)
		evals += e
		segA = segB
	}
	return bg, evals
}

func (bg *blockGrid) segFor(t float64) *sInterp {
	for _, s := range bg.segs {
		if t < s.segB || s == bg.segs[len(bg.segs)-1] {
			return s
		}
	}
	return bg.segs[len(bg.segs)-1]
}

// scanZ evaluates Z via interpolation on the dyadic lattice of spacing
// h covering [a, b] (h must be a power of two so lattice points are
// exactly representable; adjacent blocks then share the same global
// lattice and no interval is scanned twice or missed). Returns the
// sign-change midpoints and the number of evaluations.
func (bg *blockGrid) scanZ(a, b, h float64, workers int) ([]float64, int64) {
	if h < math.Nextafter(b, math.Inf(1))-b {
		// Below one ulp of t, consecutive lattice points collapse onto
		// the same float64 while the theta expansion keeps advancing --
		// producing millions of phantom crossings. Callers must switch
		// to the direct dd-grid engine instead.
		panic(fmt.Sprintf("scanZ: lattice spacing %g is below ulp(%g); use the direct engine", h, b))
	}
	a0 := math.Ceil(a/h) * h
	count := int(math.Floor((b-a0)/h)) + 1
	if count < 2 {
		return nil, 0
	}
	// The lattice point below a0 seeds the sign. Its interval
	// (a0-h, a0] is counted only when a is off-lattice (then this call
	// owns the sliver (a, a0]); when a IS a lattice point that interval
	// belongs to the neighbor below, which scanned up to a itself.
	includeSeed := a0 > a
	total := count + 1
	var mids []float64
	var prev float64
	const chunk = 1 << 22
	zs := make([]float64, min(total, chunk))
	for cs := 0; cs < total; cs += chunk {
		ce := min(total, cs+chunk)
		bg.evalRange(a0-h, h, cs, ce, workers, zs)
		lo := 0
		if cs == 0 {
			prev = zs[0]
			lo = 1
		}
		for j := lo; j < ce-cs; j++ {
			cur := zs[j]
			if (prev < 0) != (cur < 0) && (includeSeed || cs+j > 1) {
				m := a0 - h + (float64(cs+j)-0.5)*h
				if m < a {
					// Seed-interval crossings belong to (a, a0]; clamp the
					// estimate so callers' [a, b] bookkeeping stays exact.
					m = a
				}
				mids = append(mids, m)
			}
			prev = cur
		}
	}
	return mids, int64(total)
}

// evalRange fills zs[0:ce-cs] with Z at lattice points base + j*h for
// j in [cs, ce), using the windowed quadratic theta expansion (with the
// interpolation demodulation folded into the linear term).
func (bg *blockGrid) evalRange(base, h float64, cs, ce, workers int, zs []float64) {
	parallelRange(ce-cs, workers, func(lo, hi int) {
		si := bg.segs[0]
		winStart, winEnd := -1, -1
		var thC, A dd
		var Bq float64
		var q0, q1, q2 float64
		var kc int
		for j := lo; j < hi; j++ {
			t := base + float64(cs+j)*h
			if t >= si.segB && si != bg.segs[len(bg.segs)-1] {
				si = bg.segFor(t)
				winEnd = -1 // force re-anchor
			}
			if cs+j >= winEnd || cs+j < winStart {
				// New expansion window: span keeps the cubic theta term
				// below 1e-9 (span^3/(96 t^2)), at least 256 points.
				winStart = cs + j
				span := math.Cbrt(9.6e-8*t*t) / h
				w := int(span)
				if w < 256 {
					w = 256
				}
				winEnd = winStart + w
				kc = winStart + w/2
				tc := ddAddD(twoProd(h, float64(kc)), base)
				// theta' minus the demodulation slope wc, per lattice step
				thC = ddSub(thetaDDt(tc), dd{math.Mod((tc.hi-si.tA)*si.wc, 2*math.Pi), 0})
				A = ddAddD(ddMulD(ddSub(lnDDdd(tc), ddLn2Pi), h/2), -h*si.wc)
				Bq = h * h / (4 * tc.hi)
				x1 := float64(winStart - kc)
				x2 := float64(winEnd - 1 - kc)
				r0 := rsRemainder(base+float64(winStart)*h, si.m)
				rc := rsRemainder(tc.hi, si.m)
				r2 := rsRemainder(base+float64(winEnd-1)*h, si.m)
				q2 = ((r0-rc)/x1 - (r2-rc)/x2) / (x1 - x2)
				q1 = (r0-rc)/x1 - q2*x1
				q0 = rc
			}
			k := float64(cs + j - kc)
			ph := ddMod2Pi(ddAddD(ddAdd(thC, ddMulD(A, k)), Bq*k*k))
			wi, wr := math.Sincos(ph)
			u := (t - si.tA) / si.hi
			S := si.evalB(u)
			zs[j] = 2*(wr*real(S)-wi*imag(S)) + (q0 + k*(q1+k*q2))
		}
	})
}

// scanRange builds interpolation grids over [a, b] and scans at lattice
// spacing h. One-shot helper for backscans and standalone rescans.
func scanRange(a, b, h float64, workers int) ([]float64, int64) {
	bg, e1 := buildBlockGrid(a, b, workers)
	mids, e2 := bg.scanZ(a, b, h, workers)
	return mids, e1 + e2
}

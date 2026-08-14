package main

import (
	"fmt"
	"math"
	"sync/atomic"
)

// phantomDrops counts crossings vetoed by the full-kernel re-check in
// fast scans. Read-and-reset by the hunt loop for per-block logging.
var phantomDrops atomic.Int64

// tangentVerify: fast-pass crossings whose smaller flanking |Z| is below
// this are re-evaluated with the full kernel (error ~6e-9) before being
// counted. The fast kernel's ~1.2e-5 error could otherwise fabricate a
// crossing pair out of a near-tangent minimum -- the one failure mode
// that silently BALANCES the Turing ledger if a real pair goes missing
// in the same certified stretch (a phantom alone is caught as a
// surplus; phantom + miss together cancel). 1e-4 sits ~8x above the
// fast kernel's error floor yet flags well under 1% of crossings, so
// the veto costs a handful of full-kernel evaluations per block.
const tangentVerify = 1e-4

// Band-limited interpolation of the Riemann-Siegel main sum.
//
// S(t) = sum_{n<=m} n^{-1/2} e^{-i t ln n} has spectrum in [-ln m, 0].
// Sampled at interpSigma times the complex Nyquist rate (spacing
// 2pi/(sigma*ln m), about 1.25 samples per zero spacing) it can be
// reconstructed anywhere with a Gaussian-tapered sinc kernel over
// 2*interpL neighboring samples. One NUFFT pass per block then supports
// re-evaluation at any t without touching the full sum.

const (
	interpSigma = 2.5 // oversampling vs complex Nyquist
	interpL     = 20  // kernel taps per side (error ~6e-9)
	// Fast kernel for the base pass: 12 taps per side, error ~1.2e-5.
	// The base scan's decisions live far above that (dip threshold
	// 2000h^2 ~ 3e-2, crossing slopes O(1) per unit); every rescan,
	// probe and dipHunt keeps the full kernel.
	interpF = 12
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
	a2f  float64 // fast-kernel taper width^2
	e3f  [interpF + 1]float64
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
	si.a2f = 2 * float64(interpF) / (math.Pi * (1 - 1/interpSigma))
	for l := 0; l <= interpF; l++ {
		si.e3f[l] = math.Exp(-float64(l*l) / si.a2f)
	}
}

// evalB reconstructs the demodulated (baseband) sum at fractional
// sample coordinate u. Gaussian-tapered sinc; the Gaussian is split as
// exp(-(f-l)^2/a2) = E1 * E2^l * e3[|l|] so the setup is two exps total.
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
	return si.taps(j0, f, e1*s0, e2)
}

// taps is the full-kernel tap loop given precomputed taper factors:
// p = exp(-f^2/a2)*sin(pi f)/pi and e2 = exp(2f/a2).
func (si *sInterp) taps(j0 int, f, p, e2 float64) complex128 {
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

// tapsF is taps with the truncated fast kernel (interpF taps per side,
// error ~1.2e-5). Same structure, narrower taper.
func (si *sInterp) tapsF(j0 int, f, p, e2 float64) complex128 {
	var acc complex128
	q, sign := p, 1.0
	e2inv := 1 / e2
	for k := 0; k < interpF; k++ {
		acc += si.B[j0-k] * complex(sign*q*si.e3f[k]/(f+float64(k)), 0)
		q *= e2inv
		sign = -sign
	}
	q, sign = p, -1.0
	for l := 1; l <= interpF; l++ {
		q *= e2
		acc += si.B[j0+l] * complex(sign*q*si.e3f[l]/(f-float64(l)), 0)
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
	mids, _, evals := bg.scan(a, b, h, workers, false, false)
	return mids, evals
}

// scan is scanZ with optional dip-candidate collection: same-sign local
// minima of |Z| below a spacing-scaled threshold. A pair straddled
// between lattice points always leaves such a dip (flanking samples
// read |Z| of order |Z”|h^2), so collecting dips during the base pass
// locates pair hiding spots at zero extra evaluation cost. Callers
// confirm each candidate with a cheap probe. fast selects the truncated
// reconstruction kernel; dense rescans keep the full kernel because
// their dip signals scale with h^2 and sit closer to the fast kernel's
// error floor.
func (bg *blockGrid) scan(a, b, h float64, workers int, dips, fast bool) ([]float64, []float64, int64) {
	// The lattice may sit far below one ulp of t: alignment arithmetic is
	// exact at any spacing (dividing by a dyadic h never rounds, and when
	// a/h >= 2^52 it is exactly an integer, so a0 == a), and evaluation
	// runs in index space where neither the phase nor the interpolation
	// coordinate ever quantizes. Sub-ulp lattice points are not distinct
	// float64s but their Z values are genuine band-limited reconstructions
	// at the true real-number positions.
	if h < math.Ldexp(1, -40) {
		panic(fmt.Sprintf("scanZ: lattice spacing %g is absurdly fine", h))
	}
	dipThresh := math.Min(2e-2, math.Max(1e-6, 2000*h*h))
	a0 := math.Ceil(a/h) * h
	if (b-a0)/h > float64(int64(1)<<40) {
		panic(fmt.Sprintf("scanZ: %g lattice points over [%g,%g] (runaway density)", (b-a0)/h, a, b))
	}
	count := int(math.Floor((b-a0)/h)) + 1
	if count < 2 {
		return nil, nil, 0
	}
	// The lattice point below a0 seeds the sign. Its interval (a0-h, a0]
	// is counted only when a is off-lattice. When a is on the lattice
	// that interval belongs to the neighbor scan below.
	includeSeed := a0 > a
	total := count + 1
	var mids, cands []float64
	var prev, p2 float64
	const chunk = 1 << 22
	zs := make([]float64, min(total, chunk))
	for cs := 0; cs < total; cs += chunk {
		ce := min(total, cs+chunk)
		// Anchor at a0 (exactly representable) with index offset -1, so
		// point gj sits at a0 + (gj-1)*h in exact index arithmetic; the
		// old anchor a0-h is not representable when h is below one ulp.
		bg.evalRange(a0, h, cs-1, ce-1, workers, zs, fast)
		lo := 0
		if cs == 0 {
			prev = zs[0]
			lo = 1
		}
		for j := lo; j < ce-cs; j++ {
			cur := zs[j]
			gj := cs + j
			if (prev < 0) != (cur < 0) && (includeSeed || gj > 1) {
				keep := true
				if fast && math.Min(math.Abs(prev), math.Abs(cur)) < tangentVerify {
					// Hairline crossing: the zero sits within kernel-error
					// distance of a lattice point, so the fast and full
					// kernels can legitimately disagree about WHICH cell it
					// falls in (~150 per block do). Never re-check just the
					// flagged cell -- re-evaluate a +-4 cell window with the
					// full kernel (index-exact at any h). A real zero shows
					// a crossing somewhere in the window and is kept (a cell
					// of position slack is well inside the midpoint budget);
					// an isolated tiny-|Z'| zero cannot shift further than
					// that because tiny |Z'| implies a near-double zero,
					// whose partner is inside the window too. A phantom -- a
					// same-sign dip pushed across zero by the fast kernel's
					// ~1.2e-5 error -- shows no crossing at all: vetoed.
					var vf [10]float64
					bg.evalRange(a0, h, gj-6, gj+4, 1, vf[:], false)
					crossed := false
					for k := 1; k < len(vf); k++ {
						if (vf[k-1] < 0) != (vf[k] < 0) {
							crossed = true
							break
						}
					}
					if !crossed {
						phantomDrops.Add(1)
						keep = false
					}
				}
				if keep {
					m := a0 + (float64(gj)-1.5)*h
					if m < a {
						// Seed-interval crossings belong to (a, a0]; clamp the
						// estimate so callers' [a, b] bookkeeping stays exact.
						m = a
					}
					mids = append(mids, m)
				}
			}
			if dips && gj >= 2 && math.Abs(prev) < dipThresh &&
				math.Abs(prev) <= math.Abs(p2) && math.Abs(prev) <= math.Abs(cur) &&
				(p2 < 0) == (cur < 0) && (p2 < 0) == (prev < 0) && len(cands) < 8192 {
				c := a0 + float64(gj-2)*h
				if c > a && c < b {
					cands = append(cands, c)
				}
			}
			p2, prev = prev, cur
		}
	}
	return mids, cands, int64(total)
}

// evalRange fills zs[0:ce-cs] with Z at lattice points base + j*h for
// j in [cs, ce), using the windowed quadratic theta expansion (with the
// interpolation demodulation folded into the linear term). fast selects
// the truncated reconstruction kernel (~1.2e-5 vs ~6e-9).
func (bg *blockGrid) evalRange(base, h float64, cs, ce, workers int, zs []float64, fast bool) {
	parallelRange(ce-cs, workers, func(lo, hi int) {
		si := bg.segs[0]
		winStart, winEnd := -1, -1
		var thC, A dd
		var Bq float64
		var q0, q1, q2 float64
		var kc int
		var uA, du float64
		// The lattice is uniform, so every transcendental in the loop
		// advances by a constant increment and runs as a recurrence:
		// the oscillation phase e^{i ph(k)} is a rotation advanced by a
		// step that itself rotates by the constant chirp twist (re-synced
		// from the exact dd phase every resyncN points, drift ~1e-10 rad),
		// and the kernel taper factors sin(pi f), exp(-f^2/a2), exp(2f/a2)
		// advance per point and re-seed exactly on every f wrap (every
		// ~1/du points), so per-point transcendental cost is amortized out.
		const resyncN = 1 << 18
		var rotR, rotI, stR, stI, twR, twI float64
		sync := 0
		a2 := si.a2
		var kSin, kCos, kE1, kGf, kE2 float64
		lastJ0 := math.MinInt32
		var sD, cD, c1, c2, c3 float64
		phaseInit := func(k float64) {
			s, c := math.Sincos(ddMod2Pi(ddAddD(ddAdd(thC, ddMulD(A, k)), Bq*k*k)))
			rotR, rotI = c, s
			s, c = math.Sincos(ddMod2Pi(ddAddD(A, Bq*(2*k+1))))
			stR, stI = c, s
			s, c = math.Sincos(2 * Bq)
			twR, twI = c, s
			sync = 0
		}
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
				// theta' minus the demodulation slope wc, per lattice step.
				// The demodulation anchor must use the full dd window center:
				// dropping tc.lo puts a constant phase error of up to
				// wc*2^-12 on the whole window, which jitters zero midpoints
				// by about a lattice cell between window anchorings.
				thC = ddAddD(thetaDDt(tc), -ddMod2Pi(ddMulD(ddAddD(tc, -si.tA), si.wc)))
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
				// Interpolation coordinate in index space: u = uA + j*du.
				// base - si.tA is exact (Sterbenz: anchors within a factor
				// of two), so u keeps advancing smoothly even when h is
				// below one ulp of t and float64 t itself quantizes. The
				// phase above is index-exact already; with u index-exact
				// too, sub-ulp lattices produce no phantom crossings.
				uA = (base - si.tA) / si.hi
				du = h / si.hi
				if fast {
					a2 = si.a2f
				} else {
					a2 = si.a2
				}
				sD, cD = math.Sincos(math.Pi * du)
				c1 = math.Exp(-du * du / a2)
				c2 = c1 * c1
				c3 = math.Exp(2 * du / a2)
				lastJ0 = math.MinInt32
				phaseInit(float64(cs + j - kc))
			}
			k := float64(cs + j - kc)
			if sync >= resyncN {
				phaseInit(k)
			}
			wr, wi := rotR, rotI
			rotR, rotI = rotR*stR-rotI*stI, rotR*stI+rotI*stR
			stR, stI = stR*twR-stI*twI, stR*twI+stI*twR
			sync++
			u := uA + float64(cs+j)*du
			j0 := int(u)
			f := u - float64(j0)
			if j0 != lastJ0 {
				kSin, kCos = math.Sincos(math.Pi * f)
				kE1 = math.Exp(-f * f / a2)
				kGf = math.Exp(-2 * f * du / a2)
				kE2 = math.Exp(2 * f / a2)
				lastJ0 = j0
			} else {
				kSin, kCos = kSin*cD+kCos*sD, kCos*cD-kSin*sD
				kE1 *= kGf * c1
				kGf *= c2
				kE2 *= c3
			}
			var S complex128
			if f < 1e-9 {
				S = si.B[j0]
			} else if f > 1-1e-9 {
				S = si.B[j0+1]
			} else if fast {
				S = si.tapsF(j0, f, kE1*kSin/math.Pi, kE2)
			} else {
				S = si.taps(j0, f, kE1*kSin/math.Pi, kE2)
			}
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

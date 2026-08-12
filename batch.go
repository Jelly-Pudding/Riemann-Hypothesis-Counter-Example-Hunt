package main

import (
	"math"
	"sync"
)

// ZBlock evaluates the Hardy Z function on the uniform grid
// t_j = t0 + j*h, j = 0..count-1.
//
// Phase split per term:
//
//	e^{i(theta_j - t_j ln n)} = e^{i theta_j} * e^{-i t0 ln n} * (e^{-i h ln n})^j
//
// For fixed n the j-dependence is a fixed complex rotation. Anchor
// angles are computed in double-double. Rotation drift after k steps is
// ~k*1e-16 rad, fine below ~1e7 points per call.
func ZBlock(t0, h float64, count, workers int) []float64 {
	out := make([]float64, count)
	j0 := 0
	for j0 < count {
		tA := t0 + float64(j0)*h
		m := int(math.Sqrt(tA / (2 * math.Pi)))
		// The main-sum length m is constant until t reaches 2pi(m+1)^2;
		// split the grid there so each segment has a fixed term count.
		tCross := 2 * math.Pi * float64(m+1) * float64(m+1)
		j1 := count
		if t0+float64(count-1)*h >= tCross {
			j1 = j0 + int(math.Ceil((tCross-tA)/h))
			if j1 <= j0 {
				j1 = j0 + 1
			}
			if j1 > count {
				j1 = count
			}
		}
		zBlockSegment(t0, h, j0, j1, m, workers, out)
		j0 = j1
	}
	return out
}

// parallelChunks runs fn over contiguous sub-ranges of 1..m.
func parallelChunks(m, workers int, fn func(w, nLo, nHi int)) {
	if workers < 1 {
		workers = 1
	}
	chunk := (m + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		nLo := 1 + w*chunk
		nHi := min(m, nLo+chunk-1)
		if nLo > nHi {
			continue
		}
		wg.Add(1)
		go func(w, nLo, nHi int) { defer wg.Done(); fn(w, nLo, nHi) }(w, nLo, nHi)
	}
	wg.Wait()
}

// gridT returns t0 + j*h exactly as a double-double. Grid points are not
// generally representable in float64. At t ~ 3e12 the rounding (~5e-4)
// would smear the phase by ~1e-2 rad, and anchors, theta, and the
// recurrence must all use the same exact t.
func gridT(t0, h float64, j int) dd {
	return ddAddD(twoProd(h, float64(j)), t0)
}

// Below this many grid points the rotation recurrence beats the NUFFT's
// fixed costs; it also serves as an independent oracle for tests.
const rotationThreshold = 2048

// maxNUFFTR caps one NUFFT call (FFT buffer = 16*nextPow2(2R) bytes);
// larger grids are processed in chunks.
const maxNUFFTR = 1 << 22

func zBlockSegment(t0, h float64, j0, j1, m, workers int, out []float64) {
	if j1-j0 < rotationThreshold {
		zBlockSegmentRotation(t0, h, j0, j1, m, workers, out)
		return
	}
	// Tone frequencies depend only on h; share them across chunks.
	ln, rs := getTables(m)
	x := make([]float64, m)
	parallelChunks(m, workers, func(_, nLo, nHi int) {
		for n := nLo; n <= nHi; n++ {
			x[n-1] = ddMod2Pi(ddMulD(ln[n], -h))
		}
	})
	// Equal-size chunks keep every NUFFT's FFT size identical, so the
	// twiddle table and scratch buffer are reused across chunks.
	count := j1 - j0
	nChunks := (count + maxNUFFTR - 1) / maxNUFFTR
	chunkLen := (count + nChunks - 1) / nChunks
	for cs := j0; cs < j1; cs += chunkLen {
		zBlockChunkFFT(t0, h, cs, min(j1, cs+chunkLen), m, workers, x, ln, rs, out)
	}
}

// computeS fills dst with the main sum S(t_j) = sum_n rs[n]*e^{-i t_j ln n}
// at t_j = t0 + (j0+j)*h exactly, via one NUFFT. The main sum is a sum of
// pure tones with frequencies -h*ln n. Anchor phase and frequencies are
// computed in double-double.
func computeS(t0, h float64, j0, count, workers int, x []float64, ln []dd, rs []float64, dst []complex128) {
	m := len(x)
	tMid := gridT(t0, h, j0+count/2)
	c := make([]complex128, m)
	parallelChunks(m, workers, func(_, nLo, nHi int) {
		for n := nLo; n <= nHi; n++ {
			s, cs := math.Sincos(-ddMod2Pi(ddMul(ln[n], tMid)))
			c[n-1] = complex(cs*rs[n], s*rs[n])
		}
	})
	copy(dst, nufft1(x, c, count, workers))
}

func zBlockChunkFFT(t0, h float64, j0, j1, m, workers int, x []float64, ln []dd, rs []float64, out []float64) {
	count := j1 - j0
	tMid := gridT(t0, h, j0+count/2)
	S := make([]complex128, count)
	computeS(t0, h, j0, count, workers, x, ln, rs, S)

	// Combine: Z = 2*Re(e^{i theta}*S) + remainder. theta uses a quadratic
	// Taylor expansion re-anchored in double-double at window centers:
	// theta(tc+kh) = theta_c + k*(h*ln(tc/2pi)/2) + k^2*h^2/(4tc). The
	// window span keeps the cubic term span^3/(96 tc^2) below 1e-9. The
	// C0 remainder is interpolated quadratically over the same window.
	win := count
	if s := math.Cbrt(9.6e-8*tMid.hi*tMid.hi) / h; s < float64(count) {
		win = int(s)
		if win < 256 {
			win = 256
		}
	}
	parallelRange(count, workers, func(lo, hi int) {
		jw := -1
		var thC, A dd
		var B float64
		var q0, q1, q2 float64
		var kc int
		exact := false
		for j := lo; j < hi; j++ {
			if j/win != jw {
				jw = j / win
				ws := jw * win
				we := min(count, ws+win)
				exact = we-ws < 8
				if !exact {
					kc = (ws + we) / 2
					tc := gridT(t0, h, j0+kc)
					thC = thetaDDt(tc)
					A = ddMulD(ddSub(lnDDdd(tc), ddLn2Pi), h/2)
					B = h * h / (4 * tc.hi)
					x1 := float64(ws - kc)
					x2 := float64(we - 1 - kc)
					r0 := rsRemainder(gridT(t0, h, j0+ws).hi, m)
					rc := rsRemainder(tc.hi, m)
					r2 := rsRemainder(gridT(t0, h, j0+we-1).hi, m)
					q2 = ((r0-rc)/x1 - (r2-rc)/x2) / (x1 - x2)
					q1 = (r0-rc)/x1 - q2*x1
					q0 = rc
				}
			}
			var ph, rem float64
			if exact {
				t := gridT(t0, h, j0+j)
				ph = ddMod2Pi(thetaDDt(t))
				rem = rsRemainder(t.hi, m)
			} else {
				k := float64(j - kc)
				ph = ddMod2Pi(ddAddD(ddAdd(thC, ddMulD(A, k)), B*k*k))
				rem = q0 + k*(q1+k*q2)
			}
			wi_, wr_ := math.Sincos(ph)
			out[j0+j] = 2*(wr_*real(S[j])-wi_*imag(S[j])) + rem
		}
	})
}

func zBlockSegmentRotation(t0, h float64, j0, j1, m, workers int, out []float64) {
	count := j1 - j0
	tSeg := gridT(t0, h, j0)
	if workers < 1 {
		workers = 1
	}

	Sr := make([]float64, count)
	Si := make([]float64, count)
	if m >= 1 {
		ln, rs := getTables(m)

		// Per-n state: v = phasor e^{-i t ln n}/sqrt(n) advanced in place
		// across tiles, r = per-step rotation e^{-i h ln n}. Anchor angles
		// in double-double.
		vr := make([]float64, m+1)
		vi := make([]float64, m+1)
		rr := make([]float64, m+1)
		ri := make([]float64, m+1)
		parallelChunks(m, workers, func(_, nLo, nHi int) {
			for n := nLo; n <= nHi; n++ {
				s, c := math.Sincos(-ddMod2Pi(ddMul(ln[n], tSeg)))
				vr[n], vi[n] = c*rs[n], s*rs[n]
				s, c = math.Sincos(-ddMod2Pi(ddMulD(ln[n], h)))
				rr[n], ri[n] = c, s
			}
		})

		// Accumulate S_j = sum_n v_n(j) tile-by-tile so each worker's
		// scratch stays L2-resident regardless of grid size.
		const tile = 4096
		bufR := make([][]float64, workers)
		bufI := make([][]float64, workers)
		for w := range bufR {
			bufR[w] = make([]float64, tile)
			bufI[w] = make([]float64, tile)
		}
		var mu sync.Mutex
		for ts := 0; ts < count; ts += tile {
			tl := min(tile, count-ts)
			parallelChunks(m, workers, func(w, nLo, nHi int) {
				ar := bufR[w][:tl]
				ai := bufI[w][:tl]
				for j := range ar {
					ar[j], ai[j] = 0, 0
				}
				n := nLo
				// Four independent rotation chains per pass hide the
				// multiply latency (the chains don't depend on each other).
				for ; n+3 <= nHi; n += 4 {
					v0r, v0i, r0r, r0i := vr[n], vi[n], rr[n], ri[n]
					v1r, v1i, r1r, r1i := vr[n+1], vi[n+1], rr[n+1], ri[n+1]
					v2r, v2i, r2r, r2i := vr[n+2], vi[n+2], rr[n+2], ri[n+2]
					v3r, v3i, r3r, r3i := vr[n+3], vi[n+3], rr[n+3], ri[n+3]
					for j := range ar {
						ar[j] += v0r + v1r + v2r + v3r
						ai[j] += v0i + v1i + v2i + v3i
						v0r, v0i = v0r*r0r-v0i*r0i, v0r*r0i+v0i*r0r
						v1r, v1i = v1r*r1r-v1i*r1i, v1r*r1i+v1i*r1r
						v2r, v2i = v2r*r2r-v2i*r2i, v2r*r2i+v2i*r2r
						v3r, v3i = v3r*r3r-v3i*r3i, v3r*r3i+v3i*r3r
					}
					vr[n], vi[n] = v0r, v0i
					vr[n+1], vi[n+1] = v1r, v1i
					vr[n+2], vi[n+2] = v2r, v2i
					vr[n+3], vi[n+3] = v3r, v3i
				}
				for ; n <= nHi; n++ {
					v0r, v0i, r0r, r0i := vr[n], vi[n], rr[n], ri[n]
					for j := range ar {
						ar[j] += v0r
						ai[j] += v0i
						v0r, v0i = v0r*r0r-v0i*r0i, v0r*r0i+v0i*r0r
					}
					vr[n], vi[n] = v0r, v0i
				}
				mu.Lock()
				for j := 0; j < tl; j++ {
					Sr[ts+j] += ar[j]
					Si[ts+j] += ai[j]
				}
				mu.Unlock()
			})
		}
	}

	// Combine: Z = 2*Re(e^{i theta} * S) + Riemann-Siegel remainder.
	chunk := (count + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo := w * chunk
		hi := min(count, lo+chunk)
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for j := lo; j < hi; j++ {
				t := gridT(t0, h, j0+j)
				wi_, wr_ := math.Sincos(ddMod2Pi(thetaDDt(t)))
				out[j0+j] = 2*(wr_*Sr[j]-wi_*Si[j]) + rsRemainder(t.hi, m)
			}
		}(lo, hi)
	}
	wg.Wait()
}

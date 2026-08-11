package main

import (
	"math"
	"math/cmplx"
	"runtime"
	"sort"
	"testing"
)

// Imaginary parts of the first ten non-trivial zeros (Odlyzko's tables).
var knownZeros = []float64{
	14.134725141734694,
	21.022039638771555,
	25.010857580145688,
	30.424876125859513,
	32.935061587739190,
	37.586178158825671,
	40.918719012147495,
	43.327073280914999,
	48.005150881167159,
	49.773832477672302,
}

// Exact / high-precision reference values off the line.
func TestZetaKnownValues(t *testing.T) {
	cases := []struct {
		name string
		s    complex128
		want complex128
	}{
		{"zeta(2)=pi^2/6", complex(2, 0), complex(math.Pi*math.Pi/6, 0)},
		{"zeta(4)=pi^4/90", complex(4, 0), complex(math.Pow(math.Pi, 4)/90, 0)},
		{"zeta(3)", complex(3, 0), complex(1.2020569031595943, 0)},
		{"zeta(0)=-1/2", complex(0, 0), complex(-0.5, 0)},
		{"zeta(-1)=-1/12", complex(-1, 0), complex(-1.0/12.0, 0)},
		{"zeta(1/2)", complex(0.5, 0), complex(-1.4603545088095868, 0)},
	}
	for _, c := range cases {
		got := Zeta(c.s)
		if cmplx.Abs(got-c.want) > 1e-10 {
			t.Errorf("%s: got %v, want %v (err %.3e)",
				c.name, got, c.want, cmplx.Abs(got-c.want))
		}
	}
}

// At each known zero, |zeta(1/2+it)| must vanish to within the precision
// of the tabulated t (~1e-15 relative), so ~1e-8 absolute is a safe gate.
func TestKnownZerosAreZero(t *testing.T) {
	for _, tz := range knownZeros {
		v := cmplx.Abs(Zeta(complex(0.5, tz)))
		if v > 1e-8 {
			t.Errorf("|zeta(1/2 + %vi)| = %.3e, expected ~0", tz, v)
		}
	}
}

// The same heights but OFF the critical line must be clearly non-zero:
// this is exactly the discrimination a counterexample hunt relies on.
func TestOffLineIsNotZero(t *testing.T) {
	for _, tz := range knownZeros {
		for _, sigma := range []float64{0.6, 0.75, 0.9} {
			v := cmplx.Abs(Zeta(complex(sigma, tz)))
			if v < 1e-3 {
				t.Errorf("|zeta(%v + %vi)| = %.3e, expected clearly non-zero", sigma, tz, v)
			}
		}
	}
	// And ordinary on-line points between zeros are non-zero too.
	for _, tt := range []float64{2, 10, 17.5, 23, 28} {
		v := cmplx.Abs(Zeta(complex(0.5, tt)))
		if v < 1e-3 {
			t.Errorf("|zeta(0.5 + %vi)| = %.3e, expected non-zero", tt, v)
		}
	}
}

// Two independent algorithms must agree: Riemann-Siegel (ZFast) against
// Euler-Maclaurin (Z). C0-only R-S error is O((t/2pi)^(-3/4)).
func TestFastAgreesWithAccurate(t *testing.T) {
	for _, tt := range []float64{50, 100, 250.5, 555.5, 1000, 2500.25, 5000} {
		a, f := Z(tt), ZFast(tt)
		tol := 3 * math.Pow(tt/(2*math.Pi), -0.75)
		if math.Abs(a-f) > tol {
			t.Errorf("t=%v: Z=%.10f ZFast=%.10f diff=%.3e tol=%.3e",
				tt, a, f, math.Abs(a-f), tol)
		}
	}
}

// The scanner must recover exactly the first ten zeros below t=50.
func TestFindZeros(t *testing.T) {
	got := FindZeros(0, 50)
	if len(got) != len(knownZeros) {
		t.Fatalf("found %d zeros in (0,50), want %d: %v", len(got), len(knownZeros), got)
	}
	for i, z := range got {
		if math.Abs(z-knownZeros[i]) > 1e-6 {
			t.Errorf("zero %d: got %.12f, want %.12f", i+1, z, knownZeros[i])
		}
	}
}

// Zero count from the scan must match the argument-principle estimate.
func TestCountMatchesArgumentPrinciple(t *testing.T) {
	n := float64(len(FindZeros(0, 100)))
	want := NApprox(100) // 29 zeros below t=100
	if math.Abs(n-want) > 1.5 {
		t.Errorf("found %v zeros below 100, argument principle says %.2f", n, want)
	}
}

// Double-double ln must satisfy exact identities to ~1e-29 and match
// math.Log at float64 precision.
func TestLnDD(t *testing.T) {
	d := ddSub(lnDD(6), ddAdd(lnDD(2), lnDD(3)))
	if math.Abs(d.hi) > 1e-29 {
		t.Errorf("ln6 != ln2+ln3 in dd: diff %.3e", d.hi)
	}
	d = ddSub(lnDD(1<<20), ddMulD(ddLn2, 20))
	if math.Abs(d.hi) > 1e-29 {
		t.Errorf("ln(2^20) != 20*ln2 in dd: diff %.3e", d.hi)
	}
	for _, x := range []float64{2, 3, 10, 690000, 3e12} {
		if math.Abs(lnDD(x).hi-math.Log(x)) > 4e-16*math.Log(x) {
			t.Errorf("lnDD(%v).hi = %v, math.Log = %v", x, lnDD(x).hi, math.Log(x))
		}
	}
}

// theta vanishes at the first Gram point g0 = 17.8455995405...; an
// absolute anchor for the whole dd theta machinery.
func TestThetaAtGramPoint(t *testing.T) {
	if v := Theta(17.8455995405); math.Abs(v) > 1e-5 {
		t.Errorf("theta(g0) = %.3e, want ~0", v)
	}
}

// At t = 3e12 (just above the verified frontier) a plain float64 phase
// would be wrong by ~0.008 rad; the dd phase must still deliver a zero
// count matching the argument principle.
func TestHighHeightCount(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a 690k-entry table")
	}
	t0 := 3.0e12
	t1 := t0 + 2
	exp := nDiff(t0, t1) // about 8.5 zeros expected in a window of 2
	mids, _ := scanBlock(t0, t1, 80, 4)
	if math.Abs(float64(len(mids))-exp) > 2 {
		t.Errorf("found %d zeros in (3e12, 3e12+2), argument principle says %.2f", len(mids), exp)
	}
}

// The Lehmer pair near t=7005 (zeros at 7005.0629 and 7005.1006, gap
// ~0.038 vs mean spacing ~0.9) is the classic close pair that a
// default-density scan can straddle. A 16x rescan must recover the full
// argument-principle count -- this is what the hunt's escalation and
// backscan rely on.
func TestLehmerPairRecovered(t *testing.T) {
	exp := nDiff(7000, 7010) // ~11.2
	coarse, _ := scanBlock(7000, 7010, 52, 4)
	fine, _ := scanBlock(7000, 7010, 52*16, 4)
	if len(fine) < len(coarse) {
		t.Errorf("finer scan found fewer zeros (%d < %d)", len(fine), len(coarse))
	}
	if float64(len(fine))-exp <= -1.5 {
		t.Errorf("16x scan found %d zeros in (7000,7010), argument principle says %.2f", len(fine), exp)
	}
}

// The batched grid evaluator must agree with the per-point ZFast,
// including across a main-sum-length crossing (t = 2pi*k^2), where the
// grid is split into segments with different term counts.
func TestZBlockMatchesZFast(t *testing.T) {
	cases := []struct {
		t0, h float64
		n     int
	}{
		{5000, 0.13, 64},
		{5650, 0.11, 100}, // crosses t = 2pi*30^2 = 5654.87 mid-grid
		{100000.25, 0.07, 50},
	}
	for _, c := range cases {
		zs := ZBlock(c.t0, c.h, c.n, 3)
		for j := 0; j < c.n; j++ {
			tj := c.t0 + float64(j)*c.h
			want := ZFast(tj)
			if math.Abs(zs[j]-want) > 1e-9 {
				t.Errorf("t=%.4f: ZBlock=%.12f ZFast=%.12f diff=%.3e",
					tj, zs[j], want, math.Abs(zs[j]-want))
			}
		}
	}
}

// Same agreement at the frontier height, where the anchors are dd and
// the rotation trick must not lose the phase.
func TestZBlockHighHeight(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a 690k-entry table")
	}
	// h = 0.125 is a power of two, so every t_j is exactly representable
	// and the per-point ZFast evaluates the identical t.
	zs := ZBlock(3e12, 0.125, 8, 4)
	for j := range zs {
		tj := 3e12 + float64(j)*0.125
		want := ZFast(tj)
		if math.Abs(zs[j]-want) > 1e-8 {
			t.Errorf("t=%.1f: ZBlock=%.12f ZFast=%.12f diff=%.3e",
				tj, zs[j], want, math.Abs(zs[j]-want))
		}
	}
}

// Grids of >= 2048 points take the NUFFT (Odlyzko-Schonhage) path; it
// must agree with the independent per-point ZFast. Dyadic h keeps every
// grid point exactly representable so both evaluate the identical t.
func TestZBlockFFTPath(t *testing.T) {
	const h = 1.0 / 512
	const count = 4096
	zs := ZBlock(100000, h, count, 4)
	for _, j := range []int{0, 1, 7, 100, 511, 1023, 2048, 3000, 4095} {
		tj := 100000 + float64(j)*h
		want := ZFast(tj)
		if math.Abs(zs[j]-want) > 1e-6 {
			t.Errorf("t=%.6f: ZBlock(fft)=%.10f ZFast=%.10f diff=%.3e",
				tj, zs[j], want, math.Abs(zs[j]-want))
		}
	}
}

func TestZBlockFFTHighHeight(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a 690k-entry table")
	}
	const h = 1.0 / 1024
	const count = 4096
	zs := ZBlock(3e12, h, count, 4)
	for _, j := range []int{0, 13, 512, 1024, 2047, 3333, 4095} {
		tj := 3e12 + float64(j)*h
		want := ZFast(tj)
		if math.Abs(zs[j]-want) > 1e-6 {
			t.Errorf("t=%.6f: ZBlock(fft)=%.10f ZFast=%.10f diff=%.3e",
				tj, zs[j], want, math.Abs(zs[j]-want))
		}
	}
}

// The deficit localizer at real hunt scale: scan a 3000-unit stretch at
// the frontier (~12.8k zeros), then delete an adjacent pair of found
// zeros to simulate a straddled close pair. localizeDeficit must flag a
// window covering exactly that spot, and small blocks (where windows
// would hold too few zeros to overcome the S wobble) must return nil so
// the ladder falls back to full-block rescans.
func TestLocalizeDeficit(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a 690k-entry table")
	}
	t0, t1 := 3.0e12, 3.0e12+3000
	mids, _ := scanBlock(t0, t1, 3300000, 4) // 256 pts/spacing: no straddles
	if d := float64(len(mids)) - nDiff(t0, t1); math.Abs(d) > 2.2 {
		t.Fatalf("reference scan off by %+.3f", d)
	}
	// Delete four consecutive zeros (two "hidden pairs" in one spot): a
	// -4 window deficit cannot be masked by the ~±0.8 window wobble.
	victim := len(mids) / 3
	gone := mids[victim]
	cut := append(append([]float64{}, mids[:victim]...), mids[victim+4:]...)
	wins := localizeDeficit(t0, t1, cut)
	if wins == nil {
		t.Fatal("localizeDeficit flagged nothing despite a missing pair")
	}
	covers := false
	span := 0.0
	for _, w := range wins {
		span += w[1] - w[0]
		if w[0] < gone && gone < w[1] {
			covers = true
		}
	}
	if !covers {
		t.Errorf("no flagged window covers the deleted pair at %.3f: %v", gone, wins)
	}
	if span > 0.5*(t1-t0) {
		t.Errorf("windows cover %.0f of %.0f units -- localization too loose", span, t1-t0)
	}
	// Restoring via mergeReplace must reproduce the original list.
	for _, w := range wins {
		lo := sort.SearchFloat64s(mids, w[0])
		hi := sort.SearchFloat64s(mids, w[1])
		cut = mergeReplace(cut, w[0], w[1], mids[lo:hi])
	}
	if len(cut) != len(mids) {
		t.Errorf("mergeReplace restore: %d zeros, want %d", len(cut), len(mids))
	}
	// Tiny blocks: refuse to localize.
	if w := localizeDeficit(7000, 7010, mids[:9]); w != nil {
		t.Errorf("localizeDeficit should return nil at tiny scale, got %v", w)
	}
}

// Band-limited interpolation must reproduce the per-point evaluator:
// one NUFFT pass at ~1.25 samples per zero spacing, then arbitrary-t
// reconstruction through the tapered-sinc kernel.
func TestInterpMatchesZFast(t *testing.T) {
	for _, base := range []float64{100000, 3e12} {
		if base > 1e6 && testing.Short() {
			continue
		}
		bg, _ := buildBlockGrid(base, base+40, 4)
		for i := 0; i < 60; i++ {
			tt := base + 40*float64(i*i%61)/61 // deterministic scatter
			si := bg.segFor(tt)
			want := ZFast(tt)
			if got := si.zAt(tt); math.Abs(got-want) > 1e-6 {
				t.Errorf("base=%g t=+%.4f: interp=%.10f ZFast=%.10f diff=%.3e",
					base, tt-base, got, want, math.Abs(got-want))
			}
		}
	}
}

// The interpolated bulk scan must find the same zeros as the direct
// NUFFT scan over a frontier stretch.
func TestScanZMatchesDirect(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a 690k-entry table")
	}
	a, b := 3.0e12, 3.0e12+500
	h := math.Ldexp(1, -8) // 2^-8: ~60 points per spacing, dyadic
	interp, _ := scanRange(a, b, h, 4)
	direct, _ := scanBlock(a, b, int((b-a)/h)+1, 4)
	if len(interp) != len(direct) {
		t.Fatalf("interp scan found %d zeros, direct found %d", len(interp), len(direct))
	}
	for i := range interp {
		if math.Abs(interp[i]-direct[i]) > 3*h {
			t.Errorf("zero %d: interp %.6f vs direct %.6f", i, interp[i], direct[i])
		}
	}
}

// The parallel block scanner must agree with the known zero list.
func TestScanBlockMatchesKnown(t *testing.T) {
	mids, _ := scanBlock(10, 50, 400, 4)
	if len(mids) != len(knownZeros) {
		t.Fatalf("scanBlock found %d zeros in (10,50), want %d", len(mids), len(knownZeros))
	}
	for i, z := range mids {
		if math.Abs(z-knownZeros[i]) > 0.1 {
			t.Errorf("zero %d: midpoint %.3f, want near %.3f", i+1, z, knownZeros[i])
		}
	}
}

func BenchmarkZetaEM_t100(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Zeta(complex(0.5, 100))
	}
}

func BenchmarkZetaEM_t10000(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Zeta(complex(0.5, 10000))
	}
}

func BenchmarkZFast_t10000(b *testing.B) {
	for i := 0; i < b.N; i++ {
		ZFast(10000)
	}
}

func BenchmarkZFast_t1e8(b *testing.B) {
	for i := 0; i < b.N; i++ {
		ZFast(1e8)
	}
}

func BenchmarkZFast_t3e12(b *testing.B) {
	getTables(int(math.Sqrt(3e12 / (2 * math.Pi))))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ZFast(3e12)
	}
}

// The batched evaluator: divide ns/op by 1024 for the per-point cost,
// comparable against BenchmarkZFast_t3e12.
func BenchmarkZBlock1024_t3e12(b *testing.B) {
	getTables(int(math.Sqrt(3e12/(2*math.Pi))) + 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ZBlock(3e12, 0.0588, 1024, runtime.NumCPU())
	}
}

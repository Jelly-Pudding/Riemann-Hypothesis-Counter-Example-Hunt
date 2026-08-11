package main

import "math"

// Double-double arithmetic: a value represented as an unevaluated sum
// hi + lo of two float64s, giving ~32 significant digits. Needed because
// the Riemann-Siegel phase theta(t) - t*ln(n) is ~4e13 at t = 3e12, where
// a lone float64 carries an absolute error of ~0.008 rad -- fatal for
// cos(). With double-double the phase error stays below 1e-18 rad for
// heights past 1e20.

type dd struct{ hi, lo float64 }

func twoSum(a, b float64) dd {
	s := a + b
	bb := s - a
	return dd{s, (a - (s - bb)) + (b - bb)}
}

// quickTwoSum requires |a| >= |b|.
func quickTwoSum(a, b float64) dd {
	s := a + b
	return dd{s, b - (s - a)}
}

func twoProd(a, b float64) dd {
	p := a * b
	return dd{p, math.FMA(a, b, -p)}
}

func ddAdd(a, b dd) dd {
	s := twoSum(a.hi, b.hi)
	return quickTwoSum(s.hi, s.lo+a.lo+b.lo)
}

func ddAddD(a dd, b float64) dd {
	s := twoSum(a.hi, b)
	return quickTwoSum(s.hi, s.lo+a.lo)
}

func ddSub(a, b dd) dd { return ddAdd(a, dd{-b.hi, -b.lo}) }

func ddMul(a, b dd) dd {
	p := twoProd(a.hi, b.hi)
	return quickTwoSum(p.hi, p.lo+a.hi*b.lo+a.lo*b.hi)
}

func ddMulD(a dd, b float64) dd {
	p := twoProd(a.hi, b)
	return quickTwoSum(p.hi, p.lo+a.lo*b)
}

func ddDivD(a dd, b float64) dd {
	q1 := a.hi / b
	p := twoProd(q1, b)
	r := (a.hi - p.hi) - p.lo + a.lo
	q2 := r / b
	return quickTwoSum(q1, q2)
}

func ddDiv(a, b dd) dd {
	q1 := a.hi / b.hi
	r := ddSub(a, ddMulD(b, q1))
	q2 := r.hi / b.hi
	r = ddSub(r, ddMulD(b, q2))
	q3 := r.hi / b.hi
	return ddAddD(quickTwoSum(q1, q2), q3)
}

// pi and 2*pi to double-double precision (the lo parts are the canonical
// residuals of the float64 roundings).
var (
	ddPi  = dd{3.141592653589793, 1.2246467991473532e-16}
	dd2Pi = dd{6.283185307179586, 2.4492935982947064e-16}
)

// ln(2) and ln(2*pi) are derived at startup from the atanh series rather
// than hardcoded, so there is no hand-typed constant to get wrong.
var ddLn2, ddLn2Pi dd

func init() {
	// ln 2 = 2*atanh(1/3)
	at := atanhDD(ddDivD(dd{1, 0}, 3))
	ddLn2 = dd{2 * at.hi, 2 * at.lo}
	// ln(2*pi) = ln 2 + ln(pi), with the dd-precision pi:
	// ln(pi.hi + pi.lo) = ln(pi.hi) + pi.lo/pi.hi to well past 32 digits.
	ddLn2Pi = ddAdd(ddLn2, ddAddD(lnDD(ddPi.hi), ddPi.lo/ddPi.hi))
}

func atanhDD(z dd) dd {
	z2 := ddMul(z, z)
	term := z
	sum := z
	for k := 3; k <= 75; k += 2 {
		term = ddMul(term, z2)
		sum = ddAdd(sum, ddDivD(term, float64(k)))
		if math.Abs(term.hi) < 1e-34*math.Abs(sum.hi) {
			break
		}
	}
	return sum
}

// lnDD computes ln(x) of a positive float64 to double-double precision:
// x = f * 2^e with f in [sqrt(2)/2, sqrt(2)), then
// ln x = e*ln2 + 2*atanh((f-1)/(f+1)).
func lnDD(x float64) dd {
	f, e := math.Frexp(x)
	if f < math.Sqrt2/2 {
		f *= 2
		e--
	}
	z := ddDiv(dd{f - 1, 0}, twoSum(f, 1))
	at := atanhDD(z)
	return ddAdd(ddMulD(ddLn2, float64(e)), dd{2 * at.hi, 2 * at.lo})
}

// lnDDdd extends lnDD to a double-double argument:
// ln(hi+lo) = ln(hi) + lo/hi to well past 32 digits when |lo| <= ulp(hi).
func lnDDdd(a dd) dd { return ddAddD(lnDD(a.hi), a.lo/a.hi) }

// thetaDD is the Riemann-Siegel theta function in double-double,
// theta(t) = (t/2)ln(t/2pi) - t/2 - pi/8 + 1/(48t) + 7/(5760 t^3).
// The asymptotic tail is far below dd precision for t >= 10.
func thetaDD(t float64) dd { return thetaDDt(dd{t, 0}) }

// thetaDDt is thetaDD for a double-double argument, needed by ZBlock
// where grid points t0 + j*h are generally not representable as a
// single float64.
func thetaDDt(t dd) dd {
	l := ddSub(lnDDdd(t), ddLn2Pi)
	half := dd{t.hi / 2, t.lo / 2}
	th := ddMul(l, half)
	th = ddSub(th, half)
	th = ddAdd(th, dd{-ddPi.hi / 8, -ddPi.lo / 8})
	return ddAddD(th, 1/(48*t.hi)+7/(5760*t.hi*t.hi*t.hi))
}

// ddMod2Pi reduces a double-double angle into (-pi-eps, pi+eps] and
// collapses it to a float64, which is now safe to pass to math.Cos.
func ddMod2Pi(a dd) float64 {
	k := math.Round(a.hi / dd2Pi.hi)
	if k == 0 {
		return a.hi + a.lo
	}
	r := ddSub(a, ddMulD(dd2Pi, k))
	return r.hi + r.lo
}

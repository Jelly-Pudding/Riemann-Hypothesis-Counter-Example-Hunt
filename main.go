package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"os"
	"runtime"
	"strconv"
)

func usage() {
	fmt.Fprintln(os.Stderr, `usage:
  riemann eval <re> <im>     evaluate zeta(re + im*i)          (|im| <= 1e7)
  riemann z <t>              Hardy Z(t): accurate + fast values
  riemann zeros <t0> <t1>    find critical-line zeros with t in (t0, t1)
  riemann check <t0> <t1>    compare zeros found on the line against the
                             argument-principle count; a deficit of 2 that
                             survives rescans means off-line zeros (RH false)
  riemann hunt <start> [-end N -block W -workers N -stepdiv D -summary S
                        -state F -log F -anomalylog F -zeroslog F]
                             run the check continuously from <start>,
                             logging every block and resuming from the
                             state file after restarts`)
	os.Exit(2)
}

func arg(i int) float64 {
	v, err := strconv.ParseFloat(os.Args[i], 64)
	if err != nil {
		fmt.Fprintf(os.Stderr, "bad number %q\n", os.Args[i])
		os.Exit(2)
	}
	return v
}

func main() {
	if len(os.Args) < 3 {
		usage()
	}
	switch os.Args[1] {
	case "eval":
		if len(os.Args) != 4 {
			usage()
		}
		s := complex(arg(2), arg(3))
		if math.Abs(imag(s)) > 1e7 {
			fmt.Fprintln(os.Stderr, "eval uses the O(t) Euler-Maclaurin method; above im=1e7 use `z` or `hunt`")
			os.Exit(2)
		}
		z := Zeta(s)
		fmt.Printf("zeta(%g%+gi) = %.15g %+.15gi   |zeta| = %.6e\n",
			real(s), imag(s), real(z), imag(z), cmplx.Abs(z))
	case "z":
		if len(os.Args) != 3 {
			usage()
		}
		t := arg(2)
		fmt.Printf("Zfast(%g) = %.15g  (Riemann-Siegel, dd phase)\n", t, ZFast(t))
		if t <= 1e6 {
			fmt.Printf("Z(%g)     = %.15g  (accurate, Euler-Maclaurin)\n", t, Z(t))
		} else {
			fmt.Println("(Euler-Maclaurin cross-check skipped above t=1e6: O(t) cost)")
		}
	case "zeros":
		if len(os.Args) != 4 {
			usage()
		}
		zs := FindZeros(arg(2), arg(3))
		for i, z := range zs {
			if z <= 1e6 {
				fmt.Printf("%4d  t = %.12f   |zeta(1/2+it)| = %.3e\n",
					i+1, z, cmplx.Abs(Zeta(complex(0.5, z))))
			} else {
				fmt.Printf("%4d  t = %.6f (sign-change midpoint)\n", i+1, z)
			}
		}
		fmt.Printf("found %d zeros\n", len(zs))
	case "check":
		if len(os.Args) != 4 {
			usage()
		}
		t0, t1 := math.Max(arg(2), 10), arg(3)
		expected := nDiff(t0, t1)
		var zs []float64
		if t1 > 1e7 {
			// High heights: interpolation engine, escalating density
			// until the count settles.
			spacing := 2 * math.Pi / math.Log(((t0+t1)/2)/(2*math.Pi))
			h := math.Ldexp(1, int(math.Round(math.Log2(spacing/64))))
			bg, _ := buildBlockGrid(t0, t1, runtime.NumCPU())
			for _, mult := range []float64{1, 8, 64, 512} {
				// The u-space lattice has no ulp floor: every rung runs on
				// the interpolation engine at exact index arithmetic.
				zs, _ = bg.scanZ(t0, t1, h/mult, runtime.NumCPU())
				fmt.Printf("  density %4.0fx base (interp): %d sign changes (expected %.3f)\n",
					mult, len(zs), expected)
				if float64(len(zs))-expected > -1.5 {
					break
				}
			}
			if float64(len(zs))-expected <= -1.5 && t1-t0 <= 2000 {
				// Independent confirmation: the direct dd-grid engine
				// shares no code path with the interpolation above.
				zs2, _ := scanBlockChunked(t0, t1, h/64, runtime.NumCPU())
				fmt.Printf("  direct dd-grid at 64x: %d sign changes\n", len(zs2))
				if len(zs2) > len(zs) {
					zs = zs2
				}
			}
		} else {
			zs = FindZeros(t0, t1)
		}
		fmt.Printf("zeros found on the critical line in (%g, %g): %d\n", t0, t1, len(zs))
		fmt.Printf("argument-principle count (up to |S(T)|<~2 wobble): %.3f\n", expected)
		diff := expected - float64(len(zs))
		switch {
		case math.Abs(diff) < 2:
			fmt.Println("counts agree (within normal S(T) fluctuation) -> all zeros here are on the line")
		default:
			fmt.Printf("count deficit %.1f survived escalation -- verify with independent tools\n", diff)
			fmt.Println("(mpmath spot checks, other hardware) before getting excited")
		}
	case "hunt":
		runHunt(os.Args[2:])
	default:
		usage()
	}
}

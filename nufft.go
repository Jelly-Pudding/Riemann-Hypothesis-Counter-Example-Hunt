package main

import "math"

// nufft1 is a type-1 NUFFT with Gaussian gridding:
//
//	f[j] = sum_n c[n] * exp(i * (j - R/2) * x[n]),   j = 0..R-1
//
// for arbitrary real frequencies x[n] (treated mod 2pi). Sources are
// spread onto a 2x-oversampled length-Mr grid with a truncated Gaussian
// of half-width Msp, one FFT evaluates all modes, and dividing by the
// Gaussian transform undoes the smoothing.
//
// Cost O(m*Msp + Mr log Mr). With Msp = 14 and tau = sqrt(2)*pi*Msp/Mr^2
// truncation and aliasing balance at ~3e-14 relative to sum|c|.
//
// Spreading is parallelized by bucketing sources by grid cell and
// sweeping even then odd buckets. Same-parity buckets cannot overlap
// because the bucket width exceeds twice the kernel half-width.
func nufft1(x []float64, c []complex128, R, workers int) []complex128 {
	const Msp = 14
	Mr := nextPow2(2 * R)
	if Mr < 4*Msp {
		Mr = 4 * Msp
	}
	tau := math.Sqrt2 * math.Pi * float64(Msp) / (float64(Mr) * float64(Mr))
	delta := 2 * math.Pi / float64(Mr)
	mask := Mr - 1
	m := len(x)

	// E3[l] = exp(-(l*delta)^2 / 4tau), source independent.
	var E3 [Msp + 1]float64
	for l := 0; l <= Msp; l++ {
		d := float64(l) * delta
		E3[l] = math.Exp(-d * d / (4 * tau))
	}

	spreadOne := func(w []complex128, i0 int, d float64, cc complex128) {
		E1 := math.Exp(-d * d / (4 * tau))
		E2 := math.Exp(d * delta / (2 * tau))
		w[i0&mask] += cc * complex(E1, 0)
		p, q := E1, E1
		E2inv := 1 / E2
		for l := 1; l <= Msp; l++ {
			p *= E2
			q *= E2inv
			w[(i0+l)&mask] += cc * complex(p*E3[l], 0)
			w[(i0-l)&mask] += cc * complex(q*E3[l], 0)
		}
	}

	w := make([]complex128, Mr)
	const nb = 64
	if workers <= 1 || m < 8192 || Mr < 4*nb*Msp {
		for n := range x {
			u := x[n] / delta
			i0 := int(math.Round(u))
			spreadOne(w, i0, (u-float64(i0))*delta, c[n])
		}
	} else {
		// Bucket width Mr/nb >= 4*Msp so same-parity buckets write to
		// disjoint cell ranges.
		shift := 0
		for 1<<shift < Mr/nb {
			shift++
		}
		i0s := make([]int32, m)
		ds := make([]float64, m)
		parallelRange(m, workers, func(lo, hi int) {
			for n := lo; n < hi; n++ {
				u := x[n] / delta
				i0 := int(math.Round(u))
				i0s[n] = int32(i0 & mask)
				ds[n] = (u - float64(i0)) * delta
			}
		})
		var cnt [nb + 1]int32
		for n := 0; n < m; n++ {
			cnt[(i0s[n]>>shift)+1]++
		}
		for b := 1; b <= nb; b++ {
			cnt[b] += cnt[b-1]
		}
		ord := make([]int32, m)
		var fill [nb]int32
		for n := 0; n < m; n++ {
			b := i0s[n] >> shift
			ord[cnt[b]+fill[b]] = int32(n)
			fill[b]++
		}
		for parity := 0; parity < 2; parity++ {
			parallelRange(nb/2, workers, func(lo, hi int) {
				for bi := lo; bi < hi; bi++ {
					b := 2*bi + parity
					for _, n := range ord[cnt[b]:cnt[b+1]] {
						spreadOne(w, int(i0s[n]), ds[n], c[n])
					}
				}
			})
		}
	}

	w = fft(w, workers)

	// f_k = delta * W[(-k) mod Mr] / psiHat(k), psiHat(k) = 2*sqrt(pi*tau)*exp(-k^2*tau).
	out := make([]complex128, R)
	base := delta / (2 * math.Sqrt(math.Pi*tau))
	parallelRange(R, workers, func(lo, hi int) {
		for j := lo; j < hi; j++ {
			k := j - R/2
			fk := float64(k)
			out[j] = w[(-k)&mask] * complex(base*math.Exp(fk*fk*tau), 0)
		}
	})
	return out
}

func nextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

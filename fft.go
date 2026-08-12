package main

import (
	"math"
	"math/bits"
	"sync"
)

// Iterative radix-2 complex FFT, forward transform:
// W[k] = sum_m a[m] * exp(-2*pi*i*k*m/n). n must be a power of 2.
// Bit reversal is a parallel gather into a scratch buffer. Early stages
// parallelize across sub-transforms, late stages across butterflies.

var (
	twMu    sync.Mutex
	twCache = map[int][]complex128{} // per-size twiddle tables, few entries

	scratchMu    sync.Mutex
	scratchCache = map[int][]complex128{} // one reusable buffer per size
)

func fftTwiddles(n, workers int) []complex128 {
	twMu.Lock()
	defer twMu.Unlock()
	if tw, ok := twCache[n]; ok {
		return tw
	}
	tw := make([]complex128, n/2)
	parallelRange(n/2, workers, func(lo, hi int) {
		for j := lo; j < hi; j++ {
			s, c := math.Sincos(-2 * math.Pi * float64(j) / float64(n))
			tw[j] = complex(c, s)
		}
	})
	if len(twCache) >= 4 {
		twCache = map[int][]complex128{}
	}
	twCache[n] = tw
	return tw
}

func getScratch(n int) []complex128 {
	scratchMu.Lock()
	defer scratchMu.Unlock()
	if s, ok := scratchCache[n]; ok {
		delete(scratchCache, n)
		return s
	}
	return make([]complex128, n)
}

func putScratch(s []complex128) {
	scratchMu.Lock()
	defer scratchMu.Unlock()
	if len(scratchCache) >= 4 {
		scratchCache = map[int][]complex128{}
	}
	scratchCache[len(s)] = s
}

// fft returns the transform in a new slice. The input slice is recycled
// as scratch for the next call.
func fft(a []complex128, workers int) []complex128 {
	n := len(a)
	if n&(n-1) != 0 {
		panic("fft: size must be a power of 2")
	}
	lg := bits.TrailingZeros(uint(n))
	b := getScratch(n)
	parallelRange(n, workers, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			b[i] = a[int(bits.Reverse64(uint64(i))>>(64-lg))]
		}
	})
	putScratch(a)

	tw := fftTwiddles(n, workers)
	for length := 2; length <= n; length <<= 1 {
		half := length >> 1
		step := n / length
		blocks := n / length
		butterfly := func(base, kLo, kHi int) {
			ti := kLo * step
			for k := kLo; k < kHi; k++ {
				u := b[base+k]
				v := b[base+k+half] * tw[ti]
				b[base+k] = u + v
				b[base+k+half] = u - v
				ti += step
			}
		}
		switch {
		case workers <= 1 || n < 1<<14:
			for blk := 0; blk < blocks; blk++ {
				butterfly(blk*length, 0, half)
			}
		case blocks >= workers:
			parallelRange(blocks, workers, func(lo, hi int) {
				for blk := lo; blk < hi; blk++ {
					butterfly(blk*length, 0, half)
				}
			})
		default:
			for blk := 0; blk < blocks; blk++ {
				base := blk * length
				parallelRange(half, workers, func(lo, hi int) {
					butterfly(base, lo, hi)
				})
			}
		}
	}
	return b
}

// parallelRange splits [0, n) into contiguous chunks across workers.
// Runs inline for workers <= 1 or small n.
func parallelRange(n, workers int, fn func(lo, hi int)) {
	if workers <= 1 || n < 1024 {
		fn(0, n)
		return
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := min(n, lo+chunk)
		wg.Add(1)
		go func(lo, hi int) { defer wg.Done(); fn(lo, hi) }(lo, hi)
	}
	wg.Wait()
}

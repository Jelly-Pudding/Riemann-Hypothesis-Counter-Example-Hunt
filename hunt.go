package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
)

// huntState is persisted to disk after every block so a killed process
// resumes where it left off (losing at most the block in flight).
type huntState struct {
	StartT     float64    `json:"start_t"`
	NextT      float64    `json:"next_t"`
	ZerosFound int64      `json:"zeros_found"`
	Blocks     int64      `json:"blocks"`
	Evals      int64      `json:"evals"`
	Seconds    float64    `json:"scan_seconds"`
	Anomalies  int64      `json:"anomalies"`
	AckDev     float64    `json:"ack_dev"`
	CumHistory []float64  `json:"cum_history,omitempty"`
	Hist       []blockRec `json:"hist,omitempty"`
	UpdatedAt  time.Time  `json:"updated_at"`
}

func saveState(path string, s *huntState) {
	b, _ := json.MarshalIndent(s, "", "  ")
	tmp := path + ".tmp"
	err := os.WriteFile(tmp, b, 0644)
	if err == nil {
		err = os.Rename(tmp, path)
	}
	if err != nil {
		// Disk trouble must be loud: without state the hunt cannot resume.
		fmt.Fprintf(os.Stderr, "WARNING: cannot save state to %s: %v\n", path, err)
	}
}

// blockRec remembers a recently scanned block so a later cumulative
// deficit can trigger a rescan of the block that actually lost the pair
// (often not the block where the deficit finally trips). Persisted in
// the state file so backscans survive restarts.
type blockRec struct {
	T0    float64 `json:"t0"`
	T1    float64 `json:"t1"`
	HBase float64 `json:"h_base"` // base lattice spacing the block was scanned at
	Found int     `json:"found"`
	Mult  int     `json:"mult"` // finest full-block density multiplier tried
}

// baseline is the rolling median of recent cumulative drifts. The drift
// carries an unknown constant offset S(t_start) from wherever the hunt
// began; an off-line pair shows up as a persistent STEP of -2 against
// this baseline, which is what we alarm on.
func baseline(h []float64) float64 {
	if len(h) < 20 {
		return 0
	}
	c := append([]float64(nil), h...)
	sort.Float64s(c)
	return c[len(c)/2]
}

// localizeDeficit finds sub-windows of (t0, t1) whose zero count falls
// short of the argument-principle count. A hidden pair shows as ~-2 in
// the staggered window containing it. Windows are merged and snapped
// outward to midpoints between found zeros (sign-stable boundaries).
// thresh is the flag level: -1.2 catches an unmasked pair, liberal
// values like -0.7 also catch wobble-masked pairs at the cost of extra
// wobble windows.
func localizeDeficit(t0, t1 float64, mids []float64, thresh float64) [][2]float64 {
	// Window size targets ~16k zeros: large enough that the S(T) wobble
	// (order 1) cannot mask a deficit of 2, small enough to pin a pair
	// to a thin slice of the block.
	nsub := len(mids) / 16000
	if nsub < 16 {
		nsub = 16
	}
	if nsub > 512 {
		nsub = 512
	}
	if len(mids)/nsub < 64 {
		return nil
	}
	w := (t1 - t0) / float64(nsub)
	var flagged [][2]float64
	for i := 0; i <= 2*(nsub-1); i++ {
		a := t0 + float64(i)*w/2
		b := math.Min(a+w, t1)
		lo := sort.SearchFloat64s(mids, a)
		hi := sort.SearchFloat64s(mids, b)
		if float64(hi-lo)-nDiff(a, b) <= thresh {
			flagged = append(flagged, [2]float64{a, b})
		}
	}
	if len(flagged) == 0 {
		return nil
	}
	var merged [][2]float64
	cur := flagged[0]
	for _, f := range flagged[1:] {
		if f[0] <= cur[1] {
			cur[1] = math.Max(cur[1], f[1])
		} else {
			merged = append(merged, cur)
			cur = f
		}
	}
	merged = append(merged, cur)
	for i := range merged {
		merged[i][0] = snapToGapMid(mids, merged[i][0], t0, true)
		merged[i][1] = snapToGapMid(mids, merged[i][1], t1, false)
	}
	return merged
}

// snapToGapMid moves x outward to the midpoint of a gap between found
// zeros (or to the block edge), so a rescan boundary never sits close
// to a real zero and count attribution stays unambiguous.
func snapToGapMid(mids []float64, x, edge float64, left bool) float64 {
	i := sort.SearchFloat64s(mids, x)
	if left {
		if i > len(mids)-1 {
			i = len(mids) - 1
		}
		for ; i >= 1; i-- {
			if gm := (mids[i-1] + mids[i]) / 2; gm <= x {
				return gm
			}
		}
		return edge
	}
	for ; i < len(mids); i++ {
		if i >= 1 {
			if gm := (mids[i-1] + mids[i]) / 2; gm >= x {
				return gm
			}
		}
	}
	return edge
}

// mergeReplace swaps the found zeros inside [a, b] for a rescan's finding.
func mergeReplace(mids []float64, a, b float64, repl []float64) []float64 {
	lo := sort.SearchFloat64s(mids, a)
	hi := sort.SearchFloat64s(mids, b)
	out := make([]float64, 0, lo+len(repl)+len(mids)-hi)
	out = append(out, mids[:lo]...)
	out = append(out, repl...)
	out = append(out, mids[hi:]...)
	return out
}

// scanBlockChunked runs the direct (non-interpolated) NUFFT scanner over
// [a, b] at step ~h, in sub-ranges bounded to ~8M points to cap memory.
// The direct engine's double-double grid is exact at any step size, so
// this is the tool for densities below one ulp of t, where the
// interpolation lattice cannot exist.
func scanBlockChunked(a, b, h float64, workers int) ([]float64, int64) {
	pts := int((b-a)/h) + 2
	const maxPts = 8 << 20
	if pts <= maxPts {
		return scanBlock(a, b, pts, workers)
	}
	nSub := (pts + maxPts - 1) / maxPts
	var mids []float64
	var evals int64
	for i := 0; i < nSub; i++ {
		sa := a + (b-a)*float64(i)/float64(nSub)
		sb := a + (b-a)*float64(i+1)/float64(nSub)
		m, e := scanBlock(sa, sb, pts/nSub+2, workers)
		mids = append(mids, m...)
		evals += e
	}
	return mids, evals
}

// scanBlock evaluates Z on a uniform grid of points+1 samples over
// [t0, t1] via the batched ZBlock evaluator and returns the midpoints
// of the sign-change intervals plus the number of evaluations.
func scanBlock(t0, t1 float64, points, workers int) ([]float64, int64) {
	if points < 2 {
		points = 2
	}
	h := (t1 - t0) / float64(points)
	zs := ZBlock(t0, h, points+1, workers)
	var mids []float64
	for j := 1; j < len(zs); j++ {
		if (zs[j-1] < 0) != (zs[j] < 0) {
			mids = append(mids, t0+(float64(j)-0.5)*h)
		}
	}
	return mids, int64(points + 1)
}

func runHunt(args []string) {
	fs := flag.NewFlagSet("hunt", flag.ExitOnError)
	block := fs.Float64("block", 1000000, "height covered per block (one log line each)")
	end := fs.Float64("end", 0, "stop at this height (0 = run forever)")
	workers := fs.Int("workers", runtime.NumCPU(), "parallel scan workers")
	// Close pairs follow GUE repulsion, P(gap < x*mean) ~ 0.27x^3. A base
	// scan at s points per spacing straddles ~N*0.27/s^3 pairs, which the
	// rescan ladder recovers.
	stepdiv := fs.Float64("stepdiv", 64, "scan points per mean zero spacing")
	statePath := fs.String("state", "hunt.state.json", "state file for resume")
	logPath := fs.String("log", "hunt.log", "append-only log file (every block)")
	anomPath := fs.String("anomalylog", "hunt.anomalies.log", "log file for anomalies only")
	zerosPath := fs.String("zeroslog", "", "optional file to append zero locations to")
	summarySec := fs.Float64("summary", 900, "seconds between SUMMARY progress lines (0 = off)")

	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "usage: riemann hunt <start-height> [flags]")
		fs.PrintDefaults()
		os.Exit(2)
	}
	start, err := strconv.ParseFloat(args[0], 64)
	if err != nil || start < 100 {
		fmt.Fprintln(os.Stderr, "start height must be a number >= 100")
		os.Exit(2)
	}
	fs.Parse(args[1:])
	if start > 1e14 || *end > 1e14 {
		// Above t ~ 1e14 the float64 ULP (>= 0.0156) approaches the scan
		// step (~0.05), so grid points quantize together and the density
		// guarantee silently erodes. Refuse rather than mis-scan; going
		// higher needs the grid itself carried in double-double.
		fmt.Fprintln(os.Stderr, "heights above 1e14 are not supported yet: the float64 scan grid loses resolution there")
		os.Exit(2)
	}

	st := &huntState{StartT: start, NextT: start}
	if b, err := os.ReadFile(*statePath); err == nil {
		var prev huntState
		if json.Unmarshal(b, &prev) == nil {
			if prev.StartT != start {
				fmt.Fprintf(os.Stderr,
					"state file %s belongs to a hunt starting at %g (you asked for %g);\n"+
						"delete it or pass a different -state file\n",
					*statePath, prev.StartT, start)
				os.Exit(2)
			}
			st = &prev
		}
	}

	openLog := func(path string) *os.File {
		f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return f
	}
	logF := openLog(*logPath)
	defer logF.Close()
	anomF := openLog(*anomPath)
	defer anomF.Close()
	var zerosF *os.File
	if *zerosPath != "" {
		zerosF = openLog(*zerosPath)
		defer zerosF.Close()
	}
	logf := func(format string, a ...any) {
		line := time.Now().UTC().Format(time.RFC3339) + " " + fmt.Sprintf(format, a...)
		fmt.Println(line)
		fmt.Fprintln(logF, line)
	}
	// Anomalies go to stdout, the main log AND the anomaly file.
	alogf := func(format string, a ...any) {
		line := time.Now().UTC().Format(time.RFC3339) + " " + fmt.Sprintf(format, a...)
		fmt.Println(line)
		fmt.Fprintln(logF, line)
		fmt.Fprintln(anomF, line)
	}
	writeZeros := func(mids []float64, note string) {
		if zerosF == nil {
			return
		}
		if note != "" {
			fmt.Fprintf(zerosF, "# %s\n", note)
		}
		for _, z := range mids {
			fmt.Fprintf(zerosF, "%.6f\n", z)
		}
	}

	if st.NextT > st.StartT {
		logf("resume start=%g at=%.6f zeros_so_far=%d blocks=%d", st.StartT, st.NextT, st.ZerosFound, st.Blocks)
	} else {
		logf("hunt start=%g block=%g workers=%d stepdiv=%g", start, *block, *workers, *stepdiv)
	}

	// Pre-build the ln/rsqrt tables so the first block's timing is honest.
	tb := time.Now()
	mMax := int(math.Sqrt((st.NextT + *block) / (2 * math.Pi)))
	getTables(mMax)
	if d := time.Since(tb); d > 100*time.Millisecond {
		logf("tables built n=1..%d in %.2fs", mMax, d.Seconds())
	}

	const backWindow = 64 // recent blocks eligible for backscan
	hist := st.Hist       // survives restarts via the state file
	lastSum := time.Now()
	sumT0, sumZ0 := st.NextT, st.ZerosFound

	for *end == 0 || st.NextT < *end {
		t0 := st.NextT
		t1 := t0 + *block
		if *end > 0 && t1 > *end {
			t1 = *end
		}
		spacing := 2 * math.Pi / math.Log(((t0+t1)/2)/(2*math.Pi))
		// Dyadic lattice spacing: a power of two near spacing/stepdiv, so
		// every scan point is exactly representable and adjacent blocks
		// tile the same global lattice with no gaps or double-counting.
		hBase := math.Ldexp(1, int(math.Round(math.Log2(spacing / *stepdiv))))
		getTables(int(math.Sqrt(t1 / (2 * math.Pi))))

		blockStart := time.Now()
		// One NUFFT pass captures the band-limited main sum for the whole
		// block; every scan and rescan below is interpolation against it.
		bg, evals := buildBlockGrid(t0, t1, *workers)
		mids, e := bg.scanZ(t0, t1, hBase, *workers)
		evals += e
		expected := nDiff(t0, t1)

		// Rescan ladder. The pair that survives a targeted pass is usually
		// the one whose window the S-wobble masks while wobble windows
		// elsewhere still flag, and relocalizing at the same threshold
		// chases the wrong windows. Each rung therefore relaxes the window
		// threshold, with one full-block interpolation sweep as backstop.
		// Direct dd-grid passes (required below one ulp of t) stay
		// targeted only.
		hMin := math.Nextafter(t1, math.Inf(1)) - t1
		rescans, mult := 0, 1
		for _, p := range []struct {
			mult     int
			thresh   float64 // window flag level; 0 = full-block sweep
			minDrift float64
		}{
			{2, -1.2, -0.9},  // classic: unmasked windows
			{2, -0.7, -0.9},  // liberal: wobble-masked windows, still cheap
			{8, -0.7, -0.9},  // finer grid, liberal windows
			{8, 0, -1.3},     // full-block sweep: no blind spots
			{32, -0.7, -1.7}, // direct engine on liberal windows
		} {
			drift := float64(len(mids)) - expected
			if drift > p.minDrift {
				break
			}
			h := hBase / float64(p.mult)
			direct := h < hMin
			var wins [][2]float64
			mode := ""
			if direct {
				// Direct evaluation is expensive. Use the tightest
				// localization available and refuse unbounded spans:
				// liberal window lists can cover a quarter of the block
				// and a direct sweep of that costs minutes.
				const directSpanCap = 25000.0
				for _, th := range []float64{-1.2, -1.0, -0.8} {
					w := localizeDeficit(t0, t1, mids, th)
					if w == nil {
						continue
					}
					span := 0.0
					for _, x := range w {
						span += x[1] - x[0]
					}
					if span <= directSpanCap {
						wins = w
						mode = fmt.Sprintf("targeted(%d windows, %.0f units, thresh %.1f)", len(w), span, th)
					}
					break // looser thresholds only grow the span
				}
				if wins == nil {
					logf("rescan block=%d pass=%dx skipped: no bounded localization for direct pass; backscan/anomaly will follow up", st.Blocks+1, p.mult)
					break
				}
			} else if p.thresh < 0 {
				wins = localizeDeficit(t0, t1, mids, p.thresh)
				if wins != nil {
					span := 0.0
					for _, w := range wins {
						span += w[1] - w[0]
					}
					if span < 0.5*(t1-t0) {
						mode = fmt.Sprintf("targeted(%d windows, %.0f units, thresh %.1f)", len(wins), span, p.thresh)
					} else {
						wins = [][2]float64{{t0, t1}}
						mode = "full"
					}
				}
				if wins == nil {
					continue // nothing flagged at this level; try the next rung
				}
			} else {
				wins = [][2]float64{{t0, t1}}
				mode = "full"
			}
			rescans++
			if direct {
				mode += " direct"
			}
			logf("rescan block=%d pass=%dx %s drift=%+.3f", st.Blocks+1, p.mult, mode, drift)
			prev := mids
			for _, wnd := range wins {
				var nm []float64
				var e2 int64
				if direct {
					nm, e2 = scanBlockChunked(wnd[0], wnd[1], h, *workers)
				} else {
					nm, e2 = bg.scanZ(wnd[0], wnd[1], h, *workers)
				}
				evals += e2
				mids = mergeReplace(mids, wnd[0], wnd[1], nm)
			}
			// A rescan may only nudge the count by the recovered pairs.
			// Anything larger means a broken scan and is rejected.
			d := len(mids) - len(prev)
			if d < -2 || d > 24+4*len(wins) {
				alogf("REJECTED rescan block=%d pass=%dx: count moved by %+d (implausible); keeping previous list",
					st.Blocks+1, p.mult, d)
				mids = prev
				break
			}
			if mode == "full" && !direct && p.mult > mult {
				mult = p.mult
			}
			if direct && d <= 0 {
				// A fruitless direct pass at 2048+ points per spacing is
				// conclusive for those windows.
				break
			}
		}
		dur := time.Since(blockStart)

		st.ZerosFound += int64(len(mids))
		st.Blocks++
		st.Evals += evals
		st.Seconds += dur.Seconds()
		st.NextT = t1
		st.UpdatedAt = time.Now()
		hist = append(hist, blockRec{t0, t1, hBase, len(mids), mult})
		if len(hist) > backWindow {
			hist = hist[1:]
		}
		writeZeros(mids, "")

		cum := float64(st.ZerosFound) - nDiff(st.StartT, t1)
		base := baseline(st.CumHistory)
		dev := cum - base
		// Until the baseline has history, dev equals raw cum drift and
		// S(T) wander would fire global triggers spuriously. During
		// warm-up only the per-block checks run.
		warmedUp := len(st.CumHistory) >= 20

		logf("block=%d t=[%.3f,%.3f] zeros=%d expected=%.3f drift=%+.3f cum_drift=%+.3f dev=%+.3f rescans=%d evals=%d dur=%.2fs rate=%.0f/s total_zeros=%d",
			st.Blocks, t0, t1, len(mids), expected, float64(len(mids))-expected,
			cum, dev, rescans, evals, dur.Seconds(), float64(evals)/dur.Seconds(), st.ZerosFound)

		// Backscan: a step-drop of ~2 in dev means a pair went missing in
		// a recent block, not necessarily this one.
		var suspects []string
		if warmedUp && dev <= -1.8 {
			// Walk back to the step origin: the last block whose recorded
			// cum drift was still near the baseline, plus wobble margin.
			// Wobble can delay the trigger several blocks past the loss.
			origin := 0
			ch := st.CumHistory
			off := len(ch) - (len(hist) - 1) // hist's last entry has no cum recorded yet
			for i := len(hist) - 2; i >= 0; i-- {
				j := off + i
				if j < 0 || j >= len(ch) {
					break
				}
				if ch[j] >= base-0.5 {
					origin = i - 2
					break
				}
			}
			if origin < 0 {
				origin = 0
			}
			logf("backscan window: blocks %d..%d (drift step origin, margin 2)", origin+1, len(hist))
			// Walk oldest-first. The missing pair lives where the descent began.
			for i := origin; i < len(hist) && dev <= -1.8; i++ {
				b := &hist[i]
				bExpected := nDiff(b.T0, b.T1)
				hMinB := math.Nextafter(b.T1, math.Inf(1)) - b.T1
				var last []float64
				accept := func(m2 []float64, densLabel string) {
					diff := float64(len(m2)) - bExpected
					if diff > 8 {
						// Too many crossings means a broken scan. Never
						// let it near the ledger.
						alogf("REJECTED backscan t=[%.3f,%.3f] %s: %d zeros vs expected %.1f (implausibly high)",
							b.T0, b.T1, densLabel, len(m2), bExpected)
						return
					}
					if diff < -8 {
						// A very low count is an honest scan of a block
						// rich in tight pairs. It cannot harm the ledger
						// (only higher counts are ever accepted) and its
						// gaps still localize the missing pairs.
						logf("backscan t=[%.3f,%.3f] %s found %d, %.1f short; block is tight-pair rich, finer passes follow",
							b.T0, b.T1, densLabel, len(m2), -diff)
					}
					last = m2
					if len(m2) > b.Found {
						delta := len(m2) - b.Found
						b.Found = len(m2)
						st.ZerosFound += int64(delta)
						writeZeros(m2, fmt.Sprintf("rescan of [%.3f,%.3f]: supersedes earlier entries in this range", b.T0, b.T1))
						cum = float64(st.ZerosFound) - nDiff(st.StartT, t1)
						dev = cum - base
						logf("backscan t=[%.3f,%.3f] %s recovered %d zeros, cum_drift=%+.3f dev=%+.3f",
							b.T0, b.T1, densLabel, delta, cum, dev)
					}
				}
				// Build the interpolation grid once for both density passes.
				var bgb *blockGrid
				for _, m := range []int{2, 8} {
					if m <= b.Mult || b.HBase/float64(m) < hMinB {
						continue
					}
					logf("backscan t=[%.3f,%.3f] pass=%dx dev=%+.3f", b.T0, b.T1, m, dev)
					bs := time.Now()
					if bgb == nil {
						var e1 int64
						bgb, e1 = buildBlockGrid(b.T0, b.T1, *workers)
						st.Evals += e1
					}
					m2, e2 := bgb.scanZ(b.T0, b.T1, b.HBase/float64(m), *workers)
					st.Evals += e2
					st.Seconds += time.Since(bs).Seconds()
					b.Mult = m
					accept(m2, fmt.Sprintf("%dx", m))
					if dev > -1.8 {
						break
					}
				}
				// Deep rung: direct dd-grid rescan of localized windows,
				// tightest available localization, span capped.
				if dev <= -1.8 && last != nil {
					var wins [][2]float64
					for _, th := range []float64{-1.2, -1.0, -0.8} {
						w := localizeDeficit(b.T0, b.T1, last, th)
						if w == nil {
							continue
						}
						span := 0.0
						for _, x := range w {
							span += x[1] - x[0]
						}
						if span <= 25000 {
							wins = w
						}
						break
					}
					if wins != nil {
						logf("backscan t=[%.3f,%.3f] pass=32x direct targeted(%d windows) dev=%+.3f",
							b.T0, b.T1, len(wins), dev)
						bs := time.Now()
						merged := last
						for _, w := range wins {
							nm, e2 := scanBlockChunked(w[0], w[1], b.HBase/32, *workers)
							st.Evals += e2
							merged = mergeReplace(merged, w[0], w[1], nm)
						}
						st.Seconds += time.Since(bs).Seconds()
						accept(merged, "32x-direct")
					}
				}
				// A block still short after every rung is a suspect region
				// for the anomaly report, with its best-guess windows.
				if short := bExpected - float64(b.Found); short >= 1.5 && len(suspects) < 5 {
					s := fmt.Sprintf("t=[%.0f,%.0f] short %.1f", b.T0, b.T1, short)
					if last != nil {
						if wins := localizeDeficit(b.T0, b.T1, last, -0.7); wins != nil && len(wins) <= 4 {
							for _, w := range wins {
								s += fmt.Sprintf(" window[%.3f,%.3f]", w[0], w[1])
							}
						}
					}
					suspects = append(suspects, s)
				}
			}
		}

		// Alarm only on a NEW deficit. Acknowledged ones do not repeat
		// every block. AckDev relaxes upward as the drift recovers.
		if warmedUp && dev <= -1.8 && dev <= st.AckDev-0.9 {
			st.Anomalies++
			st.AckDev = dev
			where := "no single block pins the deficit; run ./riemann check over recent heights"
			if len(suspects) > 0 {
				where = "suspect regions: " + strings.Join(suspects, "; ")
			}
			alogf("ANOMALY t<=%.6f cum_drift=%+.3f dev=%+.3f baseline=%+.3f. Deficit survived all rescans of the last %d blocks. %s. Investigate with ./riemann check <t0> <t1>, then verify independently (mpmath) before announcing.",
				t1, cum, dev, base, len(hist), where)
		} else if d := dev - 0.5; d > st.AckDev {
			st.AckDev = d
		}

		st.CumHistory = append(st.CumHistory, cum)
		if len(st.CumHistory) > 256 {
			st.CumHistory = st.CumHistory[len(st.CumHistory)-256:]
		}
		st.Hist = hist
		saveState(*statePath, st)

		if *summarySec > 0 && time.Since(lastSum).Seconds() >= *summarySec {
			el := time.Since(lastSum).Seconds()
			logf("SUMMARY height=%.0f zeros=%d rate=%.0f zeros/s pace=%.3g units/day dev=%+.3f anomalies=%d",
				st.NextT, st.ZerosFound, float64(st.ZerosFound-sumZ0)/el,
				(st.NextT-sumT0)/el*86400, dev, st.Anomalies)
			lastSum = time.Now()
			sumT0, sumZ0 = st.NextT, st.ZerosFound
		}
	}
	logf("done end=%g total_zeros=%d blocks=%d evals=%d scan_time=%.0fs anomalies=%d",
		*end, st.ZerosFound, st.Blocks, st.Evals, st.Seconds, st.Anomalies)
}

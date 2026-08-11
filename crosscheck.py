# Independent validation: compare the Go ZFast (Riemann-Siegel, dd phase)
# against mpmath's arbitrary-precision siegelz. Run anywhere:
#   pip install mpmath && python crosscheck.py [path-to-riemann-binary]
import subprocess
import sys
import time

import mpmath as mp

mp.mp.dps = 30
exe = sys.argv[1] if len(sys.argv) > 1 else "./riemann"
points = [100000.5, 1000000.3, 100000000.7, 10000000000.1,
          100000000000.9, 3000000000000.5]

worst = 0.0
for t in points:
    t0 = time.time()
    ref = float(mp.siegelz(mp.mpf(t)))
    dt = time.time() - t0
    out = subprocess.check_output([exe, "z", repr(t)], text=True)
    ours = float(out.splitlines()[0].split("=")[1].split("(")[0])
    diff = abs(ours - ref)
    worst = max(worst, diff)
    print(f"t={t:<16} mpmath={ref:+.12f} ({dt:.1f}s)  ours={ours:+.12f}  diff={diff:.3e}")

print(f"worst diff: {worst:.3e}")
sys.exit(0 if worst < 1e-4 else 1)

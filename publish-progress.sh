#!/bin/sh
cd /home/alphaalex115/Riemann-Hypothesis-Counter-Example-Hunt
git add hunt.state.json hunt.anomalies.log
git diff --cached --quiet || git commit -m "progress $(date -u +%Y-%m-%d): $(python3 -c "import json;s=json.load(open('hunt.state.json'));print(f\"height {s['next_t']:.6g}, {s['zeros_found']:,} zeros\")")"
git push

# Introduction

Vibe-coding with Claude to disprove the Riemann Hypothesis to make $1,000,000 (depending on Clay's discretion) and to let everyone know it happened on the same machine that runs my Minecraft server **[minecraftoffline.net](https://minecraftoffline.net)**. Please join it. I beg you.

## The method

Scan above the verified frontier (t ≈ 3×10¹² Platt & Trudgian 2021) and compare the number of zeros found *on* the critical line against the number the argument principle says must exist *in* the strip. Off-line zeros come in mirror pairs (σ, 1−σ) which means a counterexample appears as a persistent step-drop of ≥ 2 in the count that survives rescans at much finer steps.

## Setup I did

Install Go:

```sh
cd ~
wget -4 https://dl.google.com/go/go1.26.1.linux-amd64.tar.gz
tar -xzf go1.26.1.linux-amd64.tar.gz && mv go go-sdk && rm go1.26.1.linux-amd64.tar.gz
echo 'export PATH=$HOME/go-sdk/bin:$PATH' >> ~/.bashrc
export PATH=$HOME/go-sdk/bin:$PATH
```

Clone this repo. Test and build:

```sh
cd ~/Riemann-Hypothesis-Counter-Example-Hunt
go test ./...
GOAMD64=v3 go build -o riemann .
```

Smoke test. It should end with "CERTIFIED: every zero in the certified interval is on the critical line" (Turing-certified integer counting; ranges too short or too low for anchors fall back to "counts agree"):

```sh
./riemann check 3000000000000 3000000000100
```

## Running it 24/7

One systemd unit as root:

```ini
# /etc/systemd/system/zeta.service
[Unit]
Description=Riemann zeta zero counterexample hunt
After=network.target

[Service]
User=alphaalex115
WorkingDirectory=/home/alphaalex115/Riemann-Hypothesis-Counter-Example-Hunt
ExecStart=/home/alphaalex115/Riemann-Hypothesis-Counter-Example-Hunt/riemann hunt 3000000000000 -workers 10
Nice=19
IOSchedulingClass=idle
MemoryMax=12G
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```sh
systemctl daemon-reload
systemctl enable --now zeta
tail -f /home/alphaalex115/Riemann-Hypothesis-Counter-Example-Hunt/hunt.log
```

## Updating the hunt

Safe at any time. Restart loses at most one in-flight block and it resumes from hunt.state.json.

```sh
cd ~/Riemann-Hypothesis-Counter-Example-Hunt
git pull
GOAMD64=v3 go build -o riemann .
```

Then as root: `systemctl restart zeta`

## Normal Operation

- One block line roughly every minute. Every block is still scanned and counted but only every 10th writes a log line. Blocks that needed heavy passes or drifted hard always log. Pass -logevery 1 to log every block.
- Each block is 1 million height units which is about 4.28 million zeros.
- drift stays between −1.5 and +1.5 and dev stays between −2.5 and +2.5. That is ordinary statistical wobble.
- probes=c/r on the block line counts dip probes. A few zero pairs per block sit closer together than the scan grid. Each one leaves a fingerprint where Z dips near zero without crossing. The base scan spots these as it goes and tiny probes confirm the pair and add it to the count. c candidates checked and r zeros recovered is normal on most blocks.
- Every block ends with a silent Turing certification. It proves the exact total number of zeros below the block end using Turing's method. The count of zeros found must match that proven total exactly.
- rescan, diphunt and backscan lines are rare fallbacks. A rescan with certified_deficit means the proven total showed a zero was missed and heavier passes went and found it. Fine in moderation.
- A "rescan below" line means a certified deficit pointed into an earlier block (typically a pair hiding in the previous block's anchor window); that block gets the full heavy ladder before any alarm can fire. Also fine in moderation.
- A "turing anchor failed" line means certification could not lock on for one block and will retry next block. Rare and fine.
- A SUMMARY line every 15 minutes with totals and pace. certified_N is the proven total number of zeros below the last certified anchor.
- hunt.anomalies.log stays empty.

## Progress Checkpoints

hunt.state.json holds the height reached and total zeros.

## Jackpot Signal:

- A TURING DEFICIT line in hunt.anomalies.log is the real signal. It means zeros are PROVEN missing from the critical line by certified integer counting after every recovery pass failed to find them (including the below-block ladder over the whole certified interval). Not statistics. Investigate with `./riemann check <t0> <t1>` over the region in the line -- give it a margin of ~15 units on each side so its own Turing anchors can lock, and a CERTIFIED DEFICIT verdict there is a second independent certification -- then verify with independent tools.
- An ANOMALY line is the older statistical signal and still worth checking the same way.
- A TURING SURPLUS line would mean more crossings counted than zeros exist. That is a bug in the program not a discovery.

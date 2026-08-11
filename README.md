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

Smoke test. It should end with "counts agree":

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

## Normal Operation

- One block line roughly every 10 seconds. Each block is 1 million height units which is about 4.28 million zeros.
- drift and dev stay between −1.5 and +1.5. That is ordinary statistical wobble.
- rescan lines on most blocks. The base scan straddles a few very close zero pairs per block. The rescan passes find where they hide and recover them. The drift on a rescan line is measured before recovery.
- A SUMMARY line every 15 minutes with totals and pace.
- hunt.anomalies.log stays empty.

## Progress Checkpoints

hunt.state.json holds the height reached and total zeros.

## Jackpot Signal:

- An ANOMALY line means a deficit of about 2 zeros survived every rescan. Investigate with `./riemann check <t0> <t1>` over the region in the line. It escalates density until the count settles. If the deficit survives that then verify with independent tools.

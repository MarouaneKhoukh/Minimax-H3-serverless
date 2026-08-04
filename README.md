# MiniMax-H3 on Nebius Serverless AI (SGLang)

End-to-end deployment of [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) — a video-and-audio generation model — served with [SGLang Diffusion](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3) on a single 8×B200 [Nebius Serverless AI endpoint](https://docs.nebius.com/serverless/).

## Background

**[SGLang](https://github.com/sgl-project/sglang)** is a high-performance serving framework for LLMs and multimodal models. Its Diffusion stack serves image/video generation models behind an OpenAI-compatible async API. Docs: [docs.sglang.io](https://docs.sglang.io).

**[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)** generates a video **and** a synchronized stereo audio track in one request (MP4: H.264 @ 24 fps + AAC stereo @ 32 kHz). It ships two checkpoint partitions: `fl2va` (text / first-last-frame conditioning) and `ref2va` (image/video/audio references). This repo deploys the **`fl2va`** partition and uses the **`t2va`** (text → video+audio) task. Model card: [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) · Recipe source: [SGLang MiniMax-H3 cookbook](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3).

This deployment is the **upstream-verified SGLang recipe, unmodified**: 8 GPUs, Ulysses degree 8, BF16, resident profile — translated 1:1 to Nebius Serverless AI. No custom Docker image; the stock `lmsysorg/sglang:dev` image installs the diffusion extras at container start (that is the official launch form).

## What gets created

| | |
|---|---|
| Image | `lmsysorg/sglang:dev` (public, Docker Hub) |
| Platform / preset | `gpu-b200-sxm` / `8gpu-160vcpu-1792gb` (8×B200, us-central1) |
| Entrypoint | `pip install -e ".../python[diffusion]"` → `sglang serve` with the verified flags |
| Port | 30010 → served via Nebius managed HTTPS URL |
| Disk / shm | 1 TiB / 32 GiB |
| Auth | Bearer token (enforced at the Nebius HTTPS proxy) |

## Prerequisites

- A Nebius project in **us-central1** (B200 region; in `me-west1` use platform `gpu-b200-sxm-a`)
- Compute quota for **8× B200 GPUs** ([check quotas](https://console.nebius.com/quota))
- For the CLI path: [Nebius CLI](https://docs.nebius.com/cli/install) installed and [configured](https://docs.nebius.com/cli/configure), plus `jq`, `openssl`, `curl`
- Optional: a Hugging Face token if the model repo is gated for your account

---

## Step 1 — Create the endpoint

### Option A: Web console

[console.nebius.com](https://console.nebius.com) → **AI Services → Endpoints → Create endpoint**, then fill in:

| Field | Value |
|---|---|
| Name | `minimax-h3-sglang` |
| Image path | `lmsysorg/sglang:dev` |
| Ports | `30010` |
| Entrypoint command | see below (one line, command + args together) |
| Environment variables | `HF_TOKEN = <your token>` (only if the HF repo is gated) |
| Authentication | **Token** — copy and save the generated token |
| Computing resources | With GPU · VM type **Regular** (not preemptible — a preemption discards the loaded model) |
| Platform | NVIDIA® B200 NVLink (`gpu-b200-sxm`) |
| Preset | `8gpu-160vcpu-1792gb` (8 GPU / 160 vCPU / 1792 GiB) |
| Container disk size | `1000 GiB` |
| Access | Add your SSH key (username `nebius`) — needed later for the in-container benchmark |
| Network | Your subnet · Private IP is enough (the managed HTTPS URL is public regardless) |

Entrypoint command:

```
bash -lc 'python -m pip install -e "/sgl-workspace/sglang/python[diffusion]" && exec sglang serve --model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va --num-gpus 8 --ulysses-degree 8 --performance-mode speed --host 0.0.0.0 --port 30010'
```

Keep the quoting exactly as shown. Note: the console doesn't expose the `/dev/shm` size (CLI default is 16 GiB on GPU platforms; upstream uses 32 GiB) — this worked fine in testing, but if you ever hit NCCL/shared-memory errors, recreate via CLI with `--shm-size 32Gi`.

### Option B: CLI

```bash
export AUTH_TOKEN=$(openssl rand -hex 32)
echo "$AUTH_TOKEN"   # save this
export SUBNET_ID=$(nebius vpc subnet list --format jsonpath='{.items[0].metadata.id}')

nebius ai endpoint create \
  --name minimax-h3-sglang \
  --image lmsysorg/sglang:dev \
  --container-command "bash" \
  --args "-lc 'python -m pip install -e \"/sgl-workspace/sglang/python[diffusion]\" && exec sglang serve --model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va --num-gpus 8 --ulysses-degree 8 --performance-mode speed --host 0.0.0.0 --port 30010'" \
  --container-port 30010 \
  --env HF_TOKEN=<your-hf-token> \
  --platform gpu-b200-sxm \
  --preset 8gpu-160vcpu-1792gb \
  --disk-size 1Ti \
  --shm-size 32Gi \
  --auth token \
  --token "$AUTH_TOKEN" \
  --ssh-key "$(cat ~/.ssh/id_ed25519.pub)" \
  --subnet-id "$SUBNET_ID"
```

Drop `--env HF_TOKEN=...` if the repo isn't gated for you. For secrets hygiene, prefer `--env-secret "HF_TOKEN=<mysterybox-secret>"` ([MysteryBox](https://docs.nebius.com/mysterybox/index)) over a plaintext token.

## Step 2 — Wait for startup

```bash
export ENDPOINT_ID=$(nebius ai endpoint get-by-name --name minimax-h3-sglang --format json | jq -r '.metadata.id')
nebius ai endpoint logs $ENDPOINT_ID     # console: endpoint page → Logs tab
```

First boot sequence: image pull → pip install of diffusion extras (~1–3 min) → **Hugging Face checkpoint download (dominant, tens of minutes)** → model load (~2 min) → warmup (~30 s). The server is ready when the logs show it listening on `0.0.0.0:30010`. **Until then, requests return `502 Bad Gateway` from the proxy — that's startup, not an error.**

Get the URL (console: endpoint page → Network → Public endpoints, copy the `https://...` one):

```bash
export ENDPOINT_URL=$(nebius ai endpoint get $ENDPOINT_ID --format json \
  | jq -r '.status.public_endpoints[] | select(startswith("https://"))' | head -1)
```

Always use the managed **`https://` URL**, not the raw `IP:30010` — token auth is enforced only at the HTTPS proxy, and the raw path is unencrypted.

## Step 3 — Generate a video

The API is async: submit → poll → download. Every call needs the Bearer header.

**Submit:**

```bash
video_id=$(curl -sS -X POST "$ENDPOINT_URL/v1/videos" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMaxAI/MiniMax-H3",
    "prompt": "Night-vision bedroom footage: three cats march in playing tiny brass instruments, freeze, then file out.",
    "seconds": 5,
    "task": "t2va",
    "conditions": [],
    "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0},
    "num_outputs_per_prompt": 1,
    "num_inference_steps": 50,
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
    "seed": 1101
  }' | jq -r '.id')
echo "$video_id"
```

**Poll** (variable is `vstatus` — `status` is read-only in zsh):

```bash
while true; do
  vstatus=$(curl -sS "$ENDPOINT_URL/v1/videos/${video_id}" \
    -H "Authorization: Bearer $AUTH_TOKEN" | jq -r '.status' 2>/dev/null)
  echo "${vstatus:-no response yet}"
  [ "$vstatus" = "completed" ] && break
  [ "$vstatus" = "failed" ] && { echo "generation failed"; break; }
  sleep 5
done
```

**Download** (saved to your current directory):

```bash
curl -sS -L "$ENDPOINT_URL/v1/videos/${video_id}/content" \
  -H "Authorization: Bearer $AUTH_TOKEN" -o output.mp4
```

Duration 4–15 s via `seconds`/`duration_seconds` (keep them equal). Multiple variants: set `"num_outputs_per_prompt": N` (≤10) and fetch `.../content?variant=0`, `?variant=1`, …

## Step 4 — Benchmark

Per-request denoise/decode timing appears in the server logs after every generation — that's the zero-effort option. For comparable numbers, run the official driver inside the container (requires the SSH key from Step 1):

```bash
nebius ai endpoint ssh $ENDPOINT_ID -s bash
# inside the container:
python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
  --host 127.0.0.1 --port 30010 \
  --model MiniMaxAI/MiniMax-H3 \
  --dataset vbench --task text-to-video \
  --num-prompts 1 --max-concurrency 1 \
  --warmup-requests 1 --warmup-inference-steps 50 \
  --extra-body '{"task":"t2va","conditions":[],"target":{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":5.0},"seconds":5,"flow_shift":12.0,"audio_flow_shift":3.0}'
```

Measured results (BF16, FL2VA, t2va, 5 s @ 768/16:9, 50 steps, 1 warmup + 1 measured request):

| Setup | Latency | Peak memory / GPU |
|---|---|---|
| **This deployment — 8×B200, Nebius Serverless AI** | **20.06 s** | **83,564 MB** |
| SGLang reference — 8×B300 ([cookbook](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3)) | 19.04 s | 83,578 MB |

~5% behind B300, as expected for the older chip; peak memory matches the reference within 14 MB, confirming the serverless container runs the exact verified topology (resident pipeline, Ulysses-8, encoder folding) with no overhead. Use `--num-prompts 5` or more for meaningful percentiles. GPU/vCPU utilization is on the endpoint's **Metrics** tab in the console.

## Step 5 — Cost control

The endpoint bills per second while running ([Compute VM pricing](https://docs.nebius.com/compute/resources/pricing)).

```bash
nebius ai endpoint stop  --id $ENDPOINT_ID    # pause compute billing, keep the endpoint
nebius ai endpoint start --id $ENDPOINT_ID
nebius ai endpoint delete --id $ENDPOINT_ID   # removes VM + disk (incl. downloaded weights)
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `502 Bad Gateway` | Server still starting — check logs; wait for "listening on 0.0.0.0:30010" |
| Download stalls at 0% / auth error in logs | HF token missing/invalid, or model license not accepted on the HF page |
| `jq: parse error` | Response wasn't JSON — rerun the curl with `-i` and no `jq` to see the real error |
| `zsh: read-only variable: status` | Don't name shell variables `status` in zsh |
| NCCL / shared-memory errors | Recreate via CLI with `--shm-size 32Gi` |
| Endpoint died, nothing to debug | Failed endpoints get their VM deleted; redeploy with the sleep-on-fail wrapper ([docs](https://docs.nebius.com/serverless/jobs/failure)) — see `DEBUG_SLEEP_ON_FAIL` below |
| Requests hit `IP:port` and skip auth | Use the managed `https://` URL; the raw IP bypasses the token check |

---

## Scripts

Everything above, automated:

```bash
cp .env.example .env             # edit: HF_TOKEN, platform, debug flags
./deploy/create_endpoint.sh      # Steps 1–2: creates the endpoint, prints ID / URL / token
./scripts/generate.sh "a red fox running through fresh snow at golden hour"   # Step 3
```

- **`deploy/create_endpoint.sh`** — creates the endpoint with the verified recipe; generates and prints the auth token; supports `HF_TOKEN`, `SSH_PUBKEY`, and `DEBUG_SLEEP_ON_FAIL=1` (keeps the VM alive 24 h after a crash for SSH post-mortem) via `.env`.
- **`scripts/generate.sh`** — submit → poll → download with elapsed-time output; `SECONDS_LEN`, `STEPS`, `SEED`, `NUM_OUTPUTS` overridable via env; multi-variant downloads handled automatically. Requires `ENDPOINT_URL` and `AUTH_TOKEN` in `.env` or the environment.

`.gitignore` excludes `.env` and `*.mp4` so tokens and generated videos never get committed.

## References

- SGLang MiniMax-H3 cookbook: https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3
- SGLang: https://github.com/sgl-project/sglang · https://docs.sglang.io
- MiniMax-H3 model card: https://huggingface.co/MiniMaxAI/MiniMax-H3
- Nebius Serverless AI: https://docs.nebius.com/serverless/
- Nebius endpoint management: https://docs.nebius.com/serverless/endpoints/manage
- `nebius ai endpoint create` reference: https://docs.nebius.com/cli/reference/ai/endpoint/create

> Review the license and usage terms on the MiniMax-H3 model card before production or commercial use.

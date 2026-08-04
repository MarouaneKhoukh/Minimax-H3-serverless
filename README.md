# MiniMax-H3 on Nebius Serverless AI 

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

- [Nebius CLI](https://docs.nebius.com/cli/install) installed and [configured](https://docs.nebius.com/cli/configure) for a project in **us-central1** (B200 region; use `gpu-b200-sxm-a` in `me-west1`)
- Compute quota for **8× B200 GPUs** ([check quotas](https://console.nebius.com/quota))
- `jq`, `openssl`, `curl`
- Optional: a Hugging Face token if the model repo is gated for your account

## Deploy

```bash
cp .env.example .env        # edit if you need HF_TOKEN
./deploy/create_endpoint.sh
```

The script prints the endpoint ID, the auth token, and the managed HTTPS URL. **Save the token.**

First boot is slow: image pull → pip install of diffusion extras (~1–3 min) → Hugging Face checkpoint download (**dominant, tens of minutes**) → model load (~2 min) → warmup (~30 s). Watch progress:

```bash
nebius ai endpoint logs <endpoint_ID>
```

The server is ready when the logs show it listening on `0.0.0.0:30010`. Until then, requests return **502** from the proxy — that is startup, not an error.

## Generate a video

```bash
./scripts/generate.sh "Night-vision bedroom footage: three cats march in playing tiny brass instruments, freeze, then file out."
```

Submits the request, polls until completion, downloads `output.mp4`. All parameters (duration 4–15 s, steps, seed, resolution) are at the top of the script — they default to the upstream-verified profile (5 s, 768 short edge, 50 steps).

The API is async: `POST /v1/videos` returns an ID, `GET /v1/videos/{id}` reports status, `GET /v1/videos/{id}/content` returns the MP4. Every call needs `Authorization: Bearer <token>`.

## Benchmark / metrics

- **Server logs** print per-request denoise/decode timing — zero-effort latency numbers.
- **Official benchmark** (same driver as SGLang's published sweep). Requires the endpoint to have an SSH key; then:

  ```bash
  nebius ai endpoint ssh <endpoint_ID> -s bash
  # inside the container:
  python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
    --host 127.0.0.1 --port 30010 \
    --model MiniMaxAI/MiniMax-H3 \
    --dataset vbench --task text-to-video \
    --num-prompts 1 --max-concurrency 1 \
    --warmup-requests 1 --warmup-inference-steps 50 \
    --extra-body '{"task":"t2va","conditions":[],"target":{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":5.0},"seconds":5,"flow_shift":12.0,"audio_flow_shift":3.0}'
  ```

  Reference: SGLang measured **19.04 s** per request (BF16, FL2VA, t2va) on 8×B300 after warmup; B200 lands in the same ballpark.
- **GPU/vCPU utilization**: endpoint **Metrics** tab in the Nebius console.

## Cost control

The endpoint bills per second while running ([Compute VM pricing](https://docs.nebius.com/compute/resources/pricing)).

```bash
nebius ai endpoint stop  --id <endpoint_ID>   # pause compute billing, keep the endpoint
nebius ai endpoint start --id <endpoint_ID>
nebius ai endpoint delete --id <endpoint_ID>  # removes VM + disk (incl. downloaded weights)
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `502 Bad Gateway` | Server still starting — check logs; wait for "listening on 0.0.0.0:30010" |
| Download stalls at 0% / auth error in logs | HF token missing/invalid, or model license not accepted on the HF page |
| `jq: parse error` | Response wasn't JSON — rerun the curl with `-i` and no `jq` to see the real error |
| NCCL / shared-memory errors | Raise `--shm-size` (this repo sets 32Gi, matching upstream) |
| Endpoint died, nothing to debug | Failed endpoints get their VM deleted; redeploy with `DEBUG_SLEEP_ON_FAIL=1` in `.env` to keep the container alive 24 h after a crash ([docs](https://docs.nebius.com/serverless/jobs/failure)) |
| zsh: `read-only variable: status` | Don't name shell variables `status` in zsh (scripts here already avoid it) |

Use a **regular** (non-preemptible) VM: a preemption discards the loaded model and any in-flight generations, costing tens of minutes of re-warmup.

## References

- SGLang MiniMax-H3 cookbook: https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3
- SGLang: https://github.com/sgl-project/sglang · https://docs.sglang.io
- MiniMax-H3 model card: https://huggingface.co/MiniMaxAI/MiniMax-H3
- Nebius Serverless AI: https://docs.nebius.com/serverless/
- Nebius endpoint management: https://docs.nebius.com/serverless/endpoints/manage
- `nebius ai endpoint create` reference: https://docs.nebius.com/cli/reference/ai/endpoint/create

> Review the license and usage terms on the MiniMax-H3 model card before production or commercial use.

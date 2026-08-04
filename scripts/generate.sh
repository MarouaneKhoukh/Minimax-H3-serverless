#!/usr/bin/env bash
# Submit a t2va request to the endpoint, poll until done, download the MP4.
# Usage: ENDPOINT_URL=... AUTH_TOKEN=... ./scripts/generate.sh "your prompt" [outfile.mp4]
set -euo pipefail

[ -f "$(dirname "$0")/../.env" ] && set -a && . "$(dirname "$0")/../.env" && set +a

: "${ENDPOINT_URL:?set ENDPOINT_URL (managed https:// URL of the endpoint)}"
: "${AUTH_TOKEN:?set AUTH_TOKEN (endpoint bearer token)}"

PROMPT="${1:?usage: generate.sh \"prompt\" [outfile.mp4]}"
OUTFILE="${2:-output.mp4}"

# Upstream-verified t2va sampling profile (duration 4-15s allowed)
SECONDS_LEN="${SECONDS_LEN:-5}"
STEPS="${STEPS:-50}"
SEED="${SEED:-1101}"
NUM_OUTPUTS="${NUM_OUTPUTS:-1}"

body=$(jq -n \
  --arg prompt "$PROMPT" \
  --argjson secs "$SECONDS_LEN" \
  --argjson steps "$STEPS" \
  --argjson seed "$SEED" \
  --argjson n "$NUM_OUTPUTS" \
  '{
    model: "MiniMaxAI/MiniMax-H3",
    prompt: $prompt,
    seconds: $secs,
    task: "t2va",
    conditions: [],
    target: {short_edge: 768, aspect_ratio: "16:9", duration_seconds: $secs},
    num_outputs_per_prompt: $n,
    num_inference_steps: $steps,
    flow_shift: 12.0,
    audio_flow_shift: 3.0,
    seed: $seed
  }')

echo ">> submitting ..."
video_id=$(curl -sS -X POST "$ENDPOINT_URL/v1/videos" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d "$body" | jq -r '.id')

if [ -z "$video_id" ] || [ "$video_id" = "null" ]; then
  echo "!! submit failed (server still starting? check: nebius ai endpoint logs <ID>)" >&2
  exit 1
fi
echo ">> job id: $video_id"

# NB: variable is 'vstatus' — 'status' is read-only in zsh
t0=$(date +%s)
while true; do
  vstatus=$(curl -sS "$ENDPOINT_URL/v1/videos/${video_id}" \
    -H "Authorization: Bearer $AUTH_TOKEN" | jq -r '.status' 2>/dev/null || true)
  printf '   %ss  %s\n' "$(( $(date +%s) - t0 ))" "${vstatus:-no response}"
  [ "$vstatus" = "completed" ] && break
  [ "$vstatus" = "failed" ] && { echo "!! generation failed — check server logs" >&2; exit 1; }
  sleep 5
done

echo ">> downloading ..."
if [ "$NUM_OUTPUTS" -gt 1 ]; then
  base="${OUTFILE%.mp4}"
  for v in $(seq 0 $((NUM_OUTPUTS - 1))); do
    curl -sS -L "$ENDPOINT_URL/v1/videos/${video_id}/content?variant=${v}" \
      -H "Authorization: Bearer $AUTH_TOKEN" -o "${base}-${v}.mp4"
    echo "   ${base}-${v}.mp4"
  done
else
  curl -sS -L "$ENDPOINT_URL/v1/videos/${video_id}/content" \
    -H "Authorization: Bearer $AUTH_TOKEN" -o "$OUTFILE"
  echo "   $OUTFILE  ($(du -h "$OUTFILE" | cut -f1))"
fi
echo ">> done in $(( $(date +%s) - t0 ))s (end-to-end, incl. polling)"

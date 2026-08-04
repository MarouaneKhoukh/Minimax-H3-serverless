#!/usr/bin/env bash
# Create a Nebius Serverless AI endpoint serving MiniMax-H3 with SGLang.
# Upstream recipe: https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3
# (B200 / Resident / fl2va / BF16 / encoder auto — verified 8-GPU Ulysses-8 cell)
set -euo pipefail

# ---- config (override via .env) --------------------------------------------
ENDPOINT_NAME="${ENDPOINT_NAME:-minimax-h3-sglang}"
PLATFORM="${PLATFORM:-gpu-b200-sxm}"          # gpu-b200-sxm-a in me-west1
PRESET="${PRESET:-8gpu-160vcpu-1792gb}"
DISK_SIZE="${DISK_SIZE:-1Ti}"
SHM_SIZE="${SHM_SIZE:-32Gi}"                  # matches upstream --shm-size 32g
PORT="${PORT:-30010}"
HF_TOKEN="${HF_TOKEN:-}"                      # only needed if the HF repo is gated
SSH_PUBKEY="${SSH_PUBKEY:-$HOME/.ssh/id_ed25519.pub}"  # for debugging/benchmarks
DEBUG_SLEEP_ON_FAIL="${DEBUG_SLEEP_ON_FAIL:-0}"        # 1 = keep VM alive 24h on crash
# -----------------------------------------------------------------------------

[ -f "$(dirname "$0")/../.env" ] && set -a && . "$(dirname "$0")/../.env" && set +a

SERVE_CMD='python -m pip install -e "/sgl-workspace/sglang/python[diffusion]" && exec sglang serve --model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va --num-gpus 8 --ulysses-degree 8 --performance-mode speed --host 0.0.0.0 --port '"$PORT"

if [ "$DEBUG_SLEEP_ON_FAIL" = "1" ]; then
  # https://docs.nebius.com/serverless/jobs/failure — keep the VM for post-mortem
  SERVE_CMD="( $SERVE_CMD ) || (echo FAILED; sleep 86400)"
fi

AUTH_TOKEN="$(openssl rand -hex 32)"
SUBNET_ID="$(nebius vpc subnet list --format jsonpath='{.items[0].metadata.id}')"

EXTRA_ARGS=()
[ -n "$HF_TOKEN" ] && EXTRA_ARGS+=(--env "HF_TOKEN=$HF_TOKEN")
[ -f "$SSH_PUBKEY" ] && EXTRA_ARGS+=(--ssh-key "$(cat "$SSH_PUBKEY")")

echo ">> creating endpoint '$ENDPOINT_NAME' ($PLATFORM / $PRESET) ..."
nebius ai endpoint create \
  --name "$ENDPOINT_NAME" \
  --image lmsysorg/sglang:dev \
  --container-command "bash" \
  --args "-lc '$SERVE_CMD'" \
  --container-port "$PORT" \
  --platform "$PLATFORM" \
  --preset "$PRESET" \
  --disk-size "$DISK_SIZE" \
  --shm-size "$SHM_SIZE" \
  --auth token \
  --token "$AUTH_TOKEN" \
  --subnet-id "$SUBNET_ID" \
  "${EXTRA_ARGS[@]}"

ENDPOINT_ID="$(nebius ai endpoint get-by-name --name "$ENDPOINT_NAME" --format json | jq -r '.metadata.id')"
ENDPOINT_URL="$(nebius ai endpoint get "$ENDPOINT_ID" --format json \
  | jq -r '.status.public_endpoints[]? | select(startswith("https://"))' | head -1)"

cat <<EOF

  Endpoint ID : $ENDPOINT_ID
  URL         : ${ENDPOINT_URL:-<pending — rerun: nebius ai endpoint get $ENDPOINT_ID>}
  Auth token  : $AUTH_TOKEN        <-- SAVE THIS

First boot downloads the full checkpoint from Hugging Face (tens of minutes).
Follow progress:   nebius ai endpoint logs $ENDPOINT_ID
Ready when logs show the server listening on 0.0.0.0:$PORT.
Until then requests return 502 from the proxy.
EOF

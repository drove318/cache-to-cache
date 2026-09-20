#!/usr/bin/env bash
# road-b/identity_proof.sh — B0: prove the mounted wire changed nothing.
#
# Two acts, run on either side of launch_wired.sh:
#
#   road-b/identity_proof.sh baseline    # against the plain server, first
#   GATE=closed road-b/launch_wired.sh   # relaunch with the connector riding
#   road-b/identity_proof.sh verify      # replay, and every byte must match
#
# Greedy (temperature 0), six faces of the model — fact, arithmetic, code,
# list, summary, translation. The verdict is byte equality of the servers
# own text, answer against answer; a sha rides each row so a drifted token
# has nowhere to sit. Exit 0 means the identity held; anything else says
# which face broke.
set -euo pipefail

PORT="${WIRED_PORT:-8888}"
MODEL="${WIRED_MODEL:-qwen3.8-flash-next}"
BASE="${BASE_DIR:-/home/drove/msg/baselines}"
PY="${C2C_VENV:-/home/drove/c2c-venv}/bin/python"
ACT="${1:-}"

[ -d "$BASE" ] || mkdir -p "$BASE"
[[ "$MODEL" =~ ^[a-z0-9._-]+$ ]] || { echo "refuse: model name ${MODEL@Q} carries shell-unsafe characters" >&2; exit 1; }

PROMPTS=(
    "Q: What is the capital of France? A:"
    "Q: What is 17 times 23? A:"
    "Write one python line that prints the 9th fibonacci number:"
    "Name the four seasons, one per line:"
    "Summarize in one sentence: the quick brown fox jumps over the lazy dog while the farmer watches:"
    "Translate to french: good morning, how are you today?"
)
MAXTOK=(12 10 40 24 32 20)

ask() {                                   # $1 index -> stdout, the servers text
    local i=$1 body
    body=$(printf '{"model": "%s", "prompt": %s, "max_tokens": %d, "temperature": 0}' \
        "$MODEL" "$("$PY" -c "import json,sys; print(json.dumps(sys.argv[1]))" "${PROMPTS[i]}")" "${MAXTOK[i]}")
    curl -s -m 180 "http://127.0.0.1:${PORT}/v1/completions" \
        -H 'Content-Type: application/json' -d "$body" |
        "$PY" -c 'import json,sys
raw = sys.stdin.read()
if not raw.strip():
    sys.stderr.write("no body at all (the server is down or still booting)\n"); raise SystemExit(4)
try:
    print(json.loads(raw)["choices"][0]["text"], end="")
except (KeyError, IndexError, ValueError) as exc:
    sys.stderr.write(f"the server answered without a choices list: {exc}\n"); raise SystemExit(3)'
}

case "$ACT" in
    baseline)
        echo "[identity] baseline: six greedy answers from the plain server at :${PORT}"
        for i in "${!PROMPTS[@]}"; do
            ask "$i" > "${BASE}/b0_${i}.ans" || { echo "[identity] the server failed prompt ${i}; the baseline is not whole" >&2; exit 1; }
            printf '  %d  %s  %s\n' "$i" "$(sha1sum "${BASE}/b0_${i}.ans" | cut -c1-12)" "${PROMPTS[i]:0:56}"
        done
        echo "[identity] kept under ${BASE}; now run: GATE=closed road-b/launch_wired.sh"
        ;;
    verify)
        [ -f "${BASE}/b0_0.ans" ] || { echo "[identity] no baseline at ${BASE} — run the baseline act against the plain server first" >&2; exit 1; }
        echo "[identity] verify: awaiting the server at :${PORT} (cold starts are minutes, not seconds)"
        for ((w = 0; w < ${READY_SECS:-900}; w += 10)); do
            curl -s -m 5 -o /dev/null "http://127.0.0.1:${PORT}/health" 2>/dev/null && break
            sleep 10
        done
        curl -s -m 5 -o /dev/null "http://127.0.0.1:${PORT}/health" 2>/dev/null || {
            echo "[identity] nothing answers on :${PORT} after $(( ${READY_SECS:-900} / 60 )) minutes — is the server up? (docker ps)" >&2
            exit 1
        }
        echo "[identity] the server speaks; replaying six greedy faces against the baseline"
        fails=0
        for i in "${!PROMPTS[@]}"; do
            ask "$i" > "${BASE}/wired_${i}.ans" || { echo "  ${i}  FAILED (the server would not answer)"; fails=$((fails + 1)); continue; }
            if cmp -s "${BASE}/b0_${i}.ans" "${BASE}/wired_${i}.ans"; then
                printf '  %d  identical  %s  %s\n' "$i" "$(sha1sum "${BASE}/wired_${i}.ans" | cut -c1-12)" "${PROMPTS[i]:0:56}"
            else
                printf '  %d  DRIFTED    %s vs %s  %s\n' "$i" \
                    "$(sha1sum "${BASE}/b0_${i}.ans" | cut -c1-8)" "$(sha1sum "${BASE}/wired_${i}.ans" | cut -c1-8)" "${PROMPTS[i]:0:48}"
                fails=$((fails + 1))
            fi
        done
        if [ "$fails" -eq 0 ]; then
            echo "[identity] verdict: the wire rode, the answers stood. B0 HOLDS."
            exit 0
        fi
        echo "[identity] verdict: ${fails} of ${#PROMPTS[@]} faces drifted — with GATE=closed that is a fault, not a flavor" >&2
        exit 1
        ;;
    *)
        echo "usage: road-b/identity_proof.sh baseline | verify" >&2
        exit 2
        ;;
esac

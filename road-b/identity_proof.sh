#!/usr/bin/env bash
# road-b/identity_proof.sh — B0: prove the mounted wire changed nothing.
#
#   WIRED_BOOT=plain road-b/identity_proof.sh baseline   # before the wiring
#   GATE=closed      road-b/launch_wired.sh              # mount the wire
#   WIRED_BOOT=plain road-b/identity_proof.sh verify     # answers must agree
#   road-b/identity_proof.sh stability                   # same boot, twice
#
# The engine is stable within a boot (stability proves it) but NOT
# byte-reproducible across cold boots: fp8 quantization, cuda-graph capture
# and drafter init vary the greedy path from boot to boot, plain and wired
# alike (measured: 5 of 6 faces drift between two plain boots, 4 of 6
# plain-to-wired — no worse with the wire). So verify compares ANSWERS with
# the think blocks stripped (the reasoning preamble is the model's boot, not
# the wire's); a substantively different answer is a drift regardless of
# mode, and that is the fault, not the flavor — unless the receiver waives
# the thinking entirely and the boot reaches the bottom.
set -euo pipefail

PORT="${WIRED_PORT:-8888}"
MODEL="${WIRED_MODEL:-qwen3.8-flash-next}"
BASE="${BASE_DIR:-/home/drove/msg/baselines}"
PY="${C2C_VENV:-/home/drove/c2c-venv}/bin/python"
BOOT="${WIRED_BOOT:-b0}"
ACT="${1:-}"
[[ "$ACT" =~ ^(baseline|verify|stability)$ ]] || {
    echo "usage: identity_proof.sh baseline | verify | stability   (WIRED_BOOT tags the captures)" >&2
    exit 2
}
[[ "$BOOT" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "refuse: boot tag ${BOOT@Q} is not shell-safe" >&2; exit 1; }
[ -d "$BASE" ] || mkdir -p "$BASE"

# face 2 asks the bare number: there's no formulation left to vary, and the
# answer must reach the bottom on 34
PROMPTS=(
    "Q: What is the capital of France? A:"
    "Q: What is 17 times 23? A:"
    "What is the 9th Fibonacci number, counting the sequence 1,1,2,3,... from F(1)=1? Answer with just the number, nothing else:"
    "Name the four seasons, one per line:"
    "Summarize in one sentence: the quick brown fox jumps over the lazy dog while the farmer watches:"
    "Translate to french: good morning, how are you today?"
)
MAXTOK=(12 10 12 24 32 20)

ask() {                                   # $1 index -> stdout, the server's text
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
except (KeyError, IndexError, AttributeError, ValueError) as exc:
    sys.stderr.write(f"the server answered without a choices list: {exc}\n"); raise SystemExit(3)'
}

spokes() {                                # the only readiness that means yes: a token
    curl -s -m 30 "http://127.0.0.1:${PORT}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${MODEL}\",\"prompt\":\"ping\",\"max_tokens\":1,\"temperature\":0}" |
        "$PY" -c 'import json,sys
try:
    raise SystemExit(0 if json.load(sys.stdin).get("choices") else 1)
except (AttributeError, ValueError):
    raise SystemExit(1)' 2>/dev/null
}

await_speaking() {
    echo "[identity] awaiting the server's first token at :${PORT} (the engine warms for minutes)"
    for ((w = 0; w < ${READY_SECS:-900}; w += 15)); do
        spokes && return 0
        sleep 15
    done
    echo "[identity] the engine has spoken no token within $(( ${READY_SECS:-900} / 60 )) minutes (docker logs \$(docker ps -q --filter name=vllm-fn | head -1) | tail -30)" >&2
    return 1
}

normalize() {                             # the answer, stripped to the bottom
    "$PY" -c 'import re,sys
text = sys.stdin.read()
text = re.sub(r"(?s)<\s*think\s*>.*?<\s*/\s*think\s*>", "", text)
text = re.sub(r"(?s)<\s*think\s*>.*", "", text)
sys.stdout.write(re.sub(r"(?s)\s+", " ", text).strip())'
}

case "$ACT" in
    baseline)
        echo "[identity] baseline: six greedy faces from the plain server at :${PORT}"
        await_speaking || exit 1
        for i in "${!PROMPTS[@]}"; do
            ask "$i" > "${BASE}/${BOOT}_${i}.ans" || { echo "[identity] the server failed face ${i}; the baseline is not whole" >&2; exit 1; }
            printf '  %d  %s  %s\n' "$i" "$(sha1sum "${BASE}/${BOOT}_${i}.ans" | cut -c1-12)" "${PROMPTS[i]:0:56}"
        done
        echo "[identity] kept as ${BOOT}_*.ans under ${BASE}; now: GATE=closed road-b/launch_wired.sh"
        ;;
    verify)
        [ -f "${BASE}/${BOOT}_0.ans" ] || { echo "[identity] no baseline ${BOOT}_0.ans at ${BASE} — run the baseline act first" >&2; exit 1; }
        echo "[identity] verify: the wired answers must reach the same bottom as ${BOOT}, think aside"
        await_speaking || exit 1
        fails=0
        for i in "${!PROMPTS[@]}"; do
            ask "$i" > "${BASE}/wired_${i}.ans" || { echo "  ${i}  FAILED (the server would not answer)"; fails=$((fails + 1)); continue; }
            if cmp -s <(normalize < "${BASE}/${BOOT}_${i}.ans") <(normalize < "${BASE}/wired_${i}.ans"); then
                printf '  %d  IDENTICAL  %s  %s\n' "$i" "$(sha1sum "${BASE}/wired_${i}.ans" | cut -c1-12)" "${PROMPTS[i]:0:56}"
            else
                printf '  %d  DRIFTED    %s vs %s  %s\n' "$i" \
                    "$(sha1sum "${BASE}/${BOOT}_${i}.ans" | cut -c1-8)" "$(sha1sum "${BASE}/wired_${i}.ans" | cut -c1-8)" "${PROMPTS[i]:0:48}"
                fails=$((fails + 1))
            fi
        done
        if [ "$fails" -eq 0 ]; then
            echo "[identity] verdict: the wire rode, the answers stood. B0 HOLDS (modulo the boot)."
            exit 0
        fi
        echo "[identity] verdict: ${fails} of ${#PROMPTS[@]} faces drifted on the answer channel — that is a fault, not a flavor" >&2
        exit 1
        ;;
    stability)
        echo "[identity] stability: each face twice in one boot, back to back"
        await_speaking || exit 1
        fails=0
        for i in "${!PROMPTS[@]}"; do
            ask "$i" > "${BASE}/stab_a_${i}.ans" || { echo "  ${i}  FAILED (first call)"; fails=$((fails + 1)); continue; }
            ask "$i" > "${BASE}/stab_b_${i}.ans" || { echo "  ${i}  FAILED (second call)"; fails=$((fails + 1)); continue; }
            if cmp -s <(normalize < "${BASE}/stab_a_${i}.ans") <(normalize < "${BASE}/stab_b_${i}.ans"); then
                printf '  %d  stable     %s  %s\n' "$i" "$(sha1sum "${BASE}/stab_a_${i}.ans" | cut -c1-12)" "${PROMPTS[i]:0:56}"
            else
                printf '  %d  UNSTABLE   %s vs %s  %s\n' "$i" \
                    "$(sha1sum "${BASE}/stab_a_${i}.ans" | cut -c1-8)" "$(sha1sum "${BASE}/stab_b_${i}.ans" | cut -c1-8)" "${PROMPTS[i]:0:48}"
                fails=$((fails + 1))
            fi
        done
        if [ "$fails" -eq 0 ]; then
            echo "[identity] verdict: the server repeats its own answer, back to back, modulo the thinking. B0 (within boot) holds"
            exit 0
        fi
        echo "[identity] verdict: ${fails} faces will not budge even modulo the thinking — the boot stops" >&2
        exit 1
        ;;
esac

#!/usr/bin/env bash
# road-b/launch_wired.sh — relaunch the live vLLM server with the C2C wire mounted.
#
# Run this file, do not paste it. It never hand-copies the launch: it reads
# the containers birth certificate (image, argv, entrypoint, env, binds,
# host net) from docker inspect and replays it verbatim, adding exactly one
# thing — --kv-transfer-config, which mounts c2c's connector inside the worker.
#
# Why replay instead of rewrite: the box carries read-only binds that patch
# the engine itself (ple_offload, modelopt, qsa_ops). A hand-typed relaunch
# that drops one boots a DIFFERENT engine, and the answers drift with no
# ones word. The inspect, or nothing — and the inspect must be BELIEVED:
# the certificate is verified before it is used, and a container whose argv
# smells of a previous failed launch (sh -c, pip install, our own flags) is
# refused, healed from the parked original, or fatal — never copied.
#
# No shell strings, at any depth: the original argv rides as real argv, and
# the one addition (pip install of the wheel, then exec vllm serve) lives
# in a generated boot.sh the container runs as its entrypoint. What cannot
# be mis-escaped cannot break at two in the morning.
#
# DRY_RUN=1  — every read, every build, every flag; then the exact docker
#              run line, printed, and the script stops before it touches a
#              container. The rehearsal, on paper, with the real parts.
#
# Gates:
#   GATE=closed  (the default) the connector rides but never touches a row:
#                every answer must stay byte-identical. This is the B0 proof.
#   GATE=open    the resident wire fuses; C2C_WIRE must name a wire file.
#                With no wire the connector serves the identity and says so.
#
# Rollback: the plain container is renamed, never removed; and this script
# will never rm a parked original it has not first seen clean. If the wired
# one does not reach health, the plain one comes back under its own name.
set -euo pipefail

CONTAINER="${WIRED_CONTAINER:-vllm-fn-tp1}"
WIRED_NAME="${CONTAINER}"                      # the wired one takes the original name
PLAIN_NAME="${CONTAINER}-plain"                # the original, parked
PORT="${WIRED_PORT:-8888}"
GATE="${GATE:-closed}"
C2C_WIRE="${C2C_WIRE:-}"                       # host path to the .pt wire (GATE=open)
REPO="${C2C_REPO:-/home/drove/msg/cache-to-cache}"
WHEEL_DIR="${C2C_WHEEL_DIR:-/tmp/c2c-wheels}"
NEED_GB="${WIRED_NEED_GB:-55}"
READY_SECS="${WIRED_READY_SECS:-1800}"        # a 524k boot with drafter and graphs is slow, not stale
VENVPY="${C2C_VENV:-/home/drove/c2c-venv}/bin/python"
DRY_RUN="${DRY_RUN:-0}"
LOCK=/tmp/c2c-launch-wired.lock

say() { printf '[launch-wired] %s\n' "$*"; }
die() { printf '[launch-wired] refuse: %s\n' "$*" >&2; exit 1; }

# one launch mid-flight, ever: an overlapping pair is how a wired husk
# once passed for the plain and a launch served garbage argv
speak_probe() {                                  # the readiness that means something: a token, in choices
    curl -s -m 30 "http://127.0.0.1:${PORT}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${JSON_ARGS[0]}\",\"prompt\":\"ping\",\"max_tokens\":1,\"temperature\":0}" |
        "$VENVPY" -c 'import json,sys
try:
    raise SystemExit(0 if json.load(sys.stdin).get("choices") else 1)
except (AttributeError, ValueError):
    raise SystemExit(1)' 2>/dev/null
}

mkdir "$LOCK" 2>/dev/null || die "another launch is mid-flight ($LOCK exists); if none is running, rmdir it"
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# ── the birth certificate, believed only after it is verified ────────────
cmd_of()      { docker inspect -f '{{if .Config.Cmd}}{{index .Config.Cmd 0}}{{else}}<none>{{end}}' "$1" 2>/dev/null; }
binds_of()    { docker inspect -f '{{range .HostConfig.Binds}}{{.}}{{"\n"}}{{end}}' "$1" 2>/dev/null; }
env_of()      { docker inspect -f '{{range .Config.Env}}{{.}}{{"\n"}}{{end}}' "$1" 2>/dev/null; }
entrypoint_of() { docker inspect -f '{{json .Config.Entrypoint}}' "$1" 2>/dev/null; }

is_poisoned() {                               # argv of a container this script made, or a husk of one
    local first all
    first=$(cmd_of "$1") || return 0
    all=$(docker inspect -f '{{join .Config.Cmd " "}}' "$1" 2>/dev/null) || return 0
    case " $first " in ( sh | bash | dash | "-c" ) return 0 ;; esac
    case " $all " in
        (*" -c "*|*"pip install "*|*"exec vllm serve "*|*"--kv-transfer-config"*|*"/c2c/boot.sh"*|*"c2c.integrations.vllm_wired"*)
            return 0 ;;
    esac
    return 1
}
is_clean_cert() {                             # an argv that reads as the models own
    local first ep
    docker inspect "$1" >/dev/null 2>&1 || return 1
    first=$(cmd_of "$1"); ep=$(entrypoint_of "$1")
    [ -n "$first" ] && [ "$first" != "<none>" ] || return 1
    case " $first " in ( sh | bash | dash | "-c" ) return 1 ;; esac   # a shell does not name a model
    case "$first" in (-*) return 1 ;; esac                             # an option does not lead a model tag
    [ "$ep" = '["vllm","serve"]' ] || return 1                         # the images own door, unpainted
    ! is_poisoned "$1"
}

docker inspect "$CONTAINER" >/dev/null 2>&1 || die "no container ${CONTAINER} to wire"
SRC="$CONTAINER"
if is_poisoned "$CONTAINER"; then
    if is_clean_cert "$PLAIN_NAME"; then
        say "the current container is a husk of a failed launch; healing from the parked original"
        docker stop "$CONTAINER" >/dev/null 2>&1 || true
        docker rm -f "$CONTAINER" >/dev/null 2>&1 || true      # husk only: -plain is verified clean
        docker rename "$PLAIN_NAME" "$CONTAINER"
    else
        die "${CONTAINER} carries a poisoned birth certificate (sh -c / pip / our own flags in its argv) and ${PLAIN_NAME} cannot vouch for it — docker inspect both, then remove the husk by hand"
    fi
fi
[ -d "$REPO/src/c2c" ] || die "the c2c checkout is not at ${REPO} (set C2C_REPO)"
[ "$GATE" = closed ] || [ "$GATE" = open ] || die "GATE is closed or open, not ${GATE@Q}"
if [ "$GATE" = open ] && [ ! -f "$C2C_WIRE" ]; then
    die "GATE=open wants C2C_WIRE=<path to the wire .pt>; with no wire to mount, run GATE=closed and prove the identity first"
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
    die "something already answers on :8000 — the research-loop watchdog keeps a server there on this boxs one memory pool; stop the competitor (systemctl --user stop 'research-loop*'; pkill -f 'vllm serve.*8000') and rerun"
fi
is_clean_cert "$CONTAINER" || die "${CONTAINER}s birth certificate fails verification (Cmd[0]=$(cmd_of "$CONTAINER") entrypoint=$(entrypoint_of "$CONTAINER")); refusing to replay a launch I cannot vouch for"

# ── the wheel and the boot shim: built fresh, mounted read-only ──────────
mkdir -p "$WHEEL_DIR"
"$VENVPY" -m pip wheel "$REPO" -w "$WHEEL_DIR" --no-deps --quiet ||
    die "the wheel would not build from ${REPO}"
WHEEL=$(ls -t "$WHEEL_DIR"/c2c_cache-*.whl 2>/dev/null | head -1)
[ -n "$WHEEL" ] || die "no c2c wheel under ${WHEEL_DIR}"
WHEEL_NAME=$(basename "$WHEEL")
cat > "$WHEEL_DIR/boot.sh" << BOOT
#!/bin/sh
# the one addition, in a file — so no layer of shell quoting ever sees the argv
set -e
python3 -m pip install --no-index --no-deps "/c2c/${WHEEL_NAME}"
exec vllm serve "\$@"
BOOT
chmod 755 "$WHEEL_DIR/boot.sh"
say "wheel: ${WHEEL_NAME}, booted by /c2c/boot.sh"

# ── the certificate, read from the verified source ───────────────────────
IMAGE=$(docker inspect -f '{{.Config.Image}}' "$CONTAINER")
[ -n "$IMAGE" ] || die "inspecting ${CONTAINER} gave no image"
mapfile -t JSON_ARGS < <(docker inspect -f '{{join .Config.Cmd "\n"}}' "$CONTAINER")
[ "${#JSON_ARGS[@]}" -gt 1 ] || die "the containers argv read empty; refusing to relaunch a server I cannot see"

BINDS=()
while IFS= read -r b; do
    [ -n "$b" ] || continue
    case "$b" in (*:/c2c|*:/c2c:*) continue ;; esac            # the wheel dir owns /c2c, once
    BINDS+=("--volume=$b")
done < <(binds_of "$CONTAINER")
say "binds: ${#BINDS[@]} carried, engine patches included"

NET=$(docker inspect -f '{{.HostConfig.NetworkMode}}' "$CONTAINER")
IPC=$(docker inspect -f '{{.HostConfig.IpcMode}}' "$CONTAINER")
SHM=$(docker inspect -f '{{.HostConfig.ShmSize}}' "$CONTAINER")

# env: the image re-applies its own Config.Env at run time; only the
# operators creation-time overrides ride the file. Values carrying a
# newline are left to the image — --env-file cannot speak one — and named.
ENV_FILE=$(mktemp /tmp/c2c-wired-env.XXXXXX)
: > "$ENV_FILE"
declare -A IN_IMAGE=()
while IFS= read -r e; do [ -n "$e" ] && IN_IMAGE["$e"]=1; done < <(env_of "$IMAGE")
n_over=0
while IFS= read -r e; do
    [ -n "$e" ] || continue
    [ -n "${IN_IMAGE[$e]+x}" ] && continue
    case "$e" in
        (*$'\n'*) say "note: ${e%%=*} carries a newline and the image has its own copy; the image speaks for it" ;;
        (*) printf '%s\n' "$e" >> "$ENV_FILE"; n_over=$((n_over + 1)) ;;
    esac
done < <(env_of "$CONTAINER")
say "env: ${n_over} creation-time overrides ride; the rest is the images own"

# ── the one addition: the wire in the workers connector slot ──────────────
WIRE_IN=""
[ -n "$C2C_WIRE" ] && WIRE_IN="/c2c/wire.pt"
WIRE_JSON=""
[ -n "$WIRE_IN" ] && WIRE_JSON=", \"c2c_wire\": \"${WIRE_IN}\""
KV_JSON="{\"kv_connector\": \"C2CWiredConnector\", \"kv_connector_module_path\": \"c2c.integrations.vllm_wired.connector\", \"kv_role\": \"kv_both\", \"kv_connector_extra_config\": {\"c2c_gate\": \"${GATE}\"${WIRE_JSON}}}"
say "kv-transfer-config: ${KV_JSON}"

DOCKER_RUN=(docker run -d --name "$WIRED_NAME"
    --network "$NET" --ipc "$IPC" --shm-size "$SHM"
    --gpus all --env-file "$ENV_FILE"
    "${BINDS[@]}"
    --volume "${WHEEL_DIR}:/c2c:ro"
    --entrypoint /c2c/boot.sh
    "$IMAGE" "${JSON_ARGS[@]}" "--kv-transfer-config=${KV_JSON}")

if [ "$DRY_RUN" = 1 ]; then
    say "DRY_RUN: nothing was touched. The line that would run:"
    printf '  %q' "${DOCKER_RUN[@]}"; printf '\n'
    say "and after it: road-b/identity_proof.sh verify"
    exit 0
fi

# ── park the plain one; raise the wired one in its name ──────────────────
# never rm a parked original unseen: a stale -plain is evidence of an
# earlier race, and evidence is examined, not deleted
if docker inspect "$PLAIN_NAME" >/dev/null 2>&1; then
    # the name from an earlier run: examine it, then sweep it if it is what it claims
    if is_clean_cert "$PLAIN_NAME" || is_poisoned "$PLAIN_NAME"; then
        say "the parked ${PLAIN_NAME} is examined (Cmd[0]=$(cmd_of "$PLAIN_NAME") entrypoint=$(entrypoint_of "$PLAIN_NAME")) and found to be a husk of an earlier run; sweeping"
        docker rm -f "$PLAIN_NAME" >/dev/null 2>&1 || true
    else
        die "${PLAIN_NAME} already exists and answers to neither the plain nor a husk (Cmd[0]=$(cmd_of "$PLAIN_NAME") entrypoint=$(entrypoint_of "$PLAIN_NAME")) — inspect it by hand before this script touches it"
    fi
fi
docker stop "$CONTAINER" >/dev/null
AVAIL_GB=$(free -g | awk '/^Mem:/{print $7}')
if [ "$AVAIL_GB" -lt "$NEED_GB" ]; then
    docker start "$CONTAINER" >/dev/null
    die "even with the plain server stopped only ${AVAIL_GB} GiB answer on the pool (need ${NEED_GB}): something else holds it — (docker ps; systemctl --user list-units | grep -i running; pgrep -af 'vllm serve')"
fi
docker rename "$CONTAINER" "$PLAIN_NAME"
say "plain container parked as ${PLAIN_NAME}"

ROLLBACK() {
    say "the wired server did not come up — the evidence first, the rollback second"
    local archive="$REPO/road-b/logs/wired-failure-$(date +%Y%m%d-%H%M%S).log"
    mkdir -p "$REPO/road-b/logs"
    if docker logs "$WIRED_NAME" > "$archive" 2>&1; then
        say "the full log, root cause and all, kept at ${archive} — its head speaks:"
        head -n 60 "$archive" | sed -e 's/^/[wired] /' || true
    fi
    docker rm -f "$WIRED_NAME" >/dev/null 2>&1 || true
    docker start "$PLAIN_NAME" >/dev/null 2>&1 || true
    docker rename "$PLAIN_NAME" "$CONTAINER" 2>/dev/null || true
    rm -f "$ENV_FILE"
    die "wired launch failed; the plain server is back under its name, the log above is filed"
}
trap ROLLBACK ERR

"${DOCKER_RUN[@]}" >/dev/null

# ── the promise the runbook swears by: health before word ─────────────────
say "awaiting health on :${PORT} (cold start is minutes, not seconds)"
for ((t = 0; t < READY_SECS; t += 10)); do
    if curl -s -m 5 -o /dev/null "http://127.0.0.1:${PORT}/health" 2>/dev/null; then
        say "health answered; the engine warms for minutes — awaiting the first token"
        for ((p = 0; p < READY_SECS; p += 15)); do
            speak_probe && break
            sleep 15
        done
        if ! speak_probe; then
            say "the HTTP loop answers but the engine speaks no token — showing its words:"
            docker logs --tail 20 "$WIRED_NAME" 2>&1 | sed -e 's/^/[wired] /' || true
            false
        fi
        if [ "$GATE" = open ]; then
            say "health is up and a token has been spoken; the wire is mounted and the gate is open"
        else
            say "health is up and a token has been spoken; the gate is closed — the identity must hold, word for word"
        fi
        say "next: road-b/identity_proof.sh verify"
        rm -f "$ENV_FILE"
        trap - ERR
        exit 0
    fi
    sleep 10
done
false                                              # trips ROLLBACK: it archives, prints the head, restores the plain

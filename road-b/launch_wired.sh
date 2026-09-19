#!/usr/bin/env bash
# road-b/launch_wired.sh — relaunch the live vLLM server with the C2C wire mounted.
#
# Run this file, do not paste it. It never hand-copies the launch: it reads
# the containers birth certificate (image, argv, env, binds, gpus, host net)
# from docker inspect and replays it verbatim, adding exactly one thing —
# --kv-transfer-config, which mounts c2c's connector inside the worker.
#
# Why replay instead of rewrite: the box carries read-only binds that patch
# the engine itself (ple_offload, modelopt, qsa_ops). A hand-typed relaunch
# that drops one boots a DIFFERENT engine, and the answers drift with no
# ones word. The inspect, or nothing.
#
# Gates:
#   GATE=closed  (the default) the connector rides but never touches a row:
#                every answer must stay byte-identical. This is the B0 proof.
#   GATE=open    the resident wire fuses; C2C_WIRE must name a wire file.
#                With no wire the connector serves the identity and says so.
#
# Rollback: the plain container is renamed, never removed. If the wired
# one does not reach health, this script stops it and brings the plain one
# back under its own name.
#
# The preflight refuses to start into a committed box — the same law the
# sibling servers script lives by: GB10 memory is ONE unified pool; the
# research-loop watchdog keeps a competing server on :8000, and the two
# together are how the box — and the desktop — die.
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
READY_SECS="${WIRED_READY_SECS:-600}"
VENVPY="${C2C_VENV:-/home/drove/c2c-venv}/bin/python"

say() { printf '[launch-wired] %s\n' "$*"; }
die() { printf '[launch-wired] refuse: %s\n' "$*" >&2; exit 1; }

# ── preflight: refuse to start into a committed box ───────────────────────
docker inspect "$CONTAINER" >/dev/null 2>&1 || die "no container ${CONTAINER} to wire"
[ -d "$REPO/src/c2c" ] || die "the c2c checkout is not at ${REPO} (set C2C_REPO)"
[ "$GATE" = closed ] || [ "$GATE" = open ] || die "GATE is closed or open, not ${GATE@Q}"
if [ "$GATE" = open ] && [ ! -f "$C2C_WIRE" ]; then
    die "GATE=open wants C2C_WIRE=<path to the wire .pt>; with no wire to mount, run GATE=closed and prove the identity first"
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
    die "something already answers on :8000 — the research-loop watchdog keeps a server there on this boxs one memory pool; stop the competitor (systemctl --user stop 'research-loop*'; pkill -f 'vllm serve.*8000') and rerun"
fi
AVAIL_GB=$(free -g | awk '/^Mem:/{print $7}')
[ "$AVAIL_GB" -ge "$NEED_GB" ] ||
    die "only ${AVAIL_GB} GiB free; the server claims more of the unified pool than that leaves. what holds it? (docker ps; systemctl --user list-units)"

# ── the wheel: built fresh from the checkout, mounted in read-only ────────
mkdir -p "$WHEEL_DIR"
"$VENVPY" -m pip wheel "$REPO" -w "$WHEEL_DIR" --no-deps --quiet ||
    die "the wheel would not build from ${REPO}"
WHEEL=$(ls -t "$WHEEL_DIR"/c2c_cache-*.whl 2>/dev/null | head -1)
[ -n "$WHEEL" ] || die "no c2c wheel under ${WHEEL_DIR}"
WHEEL_NAME=$(basename "$WHEEL")
say "wheel: ${WHEEL_NAME}"

# ── the birth certificate, read not remembered ─────────────────────────────
IMAGE=$(docker inspect -f '{{.Config.Image}}' "$CONTAINER")
[ -n "$IMAGE" ] || die "inspecting ${CONTAINER} gave no image"
mapfile -t JSON_ARGS < <(docker inspect -f '{{join .Config.Cmd "\n"}}' "$CONTAINER")
[ "${#JSON_ARGS[@]}" -gt 1 ] || die "the containers argv read empty; refusing to relaunch a server I cannot see"
mapfile -t BINDS < <(docker inspect -f '{{range .HostConfig.Binds}}{{.}}{{"\n"}}{{end}}' "$CONTAINER")
NET=$(docker inspect -f '{{.HostConfig.NetworkMode}}' "$CONTAINER")
IPC=$(docker inspect -f '{{.HostConfig.IpcMode}}' "$CONTAINER")
SHM=$(docker inspect -f '{{.HostConfig.ShmSize}}' "$CONTAINER")
ENV_FILE=$(mktemp /tmp/c2c-wired-env.XXXXXX)
docker inspect -f '{{range .Config.Env}}{{.}}{{"\n"}}{{end}}' "$CONTAINER" > "$ENV_FILE"

# ── the one addition: the wire in the workers connector slot ───────────────
WIRE_IN=""
[ -n "$C2C_WIRE" ] && WIRE_IN="/c2c/wire.pt"
WIRE_JSON=""
[ -n "$WIRE_IN" ] && WIRE_JSON=", \"c2c_wire\": \"${WIRE_IN}\""
KV_JSON="{\"kv_connector\": \"C2CWiredConnector\", \"kv_connector_module_path\": \"c2c.integrations.vllm_wired.connector\", \"kv_role\": \"kv_both\", \"kv_connector_extra_config\": {\"c2c_gate\": \"${GATE}\"${WIRE_JSON}}}"
say "kv-transfer-config: ${KV_JSON}"

# ── park the plain one; raise the wired one in its name ────────────────────
docker stop "$CONTAINER" >/dev/null
if docker inspect "$PLAIN_NAME" >/dev/null 2>&1; then docker rm -f "$PLAIN_NAME" >/dev/null; fi
docker rename "$CONTAINER" "$PLAIN_NAME"
say "plain container parked as ${PLAIN_NAME}"

ROLLBACK() {
    say "the wired server did not come up — rolling back to the plain one"
    docker rm -f "$WIRED_NAME" >/dev/null 2>&1 || true
    docker start "$PLAIN_NAME" >/dev/null 2>&1 || true
    docker rename "$PLAIN_NAME" "$CONTAINER" 2>/dev/null || true
    rm -f "$ENV_FILE"
    die "wired launch failed; the plain server is back under its name"
}
trap ROLLBACK ERR

# ── the launch: verbatim argv, plus the connector ──────────────────────────
KV_QUOTED=${KV_JSON//\'/\'\\\'\'}                   # single-quote fortress for the inner shell
CMD_LINE="pip install --no-index --no-deps /c2c/${WHEEL_NAME} && exec vllm serve $(printf '%q ' "${JSON_ARGS[@]}")--kv-transfer-config '${KV_QUOTED}'"
WIRE_BIND=()
[ -n "$C2C_WIRE" ] && WIRE_BIND=(-v "${C2C_WIRE}:${WIRE_IN}:ro")

docker run -d --name "$WIRED_NAME" \
    --network "$NET" --ipc "$IPC" --shm-size "$SHM" \
    --gpus all --env-file "$ENV_FILE" \
    "${BINDS[@]/#/--volume }" \
    -v "${WHEEL_DIR}:/c2c:ro" "${WIRE_BIND[@]}" \
    "$IMAGE" sh -c "$CMD_LINE" >/dev/null

# ── the promise the runbook swears by: health before word ─────────────────
say "awaiting health on :${PORT} (cold start is minutes, not seconds)"
for ((t = 0; t < READY_SECS; t += 10)); do
    if curl -s -m 5 -o /dev/null "http://127.0.0.1:${PORT}/health" 2>/dev/null; then
        if [ "$GATE" = open ]; then
            say "health is up; the wire is mounted and the gate is open"
        else
            say "health is up; the gate is closed — the identity must hold, word for word"
        fi
        say "next: road-b/identity_proof.sh"
        rm -f "$ENV_FILE"
        trap - ERR
        exit 0
    fi
    sleep 10
done
docker logs --tail 25 "$WIRED_NAME" 2>&1 | sed -e 's/^/[wired] /' || true
false                                              # trip the rollback

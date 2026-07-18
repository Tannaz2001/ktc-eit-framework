#!/usr/bin/env bash
set -euo pipefail

PORT="${STREAMLIT_SERVER_PORT:-8501}"
ADDRESS="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"

# outputs/ is a bind mount. If the host directory is owned by a different uid,
# the dashboard still starts but every benchmark and report export fails on
# write. Say so now rather than at the first run.
if [ ! -w /app/outputs ]; then
    echo "WARNING: /app/outputs is not writable by $(id -un) (uid $(id -u))." >&2
    echo "         Benchmarks and report exports will fail." >&2
    echo "         On the host:  chown -R 1000:1000 ./outputs" >&2
    echo >&2
fi

echo
echo "=================================================="
echo " KTC EIT Reconstruction Dashboard"
echo "=================================================="
echo
echo "   http://localhost:${PORT}"
echo
# The container's own IP (172.x) was previously printed here as "network
# access"; it is not reachable from other devices, so it is not shown. To
# expose the dashboard on the LAN, publish the port on the host instead.
echo "=================================================="
echo

exec streamlit run app.py \
    --server.port="${PORT}" \
    --server.address="${ADDRESS}" \
    --server.headless=true

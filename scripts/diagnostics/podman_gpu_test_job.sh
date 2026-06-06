#!/usr/bin/env bash
set -euo pipefail

# Ensure common system binaries are visible in non-interactive Slurm shells.
export PATH="/usr/bin:/bin:/opt/slurm/bin:${PATH}"

echo "=== command discovery ==="
echo "PATH=$PATH"
command -v nvidia-smi || true
command -v podman || true

NVIDIA_SMI_BIN="$(command -v nvidia-smi || true)"
PODMAN_BIN="$(command -v podman || true)"

if [[ -z "$NVIDIA_SMI_BIN" ]]; then
  echo "nvidia-smi not found on compute node"
  exit 127
fi

if [[ -z "$PODMAN_BIN" ]]; then
  echo "podman not found on compute node"
  exit 127
fi

echo "=== host node ==="
hostname

echo "=== host nvidia-smi ==="
"$NVIDIA_SMI_BIN" --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader

echo "=== podman setup ==="
export PODMAN_TMP_ROOT=/tmp/$USER-podman-root-ovl
export PODMAN_TMP_RUNROOT=/tmp/$USER-podman-runroot-ovl
mkdir -p "$PODMAN_TMP_ROOT" "$PODMAN_TMP_RUNROOT"
"$PODMAN_BIN" --storage-driver=overlay --storage-opt ignore_chown_errors=true --root "$PODMAN_TMP_ROOT" --runroot "$PODMAN_TMP_RUNROOT" --version

echo "=== podman gpu test: cdi nvidia.com/gpu=all ==="
set +e
"$PODMAN_BIN" --storage-driver=overlay --storage-opt ignore_chown_errors=true --root "$PODMAN_TMP_ROOT" --runroot "$PODMAN_TMP_RUNROOT" run --rm --pull=always --device nvidia.com/gpu=all docker.io/nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
RC1=$?
set -e
echo "CDI_RC=$RC1"

if [[ $RC1 -ne 0 ]]; then
  echo "=== podman gpu test: fallback host devices ==="
  set +e
  "$PODMAN_BIN" --storage-driver=overlay --storage-opt ignore_chown_errors=true --root "$PODMAN_TMP_ROOT" --runroot "$PODMAN_TMP_RUNROOT" run --rm --pull=always \
    --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device /dev/nvidia0 \
    docker.io/nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  RC2=$?
  set -e
  echo "FALLBACK_RC=$RC2"
fi

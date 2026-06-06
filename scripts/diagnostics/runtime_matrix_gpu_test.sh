#!/usr/bin/env bash
set -euo pipefail

echo "=== node ==="
hostname

echo "=== host nvidia-smi ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader
else
  echo "nvidia-smi not found"
fi

echo "=== runtime binaries ==="
for c in singularity apptainer enroot podman docker; do
  printf "%s -> " "$c"
  command -v "$c" || true
done

echo "=== runtime versions ==="
set +e
for c in singularity apptainer enroot podman docker; do
  if command -v "$c" >/dev/null 2>&1; then
    echo "--- $c --version ---"
    "$c" --version
    echo "RC=$?"
  fi
done
set -e

echo "=== singularity/apptainer gpu run tests ==="
if command -v singularity >/dev/null 2>&1; then
  set +e
  singularity exec --nv docker://nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi
  echo "SINGULARITY_RC=$?"
  set -e
fi

if command -v apptainer >/dev/null 2>&1; then
  set +e
  apptainer exec --nv docker://nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi
  echo "APPTAINER_RC=$?"
  set -e
fi

echo "=== enroot gpu run tests ==="
if command -v enroot >/dev/null 2>&1; then
  set +e
  enroot import docker://nvidia/cuda:12.3.0-base-ubuntu22.04
  echo "ENROOT_IMPORT_RC=$?"
  set -e
fi

echo "=== docker gpu run test ==="
if command -v docker >/dev/null 2>&1; then
  set +e
  docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi
  echo "DOCKER_RC=$?"
  set -e
fi

echo "=== done ==="

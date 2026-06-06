#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

cd "${repo_root}"

required_submodules=(
  "third_party/Audio-Maestro"
  "third_party/DeSTA2.5-Audio"
  "third_party/ToolRL"
)

for submodule in "${required_submodules[@]}"; do
  if [[ ! -d "${submodule}" ]]; then
    echo "Missing submodule directory: ${submodule}" >&2
    exit 1
  fi
done

echo "Current submodule status:"
git submodule status

check_patch() {
  local submodule="$1"
  local patch="$2"

  if [[ ! -f "${patch}" ]]; then
    echo "Missing patch file: ${patch}" >&2
    exit 1
  fi

  echo
  echo "Dry-run patch check:"
  echo "  submodule: ${submodule}"
  echo "  patch:     ${patch}"

  if git -C "${submodule}" apply --check "../../${patch}"; then
    echo "  result:    PASS"
    return 0
  else
    echo "  result:    FAIL" >&2
    echo "Patch dry-run failed: ${patch} -> ${submodule}" >&2
    return 1
  fi
}

failed=0

check_patch "third_party/Audio-Maestro" "patches/audio-maestro-main/tracked_source_only.patch" || failed=1
check_patch "third_party/DeSTA2.5-Audio" "patches/desta25-audio/tracked_changes.patch" || failed=1
check_patch "third_party/ToolRL" "patches/desta-grpo-toolrl/tracked_changes.patch" || failed=1

echo
if [[ "${failed}" -eq 0 ]]; then
  echo "All mapped patch dry-runs passed. No patches were applied."
else
  echo "One or more mapped patch dry-runs failed. No patches were applied." >&2
  exit 1
fi

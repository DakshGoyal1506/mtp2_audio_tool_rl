#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MTP2_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
EXTERNAL_ROOT="${MTP2_EXTERNAL_ROOT:-${HOME}/mtp2_external}"
MANIFEST_ROOT="${MTP2_MANIFEST_ROOT:-${EXTERNAL_ROOT}/manifests}"
OUTPUT_ROOT="${MTP2_OUTPUT_ROOT:-${EXTERNAL_ROOT}/outputs}"

CONFIG_PATH="${GRPO_TOOL_USE_CONFIG:-${REPO_ROOT}/configs/experiments/grpo_tool_use_smoke.yaml}"
MANIFEST_PATH="${GRPO_TOOL_USE_MANIFEST:-${MANIFEST_ROOT}/grpo_tool_use_smoke.jsonl}"
OUTPUT_DIR="${GRPO_TOOL_USE_OUTPUT_DIR:-${OUTPUT_ROOT}/grpo_tool_use_smoke}"
LOG_DIR="${GRPO_TOOL_USE_LOG_DIR:-${OUTPUT_ROOT}/logs/grpo_tool_use_smoke}"
RUN_ACTUAL_GRPO="${RUN_ACTUAL_GRPO:-0}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "GRPO tool-use debug runner"
echo "repo root:      ${REPO_ROOT}"
echo "config:         ${CONFIG_PATH}"
echo "manifest:       ${MANIFEST_PATH}"
echo "output dir:     ${OUTPUT_DIR}"
echo "log dir:        ${LOG_DIR}"
echo "RUN_ACTUAL_GRPO:${RUN_ACTUAL_GRPO}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "ERROR: config not found: ${CONFIG_PATH}" >&2
    exit 1
fi

echo
echo "Config preview:"
sed -n '1,160p' "${CONFIG_PATH}"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
    echo
    echo "ERROR: manifest not found: ${MANIFEST_PATH}" >&2
    echo "Create an external smoke manifest or set GRPO_TOOL_USE_MANIFEST." >&2
    exit 1
fi

echo
echo "Validating manifest..."
python3 "${REPO_ROOT}/scripts/data/validate_manifest.py" "${MANIFEST_PATH}"

echo
echo "Running tool-call and reward sanity checks..."
PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python3 - <<'PY'
from mtp2_audio_tool_rl.rewards.tool_use_reward import compute_tool_use_reward
from mtp2_audio_tool_rl.tool_calling.validate_tool_call import validate_tool_call

tool_call = {
    "tool_name": "speaker_diarization",
    "arguments": {"audio_path": "audio/sample_001.wav"},
}
validation = validate_tool_call(tool_call)
reward = compute_tool_use_reward(
    prediction="two speakers",
    gold_answer="Two speakers.",
    tool_call_valid=validation.valid,
    tool_call_present=True,
    tool_was_needed=True,
    tool_result_used=True,
)
print(f"tool_call_valid={validation.valid} error_code={validation.error_code}")
print(f"reward_breakdown={reward}")
PY

if [[ "${RUN_ACTUAL_GRPO}" != "1" ]]; then
    echo
    echo "Dry-run complete. No training was started."
    echo "To wire real training later, set RUN_ACTUAL_GRPO=1 after selecting the canonical entrypoint."
    echo "Candidate command placeholder:"
    echo "  python3 -m mtp2_audio_tool_rl.grpo.train_trl --config ${CONFIG_PATH}"
    exit 0
fi

echo
echo "RUN_ACTUAL_GRPO=1 was requested, but no safe canonical training entrypoint is wired yet." >&2
echo "Please connect this script to the chosen GRPO trainer after validating external assets." >&2
exit 2

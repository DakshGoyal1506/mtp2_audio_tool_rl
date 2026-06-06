# shellcheck shell=bash
# =============================================================================
# snapshot_expt.sh — reusable experiment-snapshot helper
#
# Usage (from a launcher script):
#
#     source scripts/snapshot_expt.sh
#     snapshot_expt "$EXPT_DIR" \
#         --config "$CONFIG" \
#         --extra  grpo_single_phase_llm/rewards_llm.py \
#         --extra  grpo_single_phase_llm/rewards_rule.py \
#         --code-dir grpo_single_phase_llm \
#         --code-dir grpo_single_phase \
#         --launcher "${BASH_SOURCE[0]}"
#
# What it does:
#   - Creates  $EXPT_DIR/  and  $EXPT_DIR/code/
#   - Copies the config + any --extra files to $EXPT_DIR/  (quick-diff layout)
#   - Mirrors each --code-dir into $EXPT_DIR/code/<dir>/ via rsync (cp fallback)
#   - Copies the calling launcher into $EXPT_DIR/code/
#   - Writes $EXPT_DIR/code/GIT_INFO.txt and working_tree.diff if git is present
#   - Makes $EXPT_DIR/code/ read-only so it cannot be overwritten mid-run
#
# Exports SNAPSHOT_DIR (= $EXPT_DIR/code) for the caller's convenience.
# =============================================================================

snapshot_expt() {
    local expt_dir="$1"; shift
    if [ -z "$expt_dir" ]; then
        echo "snapshot_expt: missing EXPT_DIR" >&2
        return 2
    fi

    local config=""
    local launcher=""
    local extras=()
    local code_dirs=()

    while [ "$#" -gt 0 ]; do
        case "$1" in
            --config)    config="$2"; shift 2 ;;
            --extra)     extras+=("$2"); shift 2 ;;
            --code-dir)  code_dirs+=("$2"); shift 2 ;;
            --launcher)  launcher="$2"; shift 2 ;;
            *)
                echo "snapshot_expt: unknown arg '$1'" >&2
                return 2
                ;;
        esac
    done

    local code_dir="$expt_dir/code"
    mkdir -p "$expt_dir" "$code_dir"

    # ── Top-level convenience copies (config + reward files etc.) ─────────
    if [ -n "$config" ] && [ -e "$config" ]; then
        cp "$config" "$expt_dir/"
    fi
    local f
    for f in "${extras[@]}"; do
        [ -e "$f" ] && cp "$f" "$expt_dir/"
    done

    # ── Full code snapshot ────────────────────────────────────────────────
    local rsync_excludes=(
        "--exclude=__pycache__"
        "--exclude=*.pyc"
        "--exclude=*.pyo"
        "--exclude=.ipynb_checkpoints"
        "--exclude=*.bak_*"
    )
    local d
    if command -v rsync >/dev/null 2>&1; then
        for d in "${code_dirs[@]}"; do
            [ -e "$d" ] || continue
            rsync -a "${rsync_excludes[@]}" "$d/" "$code_dir/$d/"
        done
    else
        for d in "${code_dirs[@]}"; do
            [ -e "$d" ] || continue
            mkdir -p "$code_dir/$d"
            cp -r "$d/." "$code_dir/$d/"
        done
        find "$code_dir" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
        find "$code_dir" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete 2>/dev/null || true
    fi

    # Archive the calling launcher.
    if [ -n "$launcher" ] && [ -e "$launcher" ]; then
        cp "$launcher" "$code_dir/$(basename "$launcher")" 2>/dev/null || true
    fi

    # ── Git provenance ────────────────────────────────────────────────────
    if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        {
            echo "# Git provenance for $expt_dir"
            echo "commit:   $(git rev-parse HEAD 2>/dev/null)"
            echo "branch:   $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
            echo "describe: $(git describe --always --dirty 2>/dev/null)"
            echo "date:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
            echo
            echo "# git status --short"
            git status --short 2>/dev/null || true
        } > "$code_dir/GIT_INFO.txt"
        git diff HEAD > "$code_dir/working_tree.diff" 2>/dev/null || true
    fi

    # Lock the snapshot so editors / reruns can't rewrite it.
    chmod -R a-w "$code_dir" 2>/dev/null || true

    export SNAPSHOT_DIR="$code_dir"
    echo "Experiment dir : $expt_dir"
    echo "Code snapshot  : $code_dir (read-only)"
}

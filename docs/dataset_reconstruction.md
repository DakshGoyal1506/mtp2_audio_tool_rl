# Dataset Reconstruction

This repository does not commit datasets, raw audio, copied audio subsets, embeddings, checkpoints, generated tool outputs, logs, or result dumps.

## Intended Flow

1. Prepare external dataset roots outside this repository.
2. Download or stage external datasets manually, or use existing scripts where applicable and permitted by the dataset license.
3. Run dataset mapping or construction scripts from `src/mtp2_audio_tool_rl/datasets/` and `scripts/data/`.
4. Copy only selected audio into an external export directory outside Git.
5. Write a JSONL manifest with paths relative to that export directory.
6. Validate the manifest structure.
7. Keep generated JSONL, audio, tool results, embeddings, and outputs outside Git unless they are tiny examples.

## External Roots

Use placeholder paths such as:

```bash
export MTP2_DATA_ROOT=/path/to/external/data
export MTP2_AUDIO_ROOT=/path/to/external/export/audio
export MTP2_OUTPUT_ROOT=/path/to/external/output
export MTP2_MANIFEST_ROOT=/path/to/external/manifests
```

Do not use full machine-specific workspace paths inside committed manifests.

## Manifest Validation

Validate the included tiny fake example:

```bash
python3 scripts/data/validate_manifest.py manifests/example_manifest.jsonl
```

Validate an external reconstructed manifest:

```bash
python3 scripts/data/validate_manifest.py /path/to/external/manifests/train.jsonl \
  --schema manifests/audio_manifest.schema.json
```

The validator checks required fields and rejects absolute paths in `audio_path` and `tool_result_path`. It does not require audio files to exist, and it does not import `datasets`, `torch`, `librosa`, or other heavy dependencies.

## Dataset Notes

Potential external sources include AudioSet, WaveCaps, MMAU-mini style evaluation data, DeSTA-style selected samples, and synthetic/tool-output datasets. Missing audio files can be skipped during reconstruction if only a few are missing; record skipped sample IDs in external logs or metadata.

The copied export folder should contain relative audio paths and a manifest, not full old workspace paths.

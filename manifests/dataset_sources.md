# Dataset Sources

Datasets are external to this repository and should not be committed to Git.

Raw audio, copied audio subsets, generated tool outputs, embeddings, checkpoints, logs, and result dumps should remain outside the repository. Commit only schemas, tiny example manifests, reconstruction notes, and small mapping scripts.

Expected external sources may include:

- AudioSet
- WaveCaps
- MMAU-mini style evaluation data
- DeSTA-style selected samples
- Synthetic/tool-output datasets generated from external audio

Manifests should use relative paths such as `external_audio/example_001.wav`, not full workspace or machine paths. A copied export folder should contain a JSONL manifest plus audio files addressed by relative paths. If only a few audio files are missing during reconstruction, downstream scripts may skip those rows and record the missing IDs.

This file does not provide download automation. Use existing scripts only where applicable, and review each dataset license and access policy before reconstruction.

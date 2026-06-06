#!/usr/bin/env python3
"""Pre-download all HuggingFace models so the job can run offline on compute nodes."""

import sys

def main():
    print("=== Downloading Whisper model ===")
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
    print("Whisper cached.\n")

    print("=== Downloading sound classifier ===")
    from transformers import pipeline
    pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")
    print("Sound classifier cached.\n")

    print("=== Downloading emotion model ===")
    from funasr import AutoModel
    AutoModel(model="iic/emotion2vec_plus_large", hub="hf")
    print("Emotion model cached.\n")

    print("=== Downloading pyannote diarization ===")
    from pyannote.audio import Pipeline
    Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    print("Diarization cached.\n")

    print("=== Downloading speechbrain speaker encoder ===")
    from speechbrain.inference.speaker import EncoderClassifier
    EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    print("Speaker encoder cached.\n")

    print("All models downloaded and cached!")

if __name__ == "__main__":
    main()

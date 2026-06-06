#########################################
# Audio Maestro: AudioCopilot Tool      #
# Speech, Sound, Music Analysis Toolkit #
# Author: Kuan-Yi Lee                   #
# Support Cache for Efficiency          #
#########################################

import os
os.environ["PYANNOTE_AUDIO_DISABLE_TORCHCODEC"] = "1"

import json
import asyncio
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import warnings
import torch
import torchaudio
from pathlib import Path
import tempfile
import subprocess
import re

# from pyannote.audio import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    pipeline
)
import textgrid
# import autochord
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")


class AudioCopilotTool:
    """Audio analysis toolkit with speech, sound, and music processing capabilities"""

    def __init__(self):
        """Initialize tool with lazy-loaded models"""
        # Use MPS for Mac GPU, CUDA for NVIDIA GPU, CPU otherwise
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        # Model references (loaded on first use)
        self._whisper_model = None
        self._whisper_processor = None
        self._emotion_model = None
        self._sound_classifier = None
        self._diarization_pipeline = None
        self._embedding_model = None

    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file"""
        try:
            return librosa.load(audio_path, sr=None)
        except Exception as e:
            raise Exception(f"Failed to load audio: {e}")

    def _get_whisper_model(self):
        """Lazy load Whisper model"""
        if self._whisper_model is None:
            try:
                print("Loading Whisper model...")
                self._whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
                self._whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
                # Keep model in float32 to avoid type mismatch errors
                self._whisper_model = self._whisper_model.to(self.device).float()
                print(f"Whisper loaded on {self.device} in float32")
            except Exception as e:
                print(f"Failed to load Whisper: {e}")
                self._whisper_model = None
                self._whisper_processor = None
        return self._whisper_model, self._whisper_processor

    def _get_emotion_model(self):
        """Lazy load emotion recognition model"""
        if self._emotion_model is None:
            try:
                print("Loading emotion model...")
                from funasr import AutoModel
                self._emotion_model = AutoModel(
                    model="iic/emotion2vec_plus_large",
                    hub="hf",
                    device=str(self.device)
                )
                print(f"Emotion model loaded on {self.device}")
            except Exception as e:
                print(f"Failed to load emotion model: {e}")
                self._emotion_model = None
        return self._emotion_model

    def _get_sound_classifier(self):
        """Lazy load sound classification model"""
        if self._sound_classifier is None:
            try:
                print("Loading sound classifier...")
                self._sound_classifier = pipeline(
                    "audio-classification",
                    model="MIT/ast-finetuned-audioset-10-10-0.4593",
                    device=self.device
                )
                print("Sound classifier loaded")
            except Exception as e:
                print(f"Failed to load sound classifier: {e}")
                self._sound_classifier = None
        return self._sound_classifier

    # Speech Recognition

    async def speech_recognition(self, audio_path: str) -> Dict:
        """Convert speech to text using Whisper"""
        audio, sr = self._load_audio(audio_path)

        try:
            # Check cache
            base_name = Path(audio_path).stem
            cache_dir = Path(audio_path).parent / "transcriptions"
            cache_dir.mkdir(exist_ok=True)
            cache_file = cache_dir / f"{base_name}.txt"

            if cache_file.exists():
                print(f"Using cached transcription: {cache_file}")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    transcription = f.read().strip()

                if not transcription:
                    return {
                        'function': 'speech_recognition',
                        'error': 'Empty transcript',
                        'text': '',
                        'language': 'auto-detected',
                    }

                return {
                    'function': 'speech_recognition',
                    'text': transcription,
                    'language': 'auto-detected',
                }

            # Run recognition
            model, processor = self._get_whisper_model()

            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000

            inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
            # Ensure inputs are in float32 and on the correct device
            inputs = {k: v.to(self.device).float() for k, v in inputs.items()}

            with torch.no_grad():
                if "attention_mask" in inputs:
                    predicted_ids = model.generate(
                        inputs["input_features"],
                        attention_mask=inputs["attention_mask"]
                    )
                else:
                    predicted_ids = model.generate(inputs["input_features"])

            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True, language="en")

            # Cache result
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(transcription[0] if transcription else "")

            if not transcription or not transcription[0].strip():
                return {
                    'function': 'speech_recognition',
                    'error': 'Empty transcript',
                    'text': '',
                    'language': 'auto-detected',
                }

            return {
                'function': 'speech_recognition',
                'text': transcription[0],
                'language': 'auto-detected',
            }

        except Exception as e:
            return {
                'function': 'speech_recognition',
                'error': str(e),
            }

    async def speaker_diarization(self, audio_path: str) -> Dict:
        """Identify and separate different speakers in audio"""
        try:
            # Initialize models
            if self._diarization_pipeline is None:
                self._diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1"
                ).to(self.device)
                print(f"Diarization pipeline loaded on {self.device}")

            if self._embedding_model is None:
                from speechbrain.inference.speaker import EncoderClassifier
                self._embedding_model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": str(self.device)}
                )
                print(f"Embedding model loaded on {self.device}")

            def extract_segment(path, start, end):
                """Extract audio segment"""
                audio, sr = librosa.load(path, sr=None)
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                return audio[start_sample:end_sample]

            def run_diarization(path):
                """Run diarization and merge similar speakers"""
                diarization = self._diarization_pipeline(path)

                # Extract embeddings per speaker
                speaker_embeddings = {}
                for segment, track, speaker in diarization.itertracks(yield_label=True):
                    if segment.end - segment.start < 0.6:
                        continue

                    seg_waveform = extract_segment(path, segment.start, segment.end)
                    try:
                        if isinstance(seg_waveform, np.ndarray):
                            seg_waveform = torch.from_numpy(seg_waveform).float()

                        if seg_waveform.dim() == 1:
                            seg_waveform = seg_waveform.unsqueeze(0)

                        seg_waveform = seg_waveform.to(self.device)
                        emb = self._embedding_model(seg_waveform)
                    except Exception as e:
                        print(f"Embedding extraction failed for {speaker}: {e}")
                        continue

                    if isinstance(emb, tuple):
                        emb = emb[0]
                    emb = emb.detach().cpu().numpy()
                    if emb.ndim > 1:
                        emb = emb.squeeze()

                    if speaker not in speaker_embeddings:
                        speaker_embeddings[speaker] = []
                    speaker_embeddings[speaker].append(emb)

                # Average embeddings
                for speaker in speaker_embeddings:
                    speaker_embeddings[speaker] = np.mean(speaker_embeddings[speaker], axis=0)

                # Handle empty case - no speakers detected
                if len(speaker_embeddings) == 0:
                    return []

                # Compute similarity and merge
                speakers = list(speaker_embeddings.keys())
                emb_matrix = np.array([speaker_embeddings[s] for s in speakers])

                # Handle single speaker case
                if len(speakers) == 1:
                    sim_matrix = np.array([[1.0]])
                else:
                    sim_matrix = cosine_similarity(emb_matrix)

                threshold = 0.10
                merged = {}
                for i, spk1 in enumerate(speakers):
                    for j, spk2 in enumerate(speakers):
                        if i >= j:
                            continue
                        if sim_matrix[i, j] > threshold:
                            merged[spk2] = spk1

                # Build segments with merged labels
                segments = []
                for segment, track, speaker in diarization.itertracks(yield_label=True):
                    if speaker in merged:
                        speaker = merged[speaker]
                    segments.append({
                        'start': segment.start,
                        'end': segment.end,
                        'speaker': speaker,
                        'duration': segment.end - segment.start
                    })

                return segments

            segments = await asyncio.to_thread(run_diarization, audio_path)

            # Handle empty segments - no speakers detected
            if not segments:
                return {
                    'function': 'speaker_diarization',
                    'segments': [],
                    'speakers': [],
                    'num_speakers': 0,
                    'total_duration': 0,
                    'note': 'No speakers detected in audio'
                }

            # Map to ordinal names
            speaker_first_appearance = {}
            for segment in segments:
                speaker = segment['speaker']
                if speaker not in speaker_first_appearance:
                    speaker_first_appearance[speaker] = segment['start']

            sorted_speakers = sorted(speaker_first_appearance.items(), key=lambda x: x[1])
            speaker_mapping = {}
            ordinals = ["first", "second", "third", "fourth", "fifth"]

            for i, (original_speaker, _) in enumerate(sorted_speakers):
                if i < len(ordinals):
                    speaker_mapping[original_speaker] = f"{ordinals[i]} speaker"
                else:
                    speaker_mapping[original_speaker] = f"{i + 1}th speaker"

            for segment in segments:
                segment['speaker'] = speaker_mapping[segment['speaker']]

            speakers = list(set([seg['speaker'] for seg in segments]))
            total_duration = sum([seg['duration'] for seg in segments])

            return {
                'segments': segments,
                'speakers': speakers,
                'num_speakers': len(speakers),
                'total_duration': total_duration,
            }

        except Exception as e:
            return {
                'function': 'speaker_diarization',
                'error': str(e),
            }

    async def emotion_recognition(self, audio_path: str) -> Dict:
        """Detect emotion changes for each speaker"""
        try:
            # Get speaker segments first
            diarization_result = await self.speaker_diarization(audio_path)
            if 'error' in diarization_result:
                return {'error': f"Diarization failed: {diarization_result['error']}"}

            def extract_segment(path, start, end):
                """Extract audio segment"""
                audio, sr = librosa.load(path, sr=None)
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                return audio[start_sample:end_sample]

            def detect_emotion(audio_segment):
                """Detect emotion in segment"""
                emotion_model = self._get_emotion_model()
                if emotion_model is None:
                    return {"neutral": 0.5}

                # Pad short segments
                if len(audio_segment) < 8000:
                    padding_needed = 8000 - len(audio_segment)
                    audio_segment = np.pad(audio_segment, (0, padding_needed), mode='constant')

                try:
                    audio_16k = librosa.resample(audio_segment, orig_sr=22050, target_sr=16000)
                    results = emotion_model.generate(audio_16k)

                    if isinstance(results, list) and len(results) > 0:
                        entry = results[0]
                        if 'scores' in entry and 'labels' in entry:
                            scores = entry['scores']
                            labels = entry['labels']
                            # Filter out neutral and disgust
                            labels_scores = [
                                (label, score) for label, score in zip(labels, scores)
                                if label not in ['中立/neutral', '厌恶/disgusted']
                            ]
                            return {label: float(score) for label, score in labels_scores}
                    return {"neutral": 0.5}
                except Exception as e:
                    print(f"Emotion detection error: {e}")
                    return {"neutral": 0.5}

            # Analyze emotions per speaker
            speaker_emotions = {}
            emotion_flips = []

            for segment in diarization_result['segments']:
                speaker = segment['speaker']
                start_time = segment['start']
                end_time = segment['end']

                if end_time - start_time < 0.1:
                    continue

                segment_audio = await asyncio.to_thread(
                    extract_segment, audio_path, start_time, end_time
                )

                emotions = await asyncio.to_thread(detect_emotion, segment_audio)

                if speaker not in speaker_emotions:
                    speaker_emotions[speaker] = []

                dominant_emotion = max(emotions, key=emotions.get)
                speaker_emotions[speaker].append({
                    'start': start_time,
                    'end': end_time,
                    'emotion': dominant_emotion,
                })

            # Detect emotion flips
            for speaker, emotions in speaker_emotions.items():
                for i in range(1, len(emotions)):
                    prev_emotion = emotions[i-1]['emotion']
                    curr_emotion = emotions[i]['emotion']

                    if prev_emotion != curr_emotion and \
                       prev_emotion != '中立/neutral' and curr_emotion != '中立/neutral':
                        emotion_flips.append({
                            'speaker': speaker,
                            'flip_time': emotions[i]['start'],
                            'from_emotion': prev_emotion,
                            'to_emotion': curr_emotion,
                        })

            num_speakers_with_flips = len(set(flip['speaker'] for flip in emotion_flips))

            return {
                'speaker_emotions': speaker_emotions,
                'emotion_flips': emotion_flips,
                'num_flips': len(emotion_flips),
                'num_speakers_with_flips': num_speakers_with_flips,
            }

        except Exception as e:
            return {
                'function': 'emotion_recognition',
                'error': str(e),
            }

    def run_mfa(self, audio_path: str, transcript: str, mfa_bin="mfa") -> str:
        """Run Montreal Forced Aligner"""
        try:
            if not transcript.strip():
                raise ValueError("Empty transcript - cannot run MFA")

            # Clean transcript
            cleaned_transcript = transcript.lower().strip()
            cleaned_transcript = re.sub(r"[^\w\s']", " ", cleaned_transcript)
            cleaned_transcript = re.sub(r'\s+', ' ', cleaned_transcript).strip()

            if not cleaned_transcript:
                raise ValueError("Transcript empty after cleaning")

            print(f"Cleaned transcript: '{cleaned_transcript}'")

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                input_dir = temp_path / "input"
                input_dir.mkdir(exist_ok=True)

                # Copy audio
                audio_file = Path(audio_path)
                base_name = audio_file.stem
                input_audio = input_dir / f"{base_name}.wav"

                import shutil
                shutil.copy2(audio_path, input_audio)

                # Write transcript
                txt_path = input_dir / f"{base_name}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_transcript)

                output_dir = temp_path / "output"
                output_dir.mkdir(exist_ok=True)

                # Check cache
                permanent_output = Path(audio_path).parent / "mfa_output"
                tg_file = permanent_output / f"{base_name}.TextGrid"
                permanent_output.mkdir(exist_ok=True)

                if tg_file.exists():
                    print(f"Using cached TextGrid: {tg_file}")
                    return str(tg_file)

                # Run MFA
                mfa_cmd = [
                    mfa_bin, "align",
                    str(input_dir),
                    "english_us_arpa",
                    "english_us_arpa",
                    str(output_dir),
                    "--beam", "100",
                    "--retry_beam", "400",
                    "--clean",
                    "--num_jobs", "32",  # Use 32 CPU cores
                    "--verbose",
                    "--include_original_text"
                ]

                print(f"Running MFA: {' '.join(mfa_cmd)}")
                result = subprocess.run(mfa_cmd, capture_output=True, text=True, check=False)

                if result.returncode != 0:
                    print("MFA failed, retrying with larger beam...")
                    mfa_cmd_retry = [
                        mfa_bin, "align",
                        str(input_dir),
                        "english_us_arpa",
                        "english_us_arpa",
                        str(output_dir),
                        "--beam", "200",
                        "--retry_beam", "800",
                        "--clean",
                        "--num_jobs", "32",  # Use 4 CPU cores
                        "--verbose"
                    ]

                    result = subprocess.run(mfa_cmd_retry, capture_output=True, text=True, check=False)

                    if result.returncode != 0:
                        raise RuntimeError(f"MFA failed: {result.stderr}")

                # Find TextGrid
                tg_file = output_dir / f"{base_name}.TextGrid"
                if not tg_file.exists():
                    tg_files = list(output_dir.glob("*.TextGrid"))
                    if tg_files:
                        tg_file = tg_files[0]
                    else:
                        raise FileNotFoundError(f"No TextGrid in {output_dir}")

                # Copy to permanent location
                final_tg_path = permanent_output / f"{base_name}.TextGrid"
                shutil.copy2(tg_file, final_tg_path)

                print(f"MFA completed: {final_tg_path}")
                return str(final_tg_path)

        except Exception as e:
            print(f"MFA error: {e}")
            raise

    def _parse_textgrid(self, tg_path: str):
        """Parse TextGrid for word and phoneme info"""
        tg = textgrid.TextGrid.fromFile(tg_path)

        results = []
        words_tier = None
        phones_tier = None

        for tier in tg.tiers:
            tier_name = tier.name.lower()
            if "word" in tier_name:
                words_tier = tier
            elif "phone" in tier_name:
                phones_tier = tier

        if not words_tier:
            print("No words tier in TextGrid")
            return []

        # Extract words
        for interval in words_tier.intervals:
            word = interval.mark.strip()
            if word and word != "<sil>" and word != "":
                word_info = {
                    "word": word,
                    "start": float(interval.minTime),
                    "end": float(interval.maxTime),
                    "phonemes": []
                }
                results.append(word_info)

        # Extract phonemes
        if phones_tier and results:
            current_word_idx = 0
            for interval in phones_tier.intervals:
                phone = interval.mark.strip()
                if phone and phone != "<sil>" and phone != "":
                    stress = None
                    if phone[-1].isdigit():
                        stress = int(phone[-1])

                    phone_info = {
                        "phoneme": phone,
                        "stress": stress,
                        "start": float(interval.minTime),
                        "end": float(interval.maxTime)
                    }

                    phone_start = float(interval.minTime)
                    while (current_word_idx < len(results) - 1 and
                           phone_start >= results[current_word_idx]["end"]):
                        current_word_idx += 1

                    if current_word_idx < len(results):
                        results[current_word_idx]["phonemes"].append(phone_info)

        print(f"Parsed {len(results)} words from TextGrid")
        return results

    def _simplify_stress_results(self, word_alignments):
        """Convert phoneme data to word-level stress classification"""
        stressed_words = []
        unstressed_words = []

        for word_info in word_alignments:
            word = word_info['word'].strip()

            if not word or word == '<eps>':
                continue

            has_stressed = False
            has_unstressed = False

            if word_info.get('phonemes'):
                for phoneme in word_info['phonemes']:
                    stress_level = phoneme.get('stress')
                    if stress_level is not None:
                        if stress_level >= 1:
                            has_stressed = True
                        elif stress_level == 0:
                            has_unstressed = True

            if has_stressed:
                stressed_words.append(word)
            if has_unstressed:
                unstressed_words.append(word)

        return {
            "has_stressed_phoneme_list": stressed_words,
            "has_unstressed_phoneme_list": unstressed_words,
            "stressed_phoneme_words_count": len(stressed_words),
            "unstressed_phoneme_words_count": len(unstressed_words)
        }

    async def stressed_analysis(self, audio_path: str, transcript: str = None):
        """Analyze speech stress patterns

        Args:
            audio_path: Path to the audio file
            transcript: Optional pre-generated transcript. If provided, skips speech recognition.
        """
        print(f"Starting stress analysis: {audio_path}")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        try:
            # Use provided transcript or run speech recognition
            if transcript is not None and transcript.strip():
                print(f"Using pre-generated transcript")
                transcript = transcript.strip()
            else:
                # Get transcript via speech recognition
                speech_result = await self.speech_recognition(audio_path)

                if 'error' in speech_result or not speech_result.get('text', '').strip():
                    return {
                        "function": "stressed_analysis",
                        "error": speech_result.get('error', 'Empty transcript'),
                        "status": "failed"
                    }

                transcript = speech_result.get('text', '').strip()

            # Run MFA
            tg_path = self.run_mfa(audio_path, transcript)
            word_alignments = self._parse_textgrid(tg_path)
            stress_analysis = self._simplify_stress_results(word_alignments)

            return {
                "function": "stressed_analysis",
                "has_stressed_phoneme_list": stress_analysis["has_stressed_phoneme_list"],
                "has_unstressed_phoneme_list": stress_analysis["has_unstressed_phoneme_list"],
                "stressed_phoneme_words_count": stress_analysis["stressed_phoneme_words_count"],
                "unstressed_phoneme_words_count": stress_analysis["unstressed_phoneme_words_count"],
                "hint": "Choose closest match if count doesn't match options exactly",
                "transcript": transcript,
            }

        except Exception as e:
            print(f"Stress analysis error: {e}")
            return {
                "function": "stressed_analysis",
                "error": str(e),
                "status": "failed"
            }

    async def sound_classification(self, audio_path: str, categories: List[str] = None) -> Dict:
        """Classify environmental sounds"""
        audio, sr = self._load_audio(audio_path)

        try:
            sound_classifier = self._get_sound_classifier()
            result = sound_classifier(audio, sampling_rate=sr)

            if isinstance(result, list) and len(result) > 0:
                return {
                    'function': 'sound_classification',
                    'all_predictions': result[:5],
                    'Hint': 'Classification may not be perfectly accurate. Consider both audio content and predictions.',
                }

        except Exception as e:
            return {
                'function': 'sound_classification',
                'error': str(e),
            }

    async def sound_duration_analysis(self, audio_path: str, categories: List[str] = None,
                                     window_size=1.0, hop_size=0.5) -> Dict:
        """Analyze duration of detected sound events"""
        audio, sr = self._load_audio(audio_path)
        total_samples = len(audio)
        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)

        sound_classifier = self._get_sound_classifier()
        label_durations = {}

        for start in range(0, total_samples, hop_samples):
            end = min(start + window_samples, total_samples)
            segment = audio[start:end]

            try:
                result = sound_classifier(segment, sampling_rate=sr)
                if isinstance(result, list) and len(result) > 0:
                    top_label = max(result, key=lambda x: x['score'])['label']

                    if categories is None or top_label in categories:
                        duration_sec = (end - start) / sr
                        label_durations[top_label] = label_durations.get(top_label, 0.0) + duration_sec
            except Exception as e:
                print(f"Error in segment {start}-{end}: {e}")

        all_predictions = [{"label": k, "duration": v} for k, v in label_durations.items()]
        all_predictions.sort(key=lambda x: x['duration'], reverse=True)

        return {
            "function": "sound_duration_analysis",
            "duration_per_label": all_predictions,
            "Hint": "Results may not be highly accurate. Consider audio content alongside predictions.",
        }

    # async def chord_recognition(self, audio_path: str) -> dict:
    #     """Recognize chords using autochord"""
    #     try:
    #         base_name = Path(audio_path).stem
    #         lab_file = Path(audio_path).parent / f"{base_name}_chords.lab"

    #         if lab_file.exists():
    #             print(f"Using cached chords: {lab_file}")
    #         else:
    #             await asyncio.to_thread(autochord.recognize, audio_path, lab_fn=lab_file)

    #         # Parse lab file
    #         chord_changes = []
    #         with open(lab_file, 'r') as f:
    #             for line in f:
    #                 parts = line.strip().split()
    #                 if len(parts) == 3:
    #                     start, end, chord = parts
    #                     chord_changes.append({
    #                         "time": float(start),
    #                         "chord": chord,
    #                     })

    #         chord_changes_str = [
    #             f"{c['time']:.2f}s: {c['chord']}" for c in chord_changes[:100]
    #         ]

    #         return {
    #             "function": "chord_recognition",
    #             "chord_changes": chord_changes_str,
    #         }

    #     except Exception as e:
    #         return {
    #             'function': 'chord_recognition',
    #             'error': str(e),
    #         }

    async def genre_analysis(self, audio_path: str) -> Dict:
        """Analyze music genre using instrument detection and audio features"""
        try:
            sound_classifier = self._get_sound_classifier()
            results = sound_classifier(audio_path)

            # Filter and get top 3
            results = [r for r in results if r['label'] != 'music']
            results = sorted(results, key=lambda x: x['score'], reverse=True)[:3]
            results = [{"label": r['label'], "score": float(r['score'])} for r in results]

            audio_features = await self.get_audio_features(audio_path)

            return {
                "function": "genre_analysis",
                "instrument": results,
                "volume": audio_features['volume'],
                "pitch": audio_features['pitch'],
                "timbre": audio_features['timbre'],
                "tempo": audio_features['tempo'],
                "Hint": "Detect genre based on audio features and classification results.",
            }

        except Exception as e:
            return {
                'function': 'genre_analysis',
                'error': str(e),
            }

    async def get_audio_features(self, audio_path: str):
        """Extract basic audio features"""
        y, sr = librosa.load(audio_path)

        # Volume (RMS energy)
        rms = float(np.mean(librosa.feature.rms(y=y)))

        # Pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        avg_pitch = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0

        # Timbre (spectral centroid)
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)

        return {
            "volume": rms,
            "pitch": avg_pitch,
            "timbre": centroid,
            "tempo": tempo
        }

    async def speech_to_noise_ratio(self, audio_path: str) -> Dict:
        """Calculate speech-to-noise ratio"""
        try:
            y, sr = self._load_audio(audio_path)
            signal_power = np.mean(y**2)
            noise_power = np.var(y - np.mean(y))
            snr = 10 * np.log10(signal_power / noise_power)

            return {
                'function': 'speech_to_noise_ratio',
                'snr': float(snr),
            }
        except Exception as e:
            return {
                'function': 'speech_to_noise_ratio',
                'error': str(e),
            }

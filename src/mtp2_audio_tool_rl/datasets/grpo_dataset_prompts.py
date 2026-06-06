"""Prompt templates for dynamic GRPO multiple-choice dataset generation from DeSTA-AQA5M"""

from typing import List


_OUTPUT_SCHEMA = """
You MUST respond with a valid JSON object with these fields.
{
  "id": "the exact audio_path/id provided in METADATA",
  "reasoning": "Think step-by-step and slowly formulate the question, plausible distractors, and the correct answer based ONLY on the provided metadata",
  "question": "the generated multiple-choice question",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct_answer": "the text of the correct option",
  "tool": "the target tool provided in the prompt"
}
"""

_QUESTION_STYLE = """
QUESTION STYLE: Generate questions that perfectly mimic the style of the MMAU benchmark.
MMAU questions are **extremely direct, simple, clear, and objective**.

**CRITICAL: DIVERSITY IN QUESTION FORMULATION**
DO NOT start every question with "Based on the provided audio," or "From the given utterance,".
Vary your sentence structure naturally. Examples of diverse real MMAU-style questions:
- "Count the number of words that contain at least one stressed phoneme."
- "Identify the source of the ticking sound."
- "What is the primary instrument playing in this clip?"
- "Which of the following best describes the acoustic environment?"
- "Determine the gender of the speaker in this recording."
- "How many distinct speakers can be heard?"

IMPORTANT REQUIREMENTS:
1. **NO HALLUCINATION:** You are generating a question and its answer based *only* on what could be definitively known if you could listen to the audio described in the metadata.
2. The question must implicitly require the use of the targeted tool to answer it.
3. The question must be short and direct. DO NOT write overly verbose, "pseudo-intellectual", or "AI-sounding" questions.
4. DO NOT mention the tool name or any tool directly in the question text.
5. The correct answer must follow logically from the provided Audio description.
6. The 3 incorrect options must be plausible but definitively wrong.
7. Provide EXACTLY 4 string options in the `options` array.
"""


def build_prompt(metadata: dict, target_tool: str) -> str:
    """Generate a multiple-choice question for a specific target tool."""

    return f"""You are a dataset generator for training audio understanding models.

Given the following audio metadata and a target tool, generate a SINGLE difficult multiple-choice question that would require that tool to answer.

## AUDIO METADATA
- Audio id: {metadata['id']}
- Audio description: {metadata.get('seed_description', '')}

## TARGET TOOL
{target_tool}: {TOOL_DESCRIPTIONS[target_tool]}

{_QUESTION_STYLE}

{_OUTPUT_SCHEMA}
"""

# Tool descriptions mapping
TOOL_DESCRIPTIONS = {
    "speech_recognition": "Transcribe speech to text with word-level timestamps. Use for questions about what was said, transcript, words spoken.",
    "speaker_diarization": "Identify different speakers in audio with timestamps. Use for questions about number of speakers, who is speaking, speaker changes.",
    "emotion_recognition": "Detect emotions in speech (happy, sad, angry, etc.). Use for questions about mood, emotion, feeling in voice.",
    "get_audio_features": "Extract audio features (volume, pitch, tempo, beats, timbre). Use for questions about loudness, pitch, tempo, musical features, or speaker gender.",
    "sound_classification": "Classify sounds and audio events (e.g., dog barking, car horn, music). Use for questions about identifying what sound/noise is heard.",
    "sound_duration_analysis": "Analyze duration and timing of sound events. Use for questions about how long sounds last, timing, duration of events.",
    "speech_to_noise_ratio": "Calculate the speech-to-noise ratio (SNR) of audio. Use for questions about audio quality, background noise, clarity.",
    "stressed_analysis": "Analyze stress patterns in speech. Use for questions about stress, emphasis, or prosodic patterns.",
    "chord_recognition": "Recognize musical chords in audio. Use for questions about chords, harmony, musical structure.",
    "genre_analysis": "Analyze music genre and instruments. Use for questions about genre, instruments, musical style.",
}

# All tool names
ALL_TOOLS = list(TOOL_DESCRIPTIONS.keys())

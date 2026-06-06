import requests
import json

prompt = """\
You are an expert evaluator scoring an AI assistant's performance on audio-based multiple-choice questions.

## Context
Question: What is the tempo?
Choices:
  - 1: Fast
  - 2: Slow
Gold Answer: Fast

## Assistant's Raw Completion:
```xml
<think>The audio plays a fast drum beat.</think>
<answer>Fast</answer>
```

## Extracted Final Answer: 
Fast

## Scoring Instructions
Score 1-5. Avoid "3" (Neutral) unless the performance is strictly mediocre. If it is "good but not perfect," use 4. If it is "poor but has content," use 2.

1. **think (Reasoning)**
   **CRITICAL:** Evaluate thinking independently of the final answer. Reward reasoning that describes the sound and explores the logical direction, even if the final conclusion or chosen answer is wrong. Do NOT apply rigid keyword checklists (like demanding "pitch" or "timbre"). Any valid description of what is heard (e.g., "loud crash", "dramatic sound") counts as an audio observation.
   - 1: Placeholder, empty, or completely ignores the audio context.
   - 2: Non-descriptive reasoning. Merely states a choice without describing the sound itself (e.g., "based on the audio, I pick A").
   - 3: Basic description. Mentions some audio events (e.g., "loud sound", "people talking") but the logical connection to the choices is weak, or it stops short of applying the evidence.
   - 4: Strong reasoning. Clearly describes the acoustic properties or sound events heard in the audio (e.g., loudness, tone, pacing, instruments, voices) and effectively uses them to deduce or eliminate options, moving in a reasonable logical direction.
   - 5: Exceptional reasoning. Thoroughly analyzes the acoustic evidence, explicitly connects the observations to the underlying context, and systematically evaluates the options based on those sound observations.

2. **answer (Correctness)**
   - 1: Incorrect.
   - 5: Exactly matches the gold answer (including logic and label).

3. **format (Structure)**
   - 1: Missing <think> or <answer> tags.
   - 3: Tags present but contains text outside of the XML structure.
   - 5: Perfect XML compliance.

Output ONLY a JSON block matching this exact schema:
```json
{
  "reasoning": "your reasoning string here",
  "think": <1-5 integer>,
  "answer": <1-5 integer>,
  "format": <1-5 integer>
}
```
"""

payload = {
    "model": "Qwen/Qwen3.5-27B",
    "messages": [
        {
            "role": "system", 
            "content": "You are a precise scoring assistant. Respond in JSON. Provide your evaluation in the 'reasoning' field before providing the numeric scores."
        },
        {"role": "user", "content": prompt},
    ],
    "max_completion_tokens": 1000,
    "temperature": 0.2,
    "top_p": 0.95
}

try:
    resp = requests.post("http://192.168.1.30:8000/v1/chat/completions", json=payload, timeout=60)
    try:
        print("RESPONSE:", json.dumps(resp.json()['choices'][0]['message']['content'], indent=2))
    except Exception as e:
        print("TEXT:", resp.text)
except Exception as e:
    print("ERROR:", e)

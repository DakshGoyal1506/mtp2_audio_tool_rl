#!/usr/bin/env python3
"""Diagnostic: verify embed injection produces coherent output (not gibberish).

Tests:
1. Token-level alignment: placeholders land at correct positions
2. Embed injection: replacing placeholders with precomputed qformer embeddings
3. Generation: single-sample greedy decode to verify coherent output
"""
import torch, json, os, sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE_ROOT = os.environ.get("MTP2_WORKSPACE_ROOT")
DESTA_REPO = os.environ.get("MTP2_DESTA_REPO")

if not DESTA_REPO and WORKSPACE_ROOT:
    DESTA_REPO = os.path.join(WORKSPACE_ROOT, "DeSTA2.5-Audio")
if not DESTA_REPO:
    DESTA_REPO = os.path.join(REPO_ROOT, "third_party", "DeSTA2.5-Audio")

sys.path.insert(0, os.path.join(REPO_ROOT, "src", "mtp2_audio_tool_rl"))
if os.path.isdir(DESTA_REPO):
    sys.path.insert(0, DESTA_REPO)

from transformers import AutoTokenizer
from desta.models.modeling_desta25 import _prepare_audio_context_and_start_positions

print("="*60)
print("DIAGNOSTIC: Embedding injection alignment check")
print("="*60)

tok = AutoTokenizer.from_pretrained('DeSTA-ntu/Llama-3.1-8B-Instruct', local_files_only=True)
tok.pad_token = tok.eos_token
tok.pad_token_id = tok.eos_token_id
tok.padding_side = 'left'
tok.add_tokens(['<|AUDIO|>'])

# Load one sample
with open('mmau-test-mini.json') as f:
    data = json.load(f)
sample = data[0]
audio_id = sample['audio_id']
filename = os.path.basename(audio_id)
embed_name = os.path.splitext(filename)[0] + '_embed.pt'
ep = os.path.join('precomputed_embeds', embed_name)
print(f"\naudio_id: {audio_id}")
print(f"embed_path: {ep}  exists={os.path.exists(ep)}")

raw = torch.load(ep, map_location='cpu', weights_only=False)
print(f"embed: qformer={raw['qformer'].shape} dtype={raw['qformer'].dtype}, vad={raw['vad']}")

audio_size = raw['qformer'].size(0)  # 64
vad = bool(raw['vad'])

# Use actual cached transcription if available
sr_output = {}
cached_file = 'mmau-test-mini-cached.json'
if os.path.exists(cached_file):
    with open(cached_file) as f:
        cached_data = json.load(f)
    for c in cached_data:
        if c.get('audio_id') == audio_id:
            sr_output = c.get('cached_tool_outputs', {}).get('speech_recognition', {})
            break
transcription = sr_output.get('text') if isinstance(sr_output, dict) else None
if not vad:
    transcription = None
trans_size = len(tok.tokenize(transcription, add_special_tokens=False)) if transcription and transcription.strip() else 0
print(f"transcription: {repr(transcription)}")
print(f"audio_size={audio_size}, transcription_size={trans_size}")

# ---- Test 1: Tokenisation alignment ----
print("\n" + "="*60)
print("TEST 1: Placeholder token alignment")
print("="*60)

messages = [
    {'role': 'system', 'content': 'You are an audio analysis assistant.'},
    {'role': 'user', 'content': '<|AUDIO|>\nWhat do you hear?'},
]
ctx = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ctx = ctx.replace('<|AUDIO|>', '<start_audio><|AUDIO|><end_audio>')
tokens = tok.tokenize(ctx)
print(f"Tokenized length (before expansion): {len(tokens)}")

out_tokens, spos = _prepare_audio_context_and_start_positions(
    token_list=tokens, audio_locator='<|AUDIO|>',
    audio_size_list=[audio_size], transcription_size_list=[trans_size],
    placeholder_token='<|reserved_special_token_87|>')
print(f"After expansion: {len(out_tokens)} tokens")
print(f"Start positions: {spos}")

ctx_str = tok.convert_tokens_to_string(out_tokens)
ids = tok(ctx_str, return_tensors='pt', add_special_tokens=False).input_ids
sp = spos[0]
ph = tok.convert_tokens_to_ids('<|reserved_special_token_87|>')
total_embed_size = audio_size + trans_size
region = ids[0, sp:sp+total_embed_size]
all_ph = (region == ph).all().item()
print(f"input_ids shape: {ids.shape}")
print(f"placeholder_id: {ph}")
print(f"All {total_embed_size} target positions are placeholder? {all_ph}")
print(f"Round-trip token count match? {len(out_tokens) == ids.shape[1]} ({len(out_tokens)} vs {ids.shape[1]})")

if len(out_tokens) != ids.shape[1]:
    print("*** CRITICAL MISMATCH: convert_tokens_to_string -> re-tokenize changes length!")
    retok = tok.convert_ids_to_tokens(ids[0].tolist())
    for i, (a, b) in enumerate(zip(out_tokens, retok)):
        if a != b:
            print(f"    first diff at pos {i}: original='{a}' vs retokenized='{b}'")
            break

# ---- Test 2: Full model inference with embed injection ----
print("\n" + "="*60)
print("TEST 2: Model inference with precomputed embed")
print("="*60)

from grpo.modeling_grpo import GRPODeSTA25AudioModel

# Load model
GRPODeSTA25AudioModel.skip_perception = True
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
model = GRPODeSTA25AudioModel.from_pretrained('DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B', torch_dtype=torch.bfloat16)
model._setup_generation()
model = model.to('cuda')
model.eval()

question_str = sample['question']
choices_str = '\n'.join(f'  - {c}' for c in sample['choices'])

# Direct generate via model.generate() — the "known working" path
print("\n--- A) Direct model.generate() path ---")
messages_direct = [
    {'role': 'system', 'content': 'You are an audio analysis assistant.'},
    {
        'role': 'user',
        'content': f'<|AUDIO|>\nQuestion: {question_str}\n\nAnswer options:\n{choices_str}',
        'audios': [{'audio': os.path.join('.', audio_id.lstrip('./')), 'text': transcription, 'embed_path': ep}],
    },
]
with torch.inference_mode():
    out = model.generate(messages=messages_direct, do_sample=False, max_new_tokens=256, temperature=1.0, top_p=1.0)
print(f"Output text: {out.text[0][:500]}")

# Now test the GRPO path: manual tokenize + _prepare_inputs_for_llm + llm.generate
print("\n--- B) Manual embed injection path (GRPO training path) ---")
audio_context = tok.apply_chat_template(messages_direct, tokenize=False, add_generation_prompt=True)
audio_context = audio_context.replace('<|AUDIO|>', '<start_audio><|AUDIO|><end_audio>')
atokens = tok.tokenize(audio_context)
atokens, aspos = _prepare_audio_context_and_start_positions(
    token_list=atokens, audio_locator='<|AUDIO|>',
    audio_size_list=[audio_size], transcription_size_list=[trans_size],
    placeholder_token='<|reserved_special_token_87|>')
actx_str = tok.convert_tokens_to_string(atokens)
actx_inputs = tok(actx_str, return_tensors='pt', add_special_tokens=False)
input_ids = actx_inputs.input_ids.to('cuda')
attention_mask = actx_inputs.attention_mask.to('cuda')

batch_start_positions = [(0, aspos[0])]
trans_str = transcription if (trans_size > 0 and transcription) else ' '
trans_ids = tok.encode(trans_str, add_special_tokens=False, return_tensors='pt').long().to('cuda')

batch_features = torch.zeros((1, 1, 1), device='cuda', dtype=torch.bfloat16)

inputs_embeds = model._prepare_inputs_for_llm(
    input_ids=input_ids,
    attention_mask=attention_mask,
    batch_features=batch_features,
    batch_transcription_ids=[trans_ids],
    batch_start_positions=batch_start_positions,
    precomputed_embeds=[ep],
)

print(f"inputs_embeds shape: {inputs_embeds.shape}")

# Check: are the embedded positions actually different from the placeholder embedding?
ph_embedding = model.llm_model.model.embed_tokens(torch.tensor([ph], device='cuda'))
injected_region = inputs_embeds[0, aspos[0]:aspos[0]+audio_size]
mean_diff = (injected_region.float() - ph_embedding.float()).abs().mean().item()
print(f"Mean |injected - placeholder_embed|: {mean_diff:.4f} (should be >> 0)")

# Generate
with torch.inference_mode():
    gen_ids = model.llm_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=256,
        pad_token_id=tok.pad_token_id,
        eos_token_id=[128001, 128008, 128009],
    )
gen_text = tok.decode(gen_ids[0], skip_special_tokens=True)
print(f"Output text: {gen_text[:500]}")

# ---- Test 3: text-only (no audio) — should always be coherent ----
print("\n--- C) Text-only baseline (no audio embed, no placeholder) ---")
text_messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': 'What is 2+2?'},
]
text_ctx = tok.apply_chat_template(text_messages, tokenize=False, add_generation_prompt=True)
text_ids = tok(text_ctx, return_tensors='pt', add_special_tokens=False).input_ids.to('cuda')
with torch.inference_mode():
    text_gen = model.llm_model.generate(
        text_ids,
        do_sample=False,
        max_new_tokens=64,
        pad_token_id=tok.pad_token_id,
        eos_token_id=[128001, 128008, 128009],
    )
text_out = tok.decode(text_gen[0][text_ids.shape[1]:], skip_special_tokens=True)
print(f"Output text: {text_out[:200]}")

# ---- Test D: model.generate() with num_return_sequences=G and temp=1.1 (GRPO conditions) ----
print("\n--- D) model.generate(num_return_sequences=16, temperature=1.1) ---")
with torch.inference_mode():
    out_g = model.generate(
        messages=messages_direct,
        do_sample=True,
        temperature=1.1,
        top_p=0.95,
        max_new_tokens=256,
        num_return_sequences=4,  # use 4 to avoid OOM on 1 GPU
    )
for i, txt in enumerate(out_g.text[:4]):
    print(f"  [{i}]: {txt[:200]}")

# ---- Test E: Check eos_token_id in llm config and generation_config ----
print("\n--- E) Token config check ---")
print(f"model.config.eos_token_id = {model.config.eos_token_id}")
print(f"model.llm_model.config.eos_token_id = {model.llm_model.config.eos_token_id}")
if hasattr(model.llm_model, 'generation_config') and model.llm_model.generation_config:
    print(f"model.llm_model.generation_config.eos_token_id = {model.llm_model.generation_config.eos_token_id}")
else:
    print("model.llm_model.generation_config is None or missing")
print(f"tokenizer.eos_token_id = {tok.eos_token_id}")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)

#!/usr/bin/env python3
"""Quick diagnostic: check tokenisation + placeholder alignment."""
import torch, json, os, sys

# Copy the function directly to avoid heavy desta imports
def _prepare_audio_context_and_start_positions(token_list, audio_locator, audio_size_list, transcription_size_list, placeholder_token):
    result = []
    start_positions = []
    for x in token_list:
        if x == audio_locator:
            transcription_size = transcription_size_list.pop(0)
            audio_size = audio_size_list.pop(0)
            start_positions.append(len(result))
            result.extend([placeholder_token] * audio_size)
            result.extend([placeholder_token] * transcription_size)
        else:
            result.append(x)
    return result, start_positions

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('DeSTA-ntu/Llama-3.1-8B-Instruct')
tok.add_tokens(['<|AUDIO|>'])

# Load one sample
with open('mmau-test-mini.json') as f:
    data = json.load(f)
sample = data[0]
audio_id = sample['audio_id']
ep = f'precomputed_embeds/{audio_id}_embed.pt'  
raw = torch.load(ep, map_location='cpu', weights_only=False)
print(f'embed: qformer={raw["qformer"].shape}, vad={raw["vad"]}')

audio_size = raw['qformer'].size(0)
vad = bool(raw['vad'])
trans_size = 0  # vad=False

messages = [
  {'role':'system','content':'Test system prompt.'},
  {'role':'user','content':'<|AUDIO|>\nWhat do you hear?'}
]
ctx = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ctx = ctx.replace('<|AUDIO|>', '<start_audio><|AUDIO|><end_audio>')
tokens = tok.tokenize(ctx)
out_tokens, spos = _prepare_audio_context_and_start_positions(
    token_list=tokens, audio_locator='<|AUDIO|>',
    audio_size_list=[audio_size], transcription_size_list=[trans_size],
    placeholder_token='<|reserved_special_token_87|>')
ctx_str = tok.convert_tokens_to_string(out_tokens)
ids = tok(ctx_str, return_tensors='pt', add_special_tokens=False).input_ids
sp = spos[0]
ph = tok.convert_tokens_to_ids('<|reserved_special_token_87|>')
region = ids[0, sp:sp+audio_size]
print(f'start_pos={sp}  ids.shape={ids.shape}  placeholder_id={ph}')
print(f'all_placeholder_in_region={((region==ph).all().item())}')
print(f'region_unique={region.unique().tolist()}')
if sp > 0:
    print(f'token before region: {ids[0,sp-1].item()} = {repr(tok.decode([ids[0,sp-1].item()]))}')
if sp+audio_size < ids.shape[1]:
    print(f'token after region: {ids[0,sp+audio_size].item()} = {repr(tok.decode([ids[0,sp+audio_size].item()]))}')

# Now check: does convert_tokens_to_string -> re-tokenize preserve positions?
print(f'\nlen(out_tokens)={len(out_tokens)}  len(ids[0])={ids.shape[1]}')
if len(out_tokens) != ids.shape[1]:
    print('*** MISMATCH: token-level vs re-tokenized lengths differ!')
    print(f'    out_tokens has {len(out_tokens)} tokens, but re-tokenized has {ids.shape[1]}')
    # Find where divergence starts
    retok = tok.convert_ids_to_tokens(ids[0].tolist())
    for i, (a, b) in enumerate(zip(out_tokens, retok)):
        if a != b:
            print(f'    first diff at pos {i}: out_tokens={repr(a)} vs retokenized={repr(b)}')
            break
else:
    print('OK: token count matches after round-trip')

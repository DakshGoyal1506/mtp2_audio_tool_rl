"""
desta_vllm: vLLM-based inference helpers for DeSTA2.5-Audio with precomputed
embeddings.
"""

from .engine import DeSTAVLLMEngine
from .embed_utils import (
    build_prompt_embeds,
    load_embed_cache,
    resolve_embed_path,
)

__all__ = [
    "DeSTAVLLMEngine",
    "load_embed_cache",
    "build_prompt_embeds",
    "resolve_embed_path",
]

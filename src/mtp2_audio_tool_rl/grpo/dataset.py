"""
GRPO Dataset utilities for MMAU tool-use training.

Provides:
  - split_mmau_dataset(): stratified 10/10/80 train/eval/test split by task
  - GRPOAudioDataset:     PyTorch Dataset wrapping split items with cached tools
"""

import json
import os
import random
import logging
from typing import Dict, List, Optional, Tuple

import sys
from torch.utils.data import Dataset

# reuse load_cached_tools from existing code
_scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from tool_execute import load_cached_tools  # noqa: E402

logger = logging.getLogger(__name__)


def split_mmau_dataset(
    data_file: str,
    splits_dir: str,
    train_ratio: float = 0.10,
    eval_ratio: float = 0.10,
    seed: int = 42,
    force_resplit: bool = False,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Stratified 10/10/80 split of MMAU data by task (sound, speech, music).

    If split files already exist in splits_dir and force_resplit=False,
    the existing splits are loaded and returned (deterministic across runs).

    Args:
        data_file:    Path to mmau-test-mini-cached.json.
        splits_dir:   Directory to save/load train.json, eval.json, test.json.
        train_ratio:  Fraction of each task to use for training (default 0.10).
        eval_ratio:   Fraction of each task to use for eval (default 0.10).
        seed:         Random seed for reproducibility.
        force_resplit: Re-compute splits even if files exist.

    Returns:
        (train_items, eval_items, test_items) — lists of dataset dicts.
    """
    os.makedirs(splits_dir, exist_ok=True)

    train_path = os.path.join(splits_dir, "train.json")
    eval_path  = os.path.join(splits_dir, "eval.json")
    test_path  = os.path.join(splits_dir, "test.json")

    if not force_resplit and all(os.path.exists(p) for p in [train_path, eval_path, test_path]):
        logger.info(f"Loading existing splits from {splits_dir}")
        with open(train_path) as f: train_items = json.load(f)
        with open(eval_path)  as f: eval_items  = json.load(f)
        with open(test_path)  as f: test_items  = json.load(f)
        logger.info(
            f"Loaded splits — train: {len(train_items)}, "
            f"eval: {len(eval_items)}, test: {len(test_items)}"
        )
        return train_items, eval_items, test_items

    # Load full dataset
    logger.info(f"Loading dataset from {data_file}")
    with open(data_file, "r") as f:
        all_data = json.load(f)

    # Filter out items missing required fields
    all_data = [
        item for item in all_data
        if item.get("task") and item.get("question") and item.get("answer")
    ]
    logger.info(f"Total valid items: {len(all_data)}")

    # Group by task
    tasks: Dict[str, List[dict]] = {}
    for item in all_data:
        t = item["task"]
        tasks.setdefault(t, []).append(item)

    rng = random.Random(seed)

    train_items, eval_items, test_items = [], [], []

    for task_name, items in tasks.items():
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, round(n * train_ratio))
        n_eval  = max(1, round(n * eval_ratio))

        # Ensure train and eval don't overlap or exceed total
        n_eval = min(n_eval, n - n_train)

        t_train = items[:n_train]
        t_eval  = items[n_train:n_train + n_eval]
        t_test  = items[n_train + n_eval:]

        logger.info(
            f"  task={task_name}: total={n}, "
            f"train={len(t_train)}, eval={len(t_eval)}, test={len(t_test)}"
        )
        train_items.extend(t_train)
        eval_items.extend(t_eval)
        test_items.extend(t_test)

    # Save
    with open(train_path, "w") as f: json.dump(train_items, f, indent=2)
    with open(eval_path,  "w") as f: json.dump(eval_items,  f, indent=2)
    with open(test_path,  "w") as f: json.dump(test_items,  f, indent=2)

    logger.info(
        f"Saved splits → train: {len(train_items)}, "
        f"eval: {len(eval_items)}, test: {len(test_items)}"
    )
    return train_items, eval_items, test_items


class GRPOAudioDataset(Dataset):
    """
    PyTorch Dataset for GRPO training over MMAU audio items.

    Each __getitem__ returns a dict with:
        audio_path          (str)  — absolute path to .wav file
        question            (str)  — question text
        choices             (list) — answer option strings
        gold_answer         (str)  — correct answer (exact option text)
        cached_tool_outputs (dict) — {tool_name: result_dict} from dataset
        task                (str)  — 'sound' | 'speech' | 'music'
        id                  (str)  — item UUID
    """

    def __init__(
        self,
        items: List[dict],
        audio_root: Optional[str] = None,
        precomputed_embed_dir: Optional[str] = None,
    ):
        """
        Args:
            items:      List of dataset dicts (from split_mmau_dataset).
            audio_root: Optional root directory to resolve relative audio paths.
            precomputed_embed_dir: Directory containing precomputed .pt tensor features.
        """
        self.items = items
        self.audio_root = audio_root
        self.precomputed_embed_dir = precomputed_embed_dir

        # Build cached tool lookup: audio_id → tool_outputs
        self.cached_tools = load_cached_tools(items)
        logger.info(
            f"GRPOAudioDataset: {len(self.items)} items, "
            f"cached tools for {len(self.cached_tools)} audio files"
        )

    def _resolve_audio_path(self, audio_id: str) -> str:
        if self.audio_root:
            # Strip leading ./ if present
            rel = audio_id.lstrip("./")
            return os.path.join(self.audio_root, rel)
        return audio_id

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        audio_id = item.get("audio_id", "")
        audio_path = self._resolve_audio_path(audio_id)
        
        precomputed_embed_path = None
        if self.precomputed_embed_dir and audio_id:
            filename = os.path.basename(audio_id)
            embed_name = os.path.splitext(filename)[0] + "_embed.pt"
            candidate_path = os.path.join(self.precomputed_embed_dir, embed_name)
            if os.path.exists(candidate_path):
                precomputed_embed_path = candidate_path
            
        return {
            "audio_path":          audio_path,
            "precomputed_embed":   precomputed_embed_path,
            "question":            item.get("question", ""),
            "choices":             item.get("choices", []),
            "gold_answer":         item.get("answer", ""),
            "cached_tool_outputs": self.cached_tools.get(item.get("audio_id", ""), {}),
            "task":                item.get("task", ""),
            "category":            item.get("category", ""),
            "sub_category":        item.get("sub-category", ""),
            "difficulty":          item.get("difficulty", ""),
            "id":                  item.get("id", ""),
        }

"""Dataset and collation for Task-Adaptive Prefix Masked Language Modeling.

Stage 1 adapts the mmBERT backbone on the target prefix prompt:
    [CLS] <marker> [MASK]...[MASK] <marker> Context sentence... [SEP]

The MLM head reconstructs the original target word from the masked prefix tokens.
To predict the word at the prefix, the backbone is forced to attend to the context
sentence, transforming the prefix slot into a task-specific Context Sink.

Stochastic 80/10/10 Denoising:
    - 80% of tokens in the target word span are replaced with [MASK]
    - 10% are kept as the real token (identity reconstruction to prevent train/downstream discrepancy)
    - 10% are replaced with a random token from the vocabulary

Labels:
    - True token IDs at the target word span positions
    - -100 (ignore index) at all other positions (CLS, markers, context, SEP, padding)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .targets import MARKER_CODE, TARGETS


class PrefixMLMDataset(Dataset):
    """Generates prefix-masked sequences for mmBERT task adaptation."""

    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        tokenizer: Any,
        targets: Sequence[str] = TARGETS,
        max_len: int = 256,
        mask_prob: float = 0.8,
        identity_prob: float = 0.1,
        random_prob: float = 0.1,
        is_train: bool = True,
        vocab_range: Tuple[int, int] = (100, 250000),
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.identity_prob = identity_prob
        self.random_prob = random_prob
        self.is_train = is_train
        self.vocab_range = vocab_range

        self.mask_id = getattr(tokenizer, 'mask_token_id', None)
        if self.mask_id is None:
            self.mask_id = 4  # fallback standard mask token id

        self.cls_id = getattr(tokenizer, 'cls_token_id', None) or getattr(tokenizer, 'bos_token_id', 0)
        self.sep_id = getattr(tokenizer, 'sep_token_id', None) or getattr(tokenizer, 'eos_token_id', 2)
        self.pad_id = getattr(tokenizer, 'pad_token_id', 0) or 0

        # Expand rows for each requested target (e.g. mod, head, pv)
        self.samples: List[Dict[str, Any]] = []
        for r in rows:
            ctx = str(r.get('sentence') or r.get('context') or '').strip()
            if not ctx:
                continue
            for t in targets:
                if t not in MARKER_CODE:
                    continue
                if t == 'mod':
                    word = str(r.get('mod', '')).strip()
                elif t == 'head':
                    word = str(r.get('head', '')).strip()
                else:
                    word = str(r.get('compound', '')).strip()
                    if not word and 'mod' in r and 'head' in r:
                        word = f"{r['mod']} {r['head']}".strip()

                if not word:
                    continue

                self.samples.append({
                    'sentence': ctx,
                    'target': t,
                    'word': word,
                    'marker_id': MARKER_CODE[t],
                })

    def __len__(self) -> int:
        return len(self.samples)

    def _tokenize_word(self, word: str) -> List[int]:
        if hasattr(self.tokenizer, 'encode'):
            return list(self.tokenizer.encode(word, add_special_tokens=False))
        return [100]

    def _tokenize_sentence(self, sentence: str) -> List[int]:
        if hasattr(self.tokenizer, 'encode'):
            return list(self.tokenizer.encode(sentence, add_special_tokens=False))
        return [101]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        marker_id = sample['marker_id']
        word_ids = self._tokenize_word(sample['word'])
        ctx_ids = self._tokenize_sentence(sample['sentence'])

        if not word_ids:
            word_ids = [self.mask_id]

        # Apply 80/10/10 Stochastic Denoising to the prefix word tokens
        prefix_tokens: List[int] = []
        for tid in word_ids:
            if self.is_train:
                r = random.random()
                if r < self.mask_prob:
                    prefix_tokens.append(self.mask_id)
                elif r < self.mask_prob + self.identity_prob:
                    prefix_tokens.append(tid)
                else:
                    rand_id = random.randint(self.vocab_range[0], self.vocab_range[1])
                    prefix_tokens.append(rand_id)
            else:
                prefix_tokens.append(self.mask_id)

        # Assemble: [CLS] <marker> prefix_tokens <marker> context [SEP]
        input_ids = [self.cls_id, marker_id] + prefix_tokens + [marker_id] + ctx_ids + [self.sep_id]
        labels = [-100, -100] + word_ids + [-100] + ([-100] * len(ctx_ids)) + [-100]

        # Truncate if exceeds max_len
        if len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len - 1] + [self.sep_id]
            labels = labels[: self.max_len - 1] + [-100]

        attention_mask = [1] * len(input_ids)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


def collate_mlm(batch: List[Dict[str, torch.Tensor]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """Collates a list of samples into dynamically padded batch tensors."""
    max_l = max(x['input_ids'].size(0) for x in batch)

    b_input_ids = []
    b_attention_mask = []
    b_labels = []

    for item in batch:
        l = item['input_ids'].size(0)
        pad_len = max_l - l

        if pad_len > 0:
            pad_tok = torch.full((pad_len,), pad_id, dtype=torch.long)
            pad_zero = torch.zeros(pad_len, dtype=torch.long)
            pad_ign = torch.full((pad_len,), -100, dtype=torch.long)

            b_input_ids.append(torch.cat([item['input_ids'], pad_tok]))
            b_attention_mask.append(torch.cat([item['attention_mask'], pad_zero]))
            b_labels.append(torch.cat([item['labels'], pad_ign]))
        else:
            b_input_ids.append(item['input_ids'])
            b_attention_mask.append(item['attention_mask'])
            b_labels.append(item['labels'])

    return {
        'input_ids': torch.stack(b_input_ids),
        'attention_mask': torch.stack(b_attention_mask),
        'labels': torch.stack(b_labels),
    }

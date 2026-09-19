"""Trainer for Stage 1: Task-Adaptive Prefix MLM Pre-training.

Adapts mmBERT on the target prefix prompt:
    [CLS] <marker> [MASK]...[MASK] <marker> Context [SEP]

Trains LoRA adapters on upper attention layers (e.g. layers 18--22) plus the MLM
head/decoder. Once training is complete, the LoRA weights are merged directly into
the base ModernBertModel via `merge_lora` and exported as a standard Hugging Face
model directory.

Stage 2 (downstream scoring) can then load this checkpoint via `AutoModel.from_pretrained`
without any modifications to downstream model code.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .config import Config
from .data import load_labeled
from .dataset_mlm import PrefixMLMDataset, collate_mlm
from .lora import apply_lora, lora_parameters, merge_lora
from .utils import set_seed


def train_mlm_adaptation(
    cfg: Config,
    logger: logging.Logger,
    device: torch.device,
    data_dir: Path,
    output_dir: Path,
) -> Path:
    """Run Task-Adaptive Prefix MLM adaptation on the configured dataset."""
    set_seed(cfg.seed)

    logger.info("=== STAGE 1: Task-Adaptive Prefix MLM Pre-training ===")
    logger.info("Backbone: %s", cfg.backbone)
    logger.info("Epochs: %d | LR: %.2e | LoRA from layer: %d",
                cfg.mlm_epochs, cfg.mlm_lr, cfg.mlm_from_layer)

    # 1. Load data
    rows = load_labeled(cfg)
    logger.info("Loaded %d rows for MLM task adaptation", len(rows))

    # 2. Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
    train_dataset = PrefixMLMDataset(
        rows=rows,
        tokenizer=tokenizer,
        targets=cfg.targets,
        max_len=cfg.max_context_length,
        mask_prob=cfg.mlm_mask_prob,
        is_train=True,
    )
    logger.info("Generated %d prefix-masked training samples", len(train_dataset))

    pad_id = getattr(tokenizer, 'pad_token_id', 0) or 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=lambda b: collate_mlm(b, pad_id=pad_id),
    )

    # 3. Model setup with LoRA
    model = AutoModelForMaskedLM.from_pretrained(cfg.backbone)
    model.to(device)

    # Alias .model as .lm for apply_lora
    if hasattr(model, 'model') and not hasattr(model, 'lm'):
        model.lm = model.model

    adapters = apply_lora(
        model,
        rank=cfg.lora_rank,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
        targets=cfg.lora_targets,
        from_layer=cfg.mlm_from_layer,
    )
    logger.info("Attached %d LoRA adapters to attention layers >= %d",
                len(adapters), cfg.mlm_from_layer)

    # Freeze base model parameters, keep LoRA and MLM head trainable
    for p in model.parameters():
        p.requires_grad = False
    for p in lora_parameters(adapters):
        p.requires_grad = True
    if hasattr(model, 'head'):
        for p in model.head.parameters():
            p.requires_grad = True
    if hasattr(model, 'decoder'):
        for p in model.decoder.parameters():
            p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info("Trainable parameters: %d", sum(p.numel() for p in trainable_params))

    # 4. Optimizer & AMP
    optimizer = AdamW(trainable_params, lr=cfg.mlm_lr, weight_decay=cfg.weight_decay)
    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp, init_scale=cfg.amp_init_scale)

    # 5. Training loop
    model.train()
    for epoch in range(1, cfg.mlm_epochs + 1):
        total_loss = 0.0
        total_tokens = 0
        correct_tokens = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            if use_amp:
                scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
                optimizer.step()

            # Track accuracy on masked positions
            with torch.no_grad():
                mask = labels != -100
                num_masked = mask.sum().item()
                total_loss += loss.item() * max(num_masked, 1)
                total_tokens += num_masked

                if outputs.logits is not None:
                    preds = outputs.logits.argmax(dim=-1)
                    if preds.shape == labels.shape:
                        correct = (preds[mask] == labels[mask]).sum().item()
                        correct_tokens += correct

        avg_loss = total_loss / max(total_tokens, 1)
        acc = (correct_tokens / max(total_tokens, 1)) * 100.0
        ppl = math.exp(min(avg_loss, 20.0))
        logger.info("MLM Epoch %d/%d | Loss: %.4f | PPL: %.2f | Acc: %.2f%%",
                    epoch, cfg.mlm_epochs, avg_loss, ppl, acc)

    # 6. Merge LoRA into base weights and unwrap
    logger.info("Merging LoRA adapters into base backbone...")
    merge_lora(model, adapters)

    # 7. Save merged checkpoint
    save_dir = Path(cfg.mlm_output_dir) if cfg.mlm_output_dir else (output_dir / "prefix_mlm_adapted")
    save_dir.mkdir(parents=True, exist_ok=True)

    base_lm = model.model if hasattr(model, 'model') else model
    base_lm.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info("Successfully saved Task-Adapted backbone to: %s", save_dir)

    return save_dir

"""Trainer for Stage 1: Task-Adaptive Prefix MLM Pre-training.

Adapts mmBERT on the target prefix prompt:
    [CLS] <marker> [MASK]...[MASK] <marker> Context [SEP]

Memory strategy for T4 / P100 GPUs (14 GB):
    - sparse_prediction=True: only the ~2 masked positions per sample pass
      through the 768→256 000 decoder, shrinking that tensor from
      (B, L, V) → (B*mask_per_row, V). For batch=16 with 2 masks/sample
      that is 32 × 256 000 instead of 4 096 × 256 000 — a ~128× reduction.
    - gradient_checkpointing: recomputes activations during backward instead
      of storing them, trading speed for ~30–40 % less activation memory.
    - decoder param frozen (only LoRA + MLM head dense/norm are trainable)
      so no gradient needs to be accumulated for the 196M decoder matrix.
    - Freeze happens BEFORE the first forward pass so PyTorch does not build
      the computation graph for any frozen tensor.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

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

    # ------------------------------------------------------------------ #
    # 1.  Data
    # ------------------------------------------------------------------ #
    rows = load_labeled(cfg)
    logger.info("Loaded %d rows for MLM task adaptation", len(rows))

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

    # Use a dedicated (smaller) batch size for Stage 1 to fit in 14 GB VRAM
    mlm_batch = getattr(cfg, 'mlm_batch_size', None) or 16
    train_loader = DataLoader(
        train_dataset,
        batch_size=mlm_batch,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=lambda b: collate_mlm(b, pad_id=pad_id),
    )
    logger.info("DataLoader: batch_size=%d, steps/epoch=%d", mlm_batch, len(train_loader))

    # ------------------------------------------------------------------ #
    # 2.  Model  —  key memory knobs set BEFORE moving to GPU
    # ------------------------------------------------------------------ #
    # sparse_prediction=True makes ModernBertForMaskedLM compute logits
    # ONLY at masked positions (labels != -100).  This is the single biggest
    # memory saving: (B×L×V) → (n_masked×V).
    model = AutoModelForMaskedLM.from_pretrained(cfg.backbone)
    model.config.sparse_prediction = True
    model.sparse_prediction = True
    model.sparse_pred_ignore_index = -100

    # gradient_checkpointing trades ~35 % speed for ~40 % less activation RAM
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    elif hasattr(model.model, 'gradient_checkpointing_enable'):
        model.model.gradient_checkpointing_enable()

    model.to(device)

    # Alias .model → .lm so apply_lora can find the backbone subtree
    if hasattr(model, 'model') and not hasattr(model, 'lm'):
        model.lm = model.model

    # ------------------------------------------------------------------ #
    # 3.  Freeze everything first, then apply LoRA to top layers
    # ------------------------------------------------------------------ #
    # Freeze BEFORE first forward so PyTorch never builds a grad graph for
    # the frozen 18-layer base or the 196M decoder matrix.
    for p in model.parameters():
        p.requires_grad = False

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

    # Unfreeze LoRA params + the small MLM prediction head (dense + norm)
    # Keep decoder (768→256k) frozen — it does not need gradients because
    # its weights are never updated; the model learns to steer attention.
    for p in lora_parameters(adapters):
        p.requires_grad = True
    if hasattr(model, 'head'):
        for p in model.head.parameters():
            p.requires_grad = True
    # decoder stays frozen on purpose to avoid its huge gradient

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable: %d / %d params (%.2f%%)",
                n_trainable, n_total, 100 * n_trainable / max(n_total, 1))

    # ------------------------------------------------------------------ #
    # 4.  Optimiser + AMP
    # ------------------------------------------------------------------ #
    optimizer = AdamW(trainable_params, lr=cfg.mlm_lr, weight_decay=cfg.weight_decay)
    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp, init_scale=cfg.amp_init_scale)

    # ------------------------------------------------------------------ #
    # 5.  Training loop
    # ------------------------------------------------------------------ #
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

            if loss is None or not torch.isfinite(loss):
                logger.warning("Step %d: loss=%s — skipping", step, loss)
                continue

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

            # Accuracy tracking (sparse logits: shape = n_masked × V)
            with torch.no_grad():
                mask = labels != -100
                n_masked = mask.sum().item()
                total_loss += loss.item() * max(n_masked, 1)
                total_tokens += n_masked

                if outputs.logits is not None and n_masked > 0:
                    # With sparse_prediction the logits tensor has already
                    # been filtered to masked rows — shape (n_masked, V)
                    logits_2d = outputs.logits.view(-1, outputs.logits.size(-1))
                    target_ids = labels[mask]
                    if logits_2d.size(0) == target_ids.size(0):
                        preds = logits_2d.argmax(dim=-1)
                        correct_tokens += (preds == target_ids).sum().item()

        avg_loss = total_loss / max(total_tokens, 1)
        acc = (correct_tokens / max(total_tokens, 1)) * 100.0
        ppl = math.exp(min(avg_loss, 20.0))
        logger.info("MLM Epoch %d/%d | Loss: %.4f | PPL: %.2f | Acc: %.2f%%",
                    epoch, cfg.mlm_epochs, avg_loss, ppl, acc)

    # ------------------------------------------------------------------ #
    # 6.  Merge LoRA → base weights and export
    # ------------------------------------------------------------------ #
    logger.info("Merging LoRA adapters into base backbone...")
    merge_lora(model, adapters)

    save_dir = Path(cfg.mlm_output_dir) if cfg.mlm_output_dir else (output_dir / "prefix_mlm_adapted")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save only the encoder (ModernBertModel), not the MLM head/decoder.
    # Stage 2 loads this via AutoModel.from_pretrained(save_dir).
    base_lm = model.model if hasattr(model, 'model') else model
    base_lm.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info("Successfully saved Task-Adapted backbone to: %s", save_dir)
    return save_dir

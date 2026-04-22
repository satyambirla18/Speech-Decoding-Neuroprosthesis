#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from speech_decoding.data import HDF5SpeechDataset, collate_ctc
from speech_decoding.model import ConformerCTC
from speech_decoding.utils import ctc_greedy_decode, levenshtein

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, blank_id: int = 0, max_batches: int | None = None):
    model.eval()
    total_edits = 0
    total_len = 0
    total_loss = 0.0
    ctc = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    with torch.no_grad():
        for b_i, batch in enumerate(tqdm(loader, desc="eval", leave=False)):
            if max_batches is not None and b_i >= max_batches:
                break
            x = batch["x"].to(device)
            x_len = batch["x_len"].to(device)
            targets = batch["targets"].to(device)
            target_len = batch["target_len"].to(device)

            logits = model(x, x_len=x_len)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)

            loss = ctc(log_probs, targets, x_len, target_len)
            total_loss += float(loss.item())

            pred_ids = logits.argmax(dim=-1)
            offset = 0
            for i in range(x.shape[0]):
                L = int(target_len[i].item())
                true = targets[offset:offset+L].tolist()
                offset += L

                pred = ctc_greedy_decode(pred_ids[i, :int(x_len[i].item())].tolist(), blank_id=blank_id)
                total_edits += levenshtein(pred, true)
                total_len += max(1, len(true))

    avg_loss = total_loss / max(1, (b_i + 1))
    per = total_edits / max(1, total_len)
    return {"val_loss": avg_loss, "phoneme_error_rate": per}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True, help="Root folder containing HDF5 data (or data_{train,val,test}.hdf5).")
    p.add_argument("--out", type=str, default="runs/speech_decoding", help="Output folder for checkpoints/logs.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--train-partition", type=str, default="train", choices=["train","val","test","none"])
    p.add_argument("--val-partition", type=str, default="val", choices=["train","val","test","none"])
    p.add_argument("--max-train-trials", type=int, default=0, help="Debug: cap number of train trials (0 = no cap).")
    p.add_argument("--max-val-trials", type=int, default=0, help="Debug: cap number of val trials (0 = no cap).")
    p.add_argument("--model-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dim-head", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--vocab-size", type=int, default=42)
    p.add_argument("--blank-id", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_partition = None if args.train_partition == "none" else args.train_partition
    val_partition = None if args.val_partition == "none" else args.val_partition

    train_ds = HDF5SpeechDataset(
        root=args.data_root,
        partition=train_partition,
        normalize=True,
        max_trials=(args.max_train_trials or None),
    )
    val_ds = HDF5SpeechDataset(
        root=args.data_root,
        partition=val_partition,
        normalize=True,
        max_trials=(args.max_val_trials or None),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_ctc
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_ctc
    )

    device = torch.device(args.device)
    model = ConformerCTC(
        input_dim=512,
        model_dim=args.model_dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        vocab_size=args.vocab_size,
        dropout=args.dropout,
        use_rope=True
    ).to(device)

    ctc = nn.CTCLoss(blank=args.blank_id, zero_infinity=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}/{args.epochs}")
        running = 0.0
        n_batches = 0

        for batch in pbar:
            x = batch["x"].to(device, non_blocking=True)
            x_len = batch["x_len"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            target_len = batch["target_len"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x, x_len=x_len)
                log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
                loss = ctc(log_probs, targets, x_len, target_len)

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()

            running += float(loss.item())
            n_batches += 1
            global_step += 1
            pbar.set_postfix(loss=running / max(1, n_batches))

        metrics = evaluate(model, val_loader, device, blank_id=args.blank_id, max_batches=50)
        val_loss = metrics["val_loss"]

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "args": vars(args),
            "metrics": metrics,
        }
        ckpt_path = out / f"ckpt_epoch{epoch}.pt"
        torch.save(ckpt, ckpt_path)

        if val_loss < best:
            best = val_loss
            torch.save(ckpt, out / "best.pt")

        with (out / "metrics.txt").open("a", encoding="utf-8") as f:
            f.write(f"epoch={epoch} train_loss={running/max(1,n_batches):.6f} val_loss={val_loss:.6f} per={metrics['phoneme_error_rate']:.4f}\n")

        print(f"[epoch {epoch}] train_loss={running/max(1,n_batches):.6f} val_loss={val_loss:.6f} per={metrics['phoneme_error_rate']:.4f}")

    print("Done. Best val_loss:", best)

if __name__ == "__main__":
    main()

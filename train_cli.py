# Patch-style CLI wrapper for finetune_t3.py and finetune_s3gen.py
# This will wrap the logic of both scripts in CLI-friendly main() functions
# Includes resume/save checkpoints and learning rate scheduling

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# === Placeholder imports — replace with actual paths ===
from t3dataset import T3Dataset
from t3_hf_backend import T3ForConditionalGeneration
from s3gen import MaskedDiffWithXvec  # or CausalMaskedDiffWithXvec
from utils import AttrDict

# === Helper ===
def save_checkpoint(model, optimizer, scheduler, step, path):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step
    }, path)

def load_checkpoint(model, optimizer, scheduler, path):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt['step']

# === T3 CLI ===
def main_t3():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--resume_ckpt', type=str)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_steps', type=int, default=5000)
    parser.add_argument('--warmup_steps', type=int, default=250)
    parser.add_argument('--save_every', type=int, default=500)
    args = parser.parse_args()

    model = T3ForConditionalGeneration().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_data = T3Dataset(args.data_dir, split='train')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)

    start_step = 0
    if args.resume_ckpt:
        start_step = load_checkpoint(model, optimizer, scheduler, args.resume_ckpt)

    model.train()
    for step, batch in enumerate(train_loader):
        if step < start_step:
            continue

        batch = {k: v.cuda() for k, v in batch.items()}
        loss = model(batch['input_ids'], batch['labels'], batch['attention_mask'])

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % 50 == 0:
            print(f"[Step {step}] Loss: {loss.item():.4f}")

        if step % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f"t3_step{step}.pt")
            save_checkpoint(model, optimizer, scheduler, step, save_path)

        if step >= args.max_steps:
            break

# === S3Gen CLI ===
def main_s3gen():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--resume_ckpt', type=str)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--save_every', type=int, default=1000)
    args = parser.parse_args()

    model = MaskedDiffWithXvec().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_data = T3Dataset(args.data_dir, split='train')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)

    start_step = 0
    if args.resume_ckpt:
        start_step = load_checkpoint(model, optimizer, scheduler, args.resume_ckpt)

    model.train()
    for step, batch in enumerate(train_loader):
        if step < start_step:
            continue

        batch = {k: v.cuda() for k, v in batch.items()}
        loss = model.compute_loss(batch['mels'], batch['text_tokens'], batch['xvector'])

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % 50 == 0:
            print(f"[Step {step}] Loss: {loss.item():.4f}")

        if step % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f"s3gen_step{step}.pt")
            save_checkpoint(model, optimizer, scheduler, step, save_path)

        if step >= args.max_steps:
            break

# === Entrypoints ===
if __name__ == '__main__':
    import sys
    if 't3' in sys.argv[0]:
        main_t3()
    else:
        main_s3gen()

"""
train_fm_dinov2.py -- fine-tune a Hugging Face DINOv2 encoder on one MURA region (PyTorch).
Same index CSV / splits / preprocessing (preprocessing_v2.load_and_preprocess_v2: autocrop + CLAHE) and the same
output files as train_tier2_strong.py (val_probs.npy, test_probs.npy, *_labels.npy, summary.json), so
study_level_eval.py, compare_runs.py and the ensemble scripts work on the result unchanged.
Phase 1: frozen encoder, train the head. Phase 2: whole network, AdamW + warmup + cosine, early stopping on val acc.
"""
import argparse, copy, json, math, os, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TV
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

from preprocessing_v2 import load_and_preprocess_v2

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class MuraDS(Dataset):
    def __init__(self, df, img_size, augment, erase_p=0.0):
        self.paths = df["image_path"].values
        self.labels = df["fracture_label"].values.astype(np.int64)
        self.img_size, self.augment = img_size, augment
        self.erase_p = erase_p
        self.eraser = TV.RandomErasing(p=erase_p, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0.0) if erase_p > 0 else None
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_and_preprocess_v2(self.paths[i], self.img_size, use_autocrop=True, use_clahe=True)
        x = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        if self.augment:
            if random.random() < 0.5:
                x = torch.flip(x, [2])
            if random.random() < 0.5:
                x = torch.flip(x, [1])
            x = TF.affine(x, angle=random.uniform(-10, 10), translate=[0, 0],
                          scale=random.uniform(0.9, 1.1), shear=[0.0, 0.0],
                          interpolation=TF.InterpolationMode.BILINEAR)
        x = (x - self.mean) / self.std
        if self.augment and self.eraser is not None:
            for _ in range(3):
                x = self.eraser(x)
        return x, int(self.labels[i])


class Net(nn.Module):
    def __init__(self, model_dir, dropout):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_dir)
        d = self.backbone.config.hidden_size
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Dropout(dropout), nn.Linear(2 * d, 2))

    def features(self, x):
        h = self.backbone(pixel_values=x).last_hidden_state
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)

    def forward(self, x):
        return self.head(self.features(x))


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, loader, device, amp, tta):
    model.eval()
    views = [lambda x: x, lambda x: torch.flip(x, [3]), lambda x: torch.flip(x, [2]),
             lambda x: torch.flip(x, [2, 3])] if tta else [lambda x: x]
    plain, tta_p = [], []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        ps = []
        for v in views:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                ps.append(F.softmax(model(v(x)).float(), 1).cpu().numpy())
        plain.append(ps[0]); tta_p.append(np.mean(ps, 0))
    return np.concatenate(plain), np.concatenate(tta_p)


def run_epoch(model, loader, device, amp, opt, sched, class_w, frozen_backbone):
    model.train()
    if frozen_backbone:
        model.backbone.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            if frozen_backbone:
                with torch.no_grad():
                    f = model.features(x)
                logits = model.head(f)
            else:
                logits = model(x)
        loss = F.cross_entropy(logits.float(), y, weight=class_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()
        loss_sum += float(loss) * len(y); tot += len(y); correct += int((logits.argmax(1) == y).sum())
    return loss_sum / tot, correct / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-dir", required=True, help="local Hugging Face model folder, e.g. ~/hf_cache/dinov2-base")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--finetune-epochs", type=int, default=40)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=2e-5, help="backbone LR in phase 2 (head gets 10x)")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--erase-p", type=float, default=0.0, help="random-erasing probability per patch (3 patches), train only")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only the first N rows of each split")
    ap.add_argument("--exclude-csv", default="", help="CSV with an image_path column: rows dropped from train and val (never test)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    print(f"device={device} amp_bf16={amp} torch={torch.__version__}", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.index_csv)
    reg = df[df["region"] == args.region]
    assert len(reg), f"no rows for region {args.region}; region values: {sorted(df['region'].unique())}"
    parts = {}
    for s in ("train", "val", "test"):
        d = reg[reg["split"] == s]
        assert len(d), f"no rows for split {s}; split values: {sorted(reg['split'].unique())}"
        parts[s] = d.head(args.limit) if args.limit else d
    if args.exclude_csv:
        ex = set(pd.read_csv(args.exclude_csv)["image_path"].astype(str))
        for sp in ("train", "val"):
            n0 = len(parts[sp])
            parts[sp] = parts[sp][~parts[sp]["image_path"].astype(str).isin(ex)]
            print(f"exclude-csv: {sp} {n0} -> {len(parts[sp])} rows", flush=True)
    tr, va, te = parts["train"], parts["val"], parts["test"]

    g = torch.Generator(); g.manual_seed(args.seed)
    mk = lambda d, aug, shuf: DataLoader(MuraDS(d, args.img_size, aug, args.erase_p if aug else 0.0), batch_size=args.batch_size, shuffle=shuf,
                                         num_workers=args.workers, pin_memory=device.type == "cuda",
                                         persistent_workers=args.workers > 0, generator=g if shuf else None)
    tr_l, va_l, te_l = mk(tr, True, True), mk(va, False, False), mk(te, False, False)

    ytr = tr["fracture_label"].values
    cnt = np.bincount(ytr, minlength=2).astype(np.float64)
    class_w = torch.tensor(len(ytr) / (2.0 * cnt), dtype=torch.float32, device=device)
    print(f"train class counts {cnt.tolist()}  class weights {class_w.tolist()}", flush=True)

    model = Net(args.model_dir, args.dropout).to(device)
    best_val, best_state, bad, epochs_run = -1.0, None, 0, 0

    # ---- Phase 1: frozen encoder, head only ----
    print(f"\n===== Phase 1: head only, {args.warmup_epochs} epochs =====", flush=True)
    for p in model.backbone.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(model.head.parameters(), lr=args.lr_head, weight_decay=0.05)
    for ep in range(args.warmup_epochs):
        t = time.time()
        l, a = run_epoch(model, tr_l, device, amp, opt, None, class_w, True)
        _, vp = predict(model, va_l, device, amp, False)
        va_acc = float((vp.argmax(1) == va["fracture_label"].values).mean())
        print(f"Epoch {ep + 1}/{args.warmup_epochs} - loss {l:.4f} acc {a:.4f} val_acc {va_acc:.4f} ({time.time() - t:.0f}s)", flush=True)

    # ---- Phase 2: full fine-tune ----
    print(f"\n===== Phase 2: fine-tuning full network, up to {args.finetune_epochs} epochs =====", flush=True)
    for p in model.backbone.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": args.lr_ft},
                             {"params": model.head.parameters(), "lr": args.lr_ft * 10}], weight_decay=0.05)
    steps_ep = max(1, len(tr_l))
    total, warm = steps_ep * args.finetune_epochs, steps_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    yva = va["fracture_label"].values
    for ep in range(args.finetune_epochs):
        t = time.time()
        l, a = run_epoch(model, tr_l, device, amp, opt, sched, class_w, False)
        _, vp = predict(model, va_l, device, amp, False)
        va_acc = float((vp.argmax(1) == yva).mean())
        epochs_run = ep + 1
        print(f"Epoch {ep + 1}/{args.finetune_epochs} - loss {l:.4f} acc {a:.4f} val_acc {va_acc:.4f} ({time.time() - t:.0f}s)", flush=True)
        if va_acc > best_val:
            best_val, bad = va_acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stopping (no val_acc gain for {args.patience} epochs)", flush=True)
                break
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(args.output_dir, "best_model.pt"))

    # ---- outputs (same files/semantics as train_tier2_strong.py) ----
    val_plain, val_tta = predict(model, va_l, device, amp, args.tta)
    test_plain, test_tta = predict(model, te_l, device, amp, args.tta)
    yte = te["fracture_label"].values
    np.save(os.path.join(args.output_dir, "val_labels.npy"), yva)
    np.save(os.path.join(args.output_dir, "test_labels.npy"), yte)
    result = {
        "region": args.region, "backbone": "dinov2:" + os.path.basename(os.path.normpath(args.model_dir)),
        "dropout": args.dropout, "lr_head": args.lr_head, "lr_ft": args.lr_ft, "img_size": args.img_size,
        "seed": args.seed, "finetune_epochs_trained": epochs_run, "limit": args.limit,
        "val_accuracy": float(best_val),
        "test_accuracy": float((test_plain.argmax(1) == yte).mean()),
        "test_loss": float(-np.log(np.clip(test_plain[np.arange(len(yte)), yte], 1e-7, 1)).mean()),
    }
    if args.tta:
        np.save(os.path.join(args.output_dir, "val_probs_noaug.npy"), val_plain)
        np.save(os.path.join(args.output_dir, "test_probs_noaug.npy"), test_plain)
        np.save(os.path.join(args.output_dir, "val_probs.npy"), val_tta)
        np.save(os.path.join(args.output_dir, "test_probs.npy"), test_tta)
        result.update({"tta": "flip4",
                       "test_accuracy_no_tta": float((test_plain.argmax(1) == yte).mean()),
                       "test_accuracy_tta": float((test_tta.argmax(1) == yte).mean()),
                       "val_accuracy_no_tta": float((val_plain.argmax(1) == yva).mean()),
                       "val_accuracy_tta": float((val_tta.argmax(1) == yva).mean())})
        print(f"TTA (flip x4): test_acc {result['test_accuracy_no_tta']:.4f} -> {result['test_accuracy_tta']:.4f}")
    else:
        np.save(os.path.join(args.output_dir, "val_probs.npy"), val_plain)
        np.save(os.path.join(args.output_dir, "test_probs.npy"), test_plain)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[{args.region} | DINOv2] val_acc={result['val_accuracy']:.4f} test_acc={result['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()

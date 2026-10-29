"""
train_fm_mil.py -- STUDY-LEVEL (multi-view / multiple-instance) DINOv2 fine-tuning on one MURA region.
MURA gives ONE label per study, but per-image training hands that label to every view, even views that do not show the
finding (label noise at view level). Here all views of a study go through the SAME DINOv2 encoder, the view features are
pooled (gated attention + max pooling) and ONE loss is applied to the study label.
Same index CSV / splits / preprocessing as train_fm_dinov2.py. Output files are written at image-row level (each image row
gets its study's probability) so study_level_eval.py, compare_runs.py and the ensemble scripts work unchanged.
summary.json: test_accuracy / *_no_tta / *_tta here are STUDY-level accuracies.
"""
import argparse, json, math, os, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

from preprocessing_v2 import load_and_preprocess_v2
import train_fm_dinov2 as T


def study_of(path):
    return os.path.dirname(str(path))


class StudyDS(Dataset):
    def __init__(self, df, img_size, augment, max_views, view_drop):
        self.img_size, self.augment, self.max_views, self.view_drop = img_size, augment, max_views, view_drop
        self.order, self.paths, lab = [], {}, {}
        for p, y in zip(df["image_path"].values, df["fracture_label"].values):
            s = study_of(p)
            if s not in self.paths:
                self.paths[s] = []; lab[s] = int(y); self.order.append(s)
            self.paths[s].append(p)
        self.labels = np.array([lab[s] for s in self.order], dtype=np.int64)
        self.mean = torch.tensor(T.MEAN).view(3, 1, 1)
        self.std = torch.tensor(T.STD).view(3, 1, 1)

    def __len__(self):
        return len(self.order)

    def _view(self, path):
        img = load_and_preprocess_v2(path, self.img_size, use_autocrop=True, use_clahe=True)
        x = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        if self.augment:
            if random.random() < 0.5:
                x = torch.flip(x, [2])
            if random.random() < 0.5:
                x = torch.flip(x, [1])
            x = TF.affine(x, angle=random.uniform(-10, 10), translate=[0, 0], scale=random.uniform(0.9, 1.1),
                          shear=[0.0, 0.0], interpolation=TF.InterpolationMode.BILINEAR)
        return (x - self.mean) / self.std

    def __getitem__(self, i):
        ps = list(self.paths[self.order[i]])
        if self.augment:
            if len(ps) > self.max_views:
                ps = random.sample(ps, self.max_views)
            if self.view_drop > 0 and len(ps) > 1:
                keep = [p for p in ps if random.random() >= self.view_drop]
                ps = keep if keep else [random.choice(ps)]
        return torch.stack([self._view(p) for p in ps]), int(self.labels[i])


def collate(batch):
    V = max(b[0].shape[0] for b in batch)
    B, shape = len(batch), batch[0][0].shape[1:]
    x = torch.zeros(B, V, *shape)
    mask = torch.zeros(B, V, dtype=torch.bool)
    for i, (v, _) in enumerate(batch):
        x[i, :v.shape[0]] = v
        mask[i, :v.shape[0]] = True
    return x, mask, torch.tensor([b[1] for b in batch], dtype=torch.long)


class MILNet(nn.Module):
    def __init__(self, model_dir, dropout, attn_dim=256):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_dir)
        d = self.backbone.config.hidden_size
        D = 2 * d
        self.D = D
        self.view_norm = nn.LayerNorm(D)
        self.att_v = nn.Linear(D, attn_dim)
        self.att_u = nn.Linear(D, attn_dim)
        self.att_w = nn.Linear(attn_dim, 1)
        self.head = nn.Sequential(nn.LayerNorm(2 * D), nn.Dropout(dropout), nn.Linear(2 * D, 2))

    def view_feats(self, x):
        h = self.backbone(pixel_values=x).last_hidden_state
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)

    def non_backbone_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]

    def forward(self, x, mask, frozen_backbone=False):
        B, V = mask.shape
        flat = x[mask]                                   # (N, 3, H, W): only the real views
        if frozen_backbone:
            with torch.no_grad():
                f = self.view_feats(flat)
        else:
            f = self.view_feats(flat)
        h = torch.zeros(B, V, self.D, device=x.device, dtype=torch.float32)
        h[mask] = self.view_norm(f.float())
        a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(-1)   # (B, V)
        a = torch.softmax(a.masked_fill(~mask, -1e9), dim=1)
        att = (a.unsqueeze(-1) * h).sum(1)
        mx = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values
        return self.head(torch.cat([att, mx], 1))


@torch.no_grad()
def predict(model, loader, device, amp, tta):
    model.eval()
    flips = [[], [4], [3], [3, 4]] if tta else [[]]     # x is (B, V, 3, H, W): dim 4 = width, dim 3 = height
    plain, tta_p = [], []
    for x, mask, _ in loader:
        x, mask = x.to(device, non_blocking=True), mask.to(device)
        ps = []
        for dims in flips:
            xv = torch.flip(x, dims) if dims else x
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                lg = model(xv, mask)
            ps.append(F.softmax(lg.float(), 1).cpu().numpy())
        plain.append(ps[0]); tta_p.append(np.mean(ps, 0))
    return np.concatenate(plain), np.concatenate(tta_p)


def run_epoch(model, loader, device, amp, opt, sched, class_w, frozen, accum=1):
    model.train()
    if frozen:
        model.backbone.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    opt.zero_grad(set_to_none=True)
    n_it = len(loader)
    for it, (x, mask, y) in enumerate(loader):
        x, mask, y = x.to(device, non_blocking=True), mask.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(x, mask, frozen_backbone=frozen)
        loss = F.cross_entropy(logits.float(), y, weight=class_w)
        (loss / accum).backward()
        if (it + 1) % accum == 0 or (it + 1) == n_it:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            opt.zero_grad(set_to_none=True)
        loss_sum += float(loss.detach()) * len(y); tot += len(y); correct += int((logits.argmax(1) == y).sum().item())
    return loss_sum / tot, correct / tot


def to_rows(df_part, ds, probs):
    """study probabilities -> one row per image row of the index CSV (same order as the CSV split)."""
    idx = {s: i for i, s in enumerate(ds.order)}
    return np.stack([probs[idx[study_of(p)]] for p in df_part["image_path"].values])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=8, help="STUDIES per step (about 3 views each)")
    ap.add_argument("--max-views", type=int, default=5, help="training: random subset if a study has more views")
    ap.add_argument("--view-drop", type=float, default=0.15, help="training: drop each view with this prob (keeps >= 1)")
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--finetune-epochs", type=int, default=40)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=2e-5)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps (effective batch = batch-size x accum studies)")
    ap.add_argument("--init-from", default="", help="best_model.pt of an image-level DINOv2 run: initialise the BACKBONE from it")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N rows of each split")
    args = ap.parse_args()

    T.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    print(f"device={device} amp_bf16={amp} torch={torch.__version__}", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.index_csv)
    reg = df[df["region"] == args.region]
    assert len(reg), f"no rows for region {args.region}"
    parts = {}
    for s in ("train", "val", "test"):
        d = reg[reg["split"] == s]
        assert len(d), f"no rows for split {s}"
        parts[s] = d.head(args.limit) if args.limit else d
    tr_ds = StudyDS(parts["train"], args.img_size, True, args.max_views, args.view_drop)
    va_ds = StudyDS(parts["val"], args.img_size, False, 99, 0.0)
    te_ds = StudyDS(parts["test"], args.img_size, False, 99, 0.0)
    print(f"studies: train {len(tr_ds)} ({len(parts['train'])} images), val {len(va_ds)}, test {len(te_ds)}", flush=True)

    g = torch.Generator(); g.manual_seed(args.seed)
    mk = lambda ds, shuf: DataLoader(ds, batch_size=args.batch_size, shuffle=shuf, collate_fn=collate,
                                     num_workers=args.workers, pin_memory=device.type == "cuda",
                                     persistent_workers=args.workers > 0, generator=g if shuf else None)
    tr_l, va_l, te_l = mk(tr_ds, True), mk(va_ds, False), mk(te_ds, False)

    cnt = np.bincount(tr_ds.labels, minlength=2).astype(np.float64)
    class_w = torch.tensor(len(tr_ds) / (2.0 * cnt), dtype=torch.float32, device=device)
    print(f"train study class counts {cnt.tolist()}  class weights {class_w.tolist()}", flush=True)

    model = MILNet(args.model_dir, args.dropout)
    if args.init_from:
        sd = torch.load(args.init_from, map_location="cpu")
        bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        assert bb, "no backbone.* keys in --init-from checkpoint"
        model.backbone.load_state_dict(bb)
        print(f"backbone initialised from {args.init_from} ({len(bb)} tensors)", flush=True)
    model = model.to(device)
    best_val, best_state, bad, epochs_run = -1.0, None, 0, 0
    yva = va_ds.labels

    print(f"\n===== Phase 1: pooling+head only, {args.warmup_epochs} epochs =====", flush=True)
    for p in model.backbone.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(model.non_backbone_params(), lr=args.lr_head, weight_decay=0.05)
    for ep in range(args.warmup_epochs):
        t = time.time()
        l, a = run_epoch(model, tr_l, device, amp, opt, None, class_w, True, args.accum)
        vp, _ = predict(model, va_l, device, amp, False)
        print(f"Epoch {ep + 1}/{args.warmup_epochs} - loss {l:.4f} acc {a:.4f} val_acc {float((vp.argmax(1) == yva).mean()):.4f} ({time.time() - t:.0f}s)", flush=True)

    print(f"\n===== Phase 2: fine-tuning full network, up to {args.finetune_epochs} epochs =====", flush=True)
    for p in model.backbone.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": args.lr_ft},
                             {"params": model.non_backbone_params(), "lr": args.lr_ft * 10}], weight_decay=0.05)
    steps_ep = max(1, math.ceil(len(tr_l) / args.accum))
    total, warm = steps_ep * args.finetune_epochs, steps_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    for ep in range(args.finetune_epochs):
        t = time.time()
        l, a = run_epoch(model, tr_l, device, amp, opt, sched, class_w, False, args.accum)
        vp, _ = predict(model, va_l, device, amp, False)
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

    val_plain, val_tta = predict(model, va_l, device, amp, args.tta)
    test_plain, test_tta = predict(model, te_l, device, amp, args.tta)
    yte = te_ds.labels
    out = lambda name, arr: np.save(os.path.join(args.output_dir, name), arr)
    out("val_labels.npy", parts["val"]["fracture_label"].values)
    out("test_labels.npy", parts["test"]["fracture_label"].values)
    final_val, final_test = (val_tta, test_tta) if args.tta else (val_plain, test_plain)
    out("val_probs.npy", to_rows(parts["val"], va_ds, final_val))
    out("test_probs.npy", to_rows(parts["test"], te_ds, final_test))
    result = {"region": args.region, "backbone": "dinov2-mil:" + os.path.basename(os.path.normpath(args.model_dir)),
              "level": "study (accuracy fields are per study)", "dropout": args.dropout, "lr_ft": args.lr_ft,
              "img_size": args.img_size, "seed": args.seed, "max_views": args.max_views, "view_drop": args.view_drop,
              "finetune_epochs_trained": epochs_run, "limit": args.limit, "val_accuracy": float(best_val),
              "test_accuracy": float((test_plain.argmax(1) == yte).mean())}
    if args.tta:
        out("val_probs_noaug.npy", to_rows(parts["val"], va_ds, val_plain))
        out("test_probs_noaug.npy", to_rows(parts["test"], te_ds, test_plain))
        result.update({"tta": "flip4", "test_accuracy_no_tta": float((test_plain.argmax(1) == yte).mean()),
                       "test_accuracy_tta": float((test_tta.argmax(1) == yte).mean()),
                       "val_accuracy_no_tta": float((val_plain.argmax(1) == yva).mean()),
                       "val_accuracy_tta": float((val_tta.argmax(1) == yva).mean())})
        print(f"TTA (flip x4), STUDY level: test_acc {result['test_accuracy_no_tta']:.4f} -> {result['test_accuracy_tta']:.4f}")
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[{args.region} | DINOv2-MIL, study level] val_acc={result['val_accuracy']:.4f} test_acc={result['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()

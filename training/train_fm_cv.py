"""
train_fm_cv.py -- 5-fold, patient-grouped cross-validation of the DINOv2 classifier on the TRAIN+VAL rows of one
MURA region, to get an out-of-fold (OOF) fracture probability for every training image. The test split is never used.
Used to find mislabelled studies (find_label_issues.py). Resumable: a finished fold is skipped on restart.
Outputs in --output-dir: fold<k>.npz, pooled_index.csv, oof_probs.npy (aligned with pooled_index.csv).
"""
import argparse, math, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import DataLoader

import train_fm_dinov2 as T


def make_loader(df, args, device, augment, shuffle, gen=None):
    return DataLoader(T.MuraDS(df, args.img_size, augment), batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, pin_memory=device.type == "cuda",
                      persistent_workers=args.workers > 0, generator=gen if shuffle else None)


def fit_fold(tr, iv, args, device, amp, seed):
    T.set_seed(seed)
    g = torch.Generator(); g.manual_seed(seed)
    tr_l, iv_l = make_loader(tr, args, device, True, True, g), make_loader(iv, args, device, False, False)
    cnt = np.bincount(tr["fracture_label"].values, minlength=2).astype(np.float64)
    class_w = torch.tensor(len(tr) / (2.0 * cnt), dtype=torch.float32, device=device)
    model = T.Net(args.model_dir, args.dropout).to(device)
    yiv = iv["fracture_label"].values
    for p in model.backbone.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(model.head.parameters(), lr=args.lr_head, weight_decay=0.05)
    for ep in range(args.warmup_epochs):
        T.run_epoch(model, tr_l, device, amp, opt, None, class_w, True)
    for p in model.backbone.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": args.lr_ft},
                             {"params": model.head.parameters(), "lr": args.lr_ft * 10}], weight_decay=0.05)
    steps = max(1, len(tr_l)); total, warm = steps * args.finetune_epochs, steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    best, best_state, bad = -1.0, None, 0
    for ep in range(args.finetune_epochs):
        l, a = T.run_epoch(model, tr_l, device, amp, opt, sched, class_w, False)
        _, vp = T.predict(model, iv_l, device, amp, False)
        acc = float((vp.argmax(1) == yiv).mean())
        print(f"   epoch {ep + 1} loss {l:.4f} acc {a:.4f} inner_val_acc {acc:.4f}", flush=True)
        if acc > best:
            best, bad = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--finetune-epochs", type=int, default=20)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=2e-5)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: keep only the first N rows")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.index_csv)
    pool = df[(df["region"] == args.region) & (df["split"].isin(["train", "val"]))].reset_index(drop=True)
    assert len(pool), f"no train/val rows for {args.region}"
    if args.limit:
        pool = pool.head(args.limit)
    print(f"{args.region}: pooled train+val rows {len(pool)}, patients {pool['patient_id'].nunique()}", flush=True)

    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=0)
    folds = list(skf.split(pool, pool["fracture_label"].values, groups=pool["patient_id"].values))
    fold_of = np.zeros(len(pool), dtype=int)
    for k, (_, te_idx) in enumerate(folds):
        fold_of[te_idx] = k
    pool["fold"] = fold_of
    pool.to_csv(os.path.join(args.output_dir, "pooled_index.csv"), index=False)

    oof = np.full((len(pool), 2), np.nan)
    for k, (tr_idx, ho_idx) in enumerate(folds):
        fn = os.path.join(args.output_dir, f"fold{k}.npz")
        if os.path.exists(fn):
            z = np.load(fn); oof[z["idx"]] = z["probs"]
            print(f"fold {k}: already done, skipped", flush=True)
            continue
        t0 = time.time()
        tr_all = pool.iloc[tr_idx]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=k)
        a, b = next(gss.split(tr_all, groups=tr_all["patient_id"].values))
        tr, iv = tr_all.iloc[a], tr_all.iloc[b]
        print(f"\n===== fold {k + 1}/{args.folds}: train {len(tr)}, inner-val {len(iv)}, held-out {len(ho_idx)} =====", flush=True)
        model, best = fit_fold(tr, iv, args, device, amp, args.seed + k)
        ho = pool.iloc[ho_idx]
        _, probs = T.predict(model, make_loader(ho, args, device, False, False), device, amp, True)  # flip-x4 TTA
        np.savez(fn, idx=ho_idx, probs=probs)
        oof[ho_idx] = probs
        acc = float((probs.argmax(1) == ho["fracture_label"].values).mean())
        print(f"fold {k + 1}: inner-val {best:.4f}  held-out image acc {acc:.4f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert not np.isnan(oof).any(), "some rows have no OOF prediction"
    np.save(os.path.join(args.output_dir, "oof_probs.npy"), oof)
    print(f"\nOOF image accuracy {float((oof.argmax(1) == pool['fracture_label'].values).mean()):.4f}  -> oof_probs.npy written", flush=True)


if __name__ == "__main__":
    main()

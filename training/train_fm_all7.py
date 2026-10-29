"""
train_fm_all7.py -- ONE DINOv2-base model trained on ALL MURA body parts (the 3 target regions + extra regions from
--extra-train-csv), evaluated on the target regions only (same val/test studies as every other experiment).
Tests the hypothesis that the published model's advantage on hand/elbow/shoulder comes from training on all 7 body parts.
Training rows: train split of the target regions + all extra rows. Early stopping: accuracy on the pooled VALIDATION rows
of the 3 target regions. Writes, per target region R, results/tier2_<R>_dinov2_<tag>_s<seed>/ with the same files as
train_fm_dinov2.py (val/test probs + labels + summary.json), so compare_runs.py / final_metrics-style scripts work unchanged.
"""
import argparse, json, math, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train_fm_dinov2 as T

TARGETS = ["XR_ELBOW", "XR_HAND", "XR_SHOULDER"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--extra-train-csv", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tag", default="all7")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--finetune-epochs", type=int, default=25)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=2e-5)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N rows of each part")
    args = ap.parse_args()

    T.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    print(f"device={device} amp_bf16={amp} torch={torch.__version__}", flush=True)

    df = pd.read_csv(args.index_csv)
    extra = pd.read_csv(args.extra_train_csv)
    cols = ["image_path", "fracture_label"]
    head = (lambda d: d.head(args.limit)) if args.limit else (lambda d: d)
    tr_t = head(df[df["region"].isin(TARGETS) & (df["split"] == "train")])
    tr_e = head(extra) if args.limit else extra
    tr = pd.concat([tr_t[cols], tr_e[cols]], ignore_index=True)
    vals = {R: head(df[(df["region"] == R) & (df["split"] == "val")]) for R in TARGETS}
    tests = {R: head(df[(df["region"] == R) & (df["split"] == "test")]) for R in TARGETS}
    va_all = pd.concat([vals[R][cols] for R in TARGETS], ignore_index=True)
    print(f"train rows: {len(tr_t)} target + {len(tr_e)} extra = {len(tr)}; pooled val rows {len(va_all)}", flush=True)

    g = torch.Generator(); g.manual_seed(args.seed)
    mk = lambda d, aug, shuf: DataLoader(T.MuraDS(d, args.img_size, aug), batch_size=args.batch_size, shuffle=shuf,
                                         num_workers=args.workers, pin_memory=device.type == "cuda",
                                         persistent_workers=args.workers > 0, generator=g if shuf else None)
    tr_l, va_l = mk(tr, True, True), mk(va_all, False, False)
    cnt = np.bincount(tr["fracture_label"].values, minlength=2).astype(np.float64)
    class_w = torch.tensor(len(tr) / (2.0 * cnt), dtype=torch.float32, device=device)
    print(f"train class counts {cnt.tolist()} weights {class_w.tolist()}", flush=True)

    model = T.Net(args.model_dir, args.dropout).to(device)
    yva = va_all["fracture_label"].values
    best_val, best_state, bad, epochs_run = -1.0, None, 0, 0

    print(f"\n===== Phase 1: head only, {args.warmup_epochs} epochs =====", flush=True)
    for p in model.backbone.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(model.head.parameters(), lr=args.lr_head, weight_decay=0.05)
    for ep in range(args.warmup_epochs):
        t = time.time()
        l, a = T.run_epoch(model, tr_l, device, amp, opt, None, class_w, True)
        _, vp = T.predict(model, va_l, device, amp, False)
        print(f"Epoch {ep + 1}/{args.warmup_epochs} - loss {l:.4f} acc {a:.4f} val_acc {float((vp.argmax(1) == yva).mean()):.4f} ({time.time() - t:.0f}s)", flush=True)

    print(f"\n===== Phase 2: full fine-tune, up to {args.finetune_epochs} epochs =====", flush=True)
    for p in model.backbone.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": args.lr_ft},
                             {"params": model.head.parameters(), "lr": args.lr_ft * 10}], weight_decay=0.05)
    steps = max(1, len(tr_l)); total, warm = steps * args.finetune_epochs, steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    for ep in range(args.finetune_epochs):
        t = time.time()
        l, a = T.run_epoch(model, tr_l, device, amp, opt, sched, class_w, False)
        _, vp = T.predict(model, va_l, device, amp, False)
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

    for R in TARGETS:
        out = os.path.join(args.results_dir, f"tier2_{R}_dinov2_{args.tag}_s{args.seed}")
        os.makedirs(out, exist_ok=True)
        vdf, tdf = vals[R], tests[R]
        vplain, vtta = T.predict(model, mk(vdf, False, False), device, amp, args.tta)
        tplain, ttta = T.predict(model, mk(tdf, False, False), device, amp, args.tta)
        yv, yt = vdf["fracture_label"].values, tdf["fracture_label"].values
        np.save(os.path.join(out, "val_labels.npy"), yv); np.save(os.path.join(out, "test_labels.npy"), yt)
        res = {"region": R, "backbone": "dinov2 trained on all MURA body parts", "tag": args.tag, "seed": args.seed,
               "train_rows": int(len(tr)), "finetune_epochs_trained": epochs_run, "limit": args.limit,
               "val_accuracy": float((vplain.argmax(1) == yv).mean()), "test_accuracy": float((tplain.argmax(1) == yt).mean())}
        if args.tta:
            np.save(os.path.join(out, "val_probs_noaug.npy"), vplain); np.save(os.path.join(out, "test_probs_noaug.npy"), tplain)
            np.save(os.path.join(out, "val_probs.npy"), vtta); np.save(os.path.join(out, "test_probs.npy"), ttta)
            res.update({"tta": "flip4", "test_accuracy_no_tta": res["test_accuracy"],
                        "test_accuracy_tta": float((ttta.argmax(1) == yt).mean()),
                        "val_accuracy_no_tta": res["val_accuracy"], "val_accuracy_tta": float((vtta.argmax(1) == yv).mean())})
        else:
            np.save(os.path.join(out, "val_probs.npy"), vplain); np.save(os.path.join(out, "test_probs.npy"), tplain)
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(f"[{R} | all-body-parts DINOv2] test_acc {res['test_accuracy']:.4f}" + (f" -> TTA {res['test_accuracy_tta']:.4f}" if args.tta else ""), flush=True)
    torch.save(best_state, os.path.join(args.results_dir, f"all7_{args.tag}_s{args.seed}_best_model.pt"))


if __name__ == "__main__":
    main()

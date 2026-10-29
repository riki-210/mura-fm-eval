"""Adds --exclude-csv to train_fm_dinov2.py: drops listed image_paths from the TRAIN and VAL splits only (never test)."""
import ast, sys
f = "train_fm_dinov2.py"
s = open(f).read()
if "--exclude-csv" in s:
    print("already patched"); sys.exit(0)
a1 = '    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only the first N rows of each split")\n'
assert a1 in s
s = s.replace(a1, a1 + '    ap.add_argument("--exclude-csv", default="", help="CSV with an image_path column: rows dropped from train and val (never test)")\n')
a2 = '    tr, va, te = parts["train"], parts["val"], parts["test"]\n'
assert a2 in s
s = s.replace(a2, '''    if args.exclude_csv:
        ex = set(pd.read_csv(args.exclude_csv)["image_path"].astype(str))
        for sp in ("train", "val"):
            n0 = len(parts[sp])
            parts[sp] = parts[sp][~parts[sp]["image_path"].astype(str).isin(ex)]
            print(f"exclude-csv: {sp} {n0} -> {len(parts[sp])} rows", flush=True)
''' + a2)
ast.parse(s)
open(f, "w").write(s)
print("patched OK")

"""make_flag_montage.py -- picture the most suspicious studies so you can eyeball them (run in env fm).
usage: python make_flag_montage.py LABEL_ISSUES_CSV POOLED_INDEX_CSV OUT_PNG [N=12]"""
import os, sys
import cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
issues, pooled, out = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 12
st = pd.read_csv(issues); st = st[st["flagged"]].head(n)
pool = pd.read_csv(pooled); pool["study"] = pool["image_path"].map(lambda p: os.path.dirname(str(p)))
first = pool.groupby("study")["image_path"].first()
cols = 4; rows = max(1, (len(st) + cols - 1) // cols)
fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4.2 * rows)); axs = axs.ravel() if hasattr(axs, "ravel") else [axs]
for ax in axs: ax.axis("off")
for ax, (_, r) in zip(axs, st.iterrows()):
    im = cv2.imread(first[r["study"]], cv2.IMREAD_GRAYSCALE)
    if im is not None: ax.imshow(im, cmap="gray")
    ax.set_title(f"label={'FRACTURE' if r['label'] else 'normal'}  model p(fx)={r['p_fracture']:.2f}\n{os.path.basename(r['study'])[:30]}", fontsize=8)
plt.tight_layout(); plt.savefig(out, dpi=80); print("saved", out)

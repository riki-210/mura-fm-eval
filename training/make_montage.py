"""Draw 12 random test images with their attention boxes (run in the 'fm' env; needs numpy, pandas, cv2, matplotlib)."""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from preprocessing_v2 import load_and_preprocess_v2

ap = argparse.ArgumentParser()
ap.add_argument("--index-csv", required=True)
ap.add_argument("--region", required=True)
ap.add_argument("--boxes-npz", required=True)
ap.add_argument("--out-png", required=True)
ap.add_argument("--img-size", type=int, default=224)
ap.add_argument("--n", type=int, default=12)
a = ap.parse_args()

z = np.load(a.boxes_npz, allow_pickle=True)
box = {str(p): b for p, b in zip(z["paths"], z["boxes"])}
df = pd.read_csv(a.index_csv)
d = df[(df["region"] == a.region) & (df["split"] == "test")].sample(a.n, random_state=0)
cols = 4
rows = int(np.ceil(len(d) / cols))
fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
axs = np.array(axs).reshape(-1)
for ax in axs:
    ax.axis("off")
for ax, (p, lab) in zip(axs, zip(d["image_path"], d["fracture_label"])):
    im = load_and_preprocess_v2(p, a.img_size, use_autocrop=True, use_clahe=True)
    ax.imshow(im)
    y0, x0, y1, x1 = box[str(p)]
    S = a.img_size
    ax.add_patch(plt.Rectangle((x0 * S, y0 * S), (x1 - x0) * S, (y1 - y0) * S, fill=False, edgecolor="red", linewidth=2))
    ax.set_title(f"{a.region} label={int(lab)}", fontsize=9)
plt.tight_layout()
plt.savefig(a.out_png, dpi=90)
print("wrote", a.out_png)

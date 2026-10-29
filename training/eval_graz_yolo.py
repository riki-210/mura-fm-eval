import argparse, json, os, glob
import numpy as np, pandas as pd, yaml
from sklearn.metrics import roc_auc_score
from ultralytics import YOLO
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)          # run name under results/yolo
ap.add_argument("--imgsz", type=int, default=640)
a = ap.parse_args()
D = os.path.expanduser("~/mediinsight_project")
run = f"{D}/results/yolo/{a.run}"
Y = f"{D}/data/graz_yolo"; data = f"{Y}/graz.yaml"
names = yaml.safe_load(open(data))["names"]; FR = names.index("fracture")
model = YOLO(f"{run}/weights/best.pt")
out = {"run": a.run, "imgsz": a.imgsz}
for sp in ("val", "test"):
    m = model.val(data=data, split=sp, imgsz=a.imgsz, batch=16, workers=2, conf=0.001, iou=0.6, plots=False,
                  project=f"{run}/eval", name=sp, exist_ok=True, verbose=False)
    per = {names[c]: {"AP50": float(m.box.ap50[i]), "AP50-95": float(m.box.ap[i])}
           for i, c in enumerate(m.box.ap_class_index)}
    out[sp] = {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map), "per_class": per}
    print(sp, "mAP50 %.3f mAP50-95 %.3f | fracture AP50 %.3f AP50-95 %.3f" % (
        m.box.map50, m.box.map, per.get("fracture", {}).get("AP50", float("nan")),
        per.get("fracture", {}).get("AP50-95", float("nan"))), flush=True)
idx = pd.read_csv(f"{D}/data/graz_index.csv")
idx["stem"] = idx.image_path.map(lambda p: os.path.splitext(os.path.basename(p))[0])
lab = dict(zip(idx.stem, idx.has_fracture))
def score(sp):
    files = sorted(glob.glob(f"{Y}/images/{sp}/*.png")); s, y = [], []
    for i in range(0, len(files), 64):
        for r, f in zip(model.predict(files[i:i+64], imgsz=a.imgsz, conf=0.001, iou=0.6, verbose=False), files[i:i+64]):
            c = r.boxes.cls.cpu().numpy(); p = r.boxes.conf.cpu().numpy()
            s.append(float(p[c == FR].max()) if (c == FR).any() else 0.0)
            y.append(int(lab[os.path.splitext(os.path.basename(f))[0]]))
    return np.array(s), np.array(y)
sv, yv = score("val"); st, yt = score("test")
ts = np.unique(np.round(sv, 4)); best, bj = 0.5, -1
for t in ts:
    p = sv >= t; j = (p[yv == 1].mean() if (yv == 1).any() else 0) - (p[yv == 0].mean() if (yv == 0).any() else 0)
    if j > bj: bj, best = j, float(t)
pt = st >= best
out["image_level"] = {"threshold_from_val_youden": best, "auc": float(roc_auc_score(yt, st)),
    "accuracy": float((pt == yt).mean()), "sensitivity": float(pt[yt == 1].mean()),
    "specificity": float((~pt)[yt == 0].mean()), "n_test": int(len(yt))}
print("image-level (fracture box score):", out["image_level"], flush=True)
json.dump(out, open(f"{run}/eval_summary.json", "w"), indent=1)
print("EVAL DONE")

import os, yaml, pandas as pd
D = os.path.expanduser("~/mediinsight_project/data")
G = f"{D}/GRAZPEDWRI-DX"; Y = f"{D}/graz_yolo"
idx = pd.read_csv(f"{D}/graz_index.csv")
meta = yaml.safe_load(open(f"{G}/yolov5/meta.yaml"))
names = meta["names"]
print("classes:", names)
print("split counts:", idx.split.value_counts().to_dict())
for sp in ("train", "val", "test"):
    os.makedirs(f"{Y}/images/{sp}", exist_ok=True)
    os.makedirs(f"{Y}/labels/{sp}", exist_ok=True)
n = empty = 0
for r in idx.itertuples():
    stem = os.path.splitext(os.path.basename(r.image_path))[0]
    sp = r.split
    di, dl = f"{Y}/images/{sp}/{stem}.png", f"{Y}/labels/{sp}/{stem}.txt"
    if not os.path.lexists(di): os.symlink(r.image_path, di)
    src = f"{G}/yolov5/labels/{stem}.txt"
    if os.path.exists(src):
        if not os.path.lexists(dl): os.symlink(src, dl)
    else:
        open(dl, "a").close(); empty += 1
    n += 1
data = {"path": Y, "train": "images/train", "val": "images/val", "test": "images/test", "names": names}
yaml.safe_dump(data, open(f"{Y}/graz.yaml", "w"))
print("linked", n, "images; missing label files made empty:", empty)
print("wrote", f"{Y}/graz.yaml")

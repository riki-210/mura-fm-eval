import sys, os, glob
import pandas as pd
from ultralytics import YOLO
run = sys.argv[1]
root = os.path.expanduser('~/mediinsight_project')
model = YOLO(f'{root}/results/yolo/{run}/weights/best.pt')
FRACTURE = 3
for split in ['val', 'test']:
    imgs = sorted(glob.glob(f'{root}/data/graz_yolo/images/{split}/*'))
    rows = []
    for i in range(0, len(imgs), 16):
        for r in model.predict(imgs[i:i+16], imgsz=640, conf=0.001, iou=0.6, verbose=False):
            c = r.boxes.cls.cpu().numpy(); s = r.boxes.conf.cpu().numpy()
            score = float(s[c == FRACTURE].max()) if (c == FRACTURE).any() else 0.0
            stem = os.path.splitext(os.path.basename(r.path))[0]
            lab = open(f'{root}/data/graz_yolo/labels/{split}/{stem}.txt').read().split('\n')
            y = int(any(l.split()[0] == str(FRACTURE) for l in lab if l.strip()))
            rows.append((stem, y, score))
    pd.DataFrame(rows, columns=['image', 'label', 'score']).to_csv(
        f'{root}/results/yolo/{run}/scores_{split}.csv', index=False)
print('SCORES DONE', run)

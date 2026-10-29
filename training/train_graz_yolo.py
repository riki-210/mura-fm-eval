import argparse, os
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
from ultralytics import YOLO
ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="yolo11m.pt")
ap.add_argument("--data", default=os.path.expanduser("~/mediinsight_project/data/graz_yolo/graz.yaml"))
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--fraction", type=float, default=1.0)
ap.add_argument("--patience", type=int, default=20)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--name", default="yolo11m_640_s1")
a = ap.parse_args()
proj = os.path.expanduser("~/mediinsight_project/results/yolo")
m = YOLO(a.weights)
m.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, fraction=a.fraction,
        patience=a.patience, seed=a.seed, workers=4, project=proj, name=a.name,
        exist_ok=True, plots=False, amp=True, cos_lr=True)
print("TRAIN DONE", os.path.join(proj, a.name, "weights", "best.pt"), flush=True)

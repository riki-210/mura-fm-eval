"""patch_fm_erase.py -- adds --erase-p to train_fm_dinov2.py: random-erasing augmentation (up to 3 random rectangles,
2-12% of the image each, each applied with probability erase-p) on TRAIN images only, so the model cannot rely on one
marker tag, one piece of hardware or one border artefact. Validation/test images are untouched."""
import ast, sys
f = "train_fm_dinov2.py"
s = open(f).read()
if "--erase-p" in s:
    print("already patched"); sys.exit(0)

def rep(old, new):
    global s
    assert s.count(old) == 1, f"pattern not found exactly once: {old[:70]!r}"
    s = s.replace(old, new)

rep("import torchvision.transforms.functional as TF\n",
    "import torchvision.transforms as TV\nimport torchvision.transforms.functional as TF\n")
rep("    def __init__(self, df, img_size, augment):\n",
    "    def __init__(self, df, img_size, augment, erase_p=0.0):\n")
rep("        self.img_size, self.augment = img_size, augment\n",
    "        self.img_size, self.augment = img_size, augment\n"
    "        self.erase_p = erase_p\n"
    "        self.eraser = TV.RandomErasing(p=erase_p, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0.0) if erase_p > 0 else None\n")
rep("        return (x - self.mean) / self.std, int(self.labels[i])\n",
    "        x = (x - self.mean) / self.std\n"
    "        if self.augment and self.eraser is not None:\n"
    "            for _ in range(3):\n"
    "                x = self.eraser(x)\n"
    "        return x, int(self.labels[i])\n")
rep('    ap.add_argument("--workers", type=int, default=4)\n',
    '    ap.add_argument("--workers", type=int, default=4)\n'
    '    ap.add_argument("--erase-p", type=float, default=0.0, help="random-erasing probability per patch (3 patches), train only")\n')
rep("    mk = lambda d, aug, shuf: DataLoader(MuraDS(d, args.img_size, aug), batch_size=args.batch_size, shuffle=shuf,",
    "    mk = lambda d, aug, shuf: DataLoader(MuraDS(d, args.img_size, aug, args.erase_p if aug else 0.0), batch_size=args.batch_size, shuffle=shuf,")
ast.parse(s)
open(f, "w").write(s)
print("patched OK")

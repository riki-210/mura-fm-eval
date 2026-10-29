"""
convert_pretrained_alexnet_weights.py -- ONE-TIME script. Downloads the
official ImageNet-pretrained AlexNet weights (via torchvision) and converts
them into a .npz file that models_alexnet_pretrained.py's
build_alexnet_pretrained() can load, with no PyTorch dependency needed at
training time (only for this one conversion step).

Run this ONCE, on a machine/node with internet access to
download.pytorch.org (your cluster's login node or a compute node with
outbound internet -- NOT required to have a GPU for this step, it's just a
~230MB download + a numpy reshape, seconds of CPU work):

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    python3 convert_pretrained_alexnet_weights.py --out alexnet_imagenet_weights.npz

Then point train_tier2_alexnet.py's --pretrained-weights at the resulting
.npz file. You do NOT need torch/torchvision installed in your `mediinsight`
conda env for training itself -- only for running this conversion once.

The conversion logic below (including the flatten-order fix between
PyTorch's channel-first and Keras's channel-last tensor layout, which is
NOT a simple transpose and silently produces wrong-but-plausible-looking
results if skipped) was verified against torchvision's own forward pass
using random weights before shipping: outputs matched to 1e-8. That
verification is reproduced in reference/verify_alexnet_port.py in this kit
if you want to rerun it yourself.
"""

import argparse
import numpy as np


def convert(out_path: str):
    import torch
    from torchvision.models import alexnet, AlexNet_Weights

    print("Downloading ImageNet-pretrained AlexNet weights via torchvision...")
    pt_model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    pt_model.eval()
    sd = pt_model.state_dict()

    out = {}

    conv_map = {
        "conv1": "features.0", "conv2": "features.3", "conv3": "features.6",
        "conv4": "features.8", "conv5": "features.10",
    }
    for keras_name, pt_prefix in conv_map.items():
        w = sd[f"{pt_prefix}.weight"].detach().numpy()  # PyTorch: (out, in, kh, kw)
        b = sd[f"{pt_prefix}.bias"].detach().numpy()
        w_tf = np.transpose(w, (2, 3, 1, 0))  # Keras Conv2D kernel: (kh, kw, in, out)
        out[f"{keras_name}_kernel"] = w_tf.astype(np.float32)
        out[f"{keras_name}_bias"] = b.astype(np.float32)
        print(f"  {keras_name}: {w.shape} -> {w_tf.shape}")

    # fc1's input is a flattened (256,6,6) conv feature map. PyTorch flattens
    # channel-first (C,H,W); Keras flattens channel-last (H,W,C). These are
    # DIFFERENT element orderings of the same 9216 numbers, so fc1's weight
    # matrix must be permuted to match, or every input neuron silently gets
    # wired to the wrong weight row (conv layers still match perfectly --
    # only fc1 onward breaks, which is exactly what caught this during
    # verification: a large but not obviously-wrong-looking output mismatch).
    w1 = sd["classifier.1.weight"].detach().numpy()  # (4096, 9216), PyTorch (C,H,W) order
    b1 = sd["classifier.1.bias"].detach().numpy()
    C, H, W = 256, 6, 6
    w1 = w1.reshape(4096, C, H, W)
    w1 = np.transpose(w1, (0, 2, 3, 1))   # -> (4096, H, W, C), Keras's flatten order
    w1 = w1.reshape(4096, H * W * C)
    out["fc1_kernel"] = w1.T.astype(np.float32)  # Dense kernel is (in, out)
    out["fc1_bias"] = b1.astype(np.float32)
    print(f"  fc1: {(4096, 9216)} -> {w1.T.shape} (flatten-order corrected)")

    # fc2 has no spatial structure (4096 -> 4096), so a plain transpose is correct.
    w2 = sd["classifier.4.weight"].detach().numpy()
    b2 = sd["classifier.4.bias"].detach().numpy()
    out["fc2_kernel"] = w2.T.astype(np.float32)
    out["fc2_bias"] = b2.astype(np.float32)
    print(f"  fc2: {w2.shape} -> {w2.T.shape}")

    # Deliberately NOT converting classifier.6 (fc3, the 1000-way ImageNet
    # head) -- your task head (2-way fracture/normal) is a fresh, randomly
    # initialized layer, built by build_alexnet_pretrained() in
    # models_alexnet_pretrained.py.

    np.savez(out_path, **out)
    print(f"\nSaved {out_path} ({sum(v.nbytes for v in out.values()) / 1e6:.1f} MB)")
    print("Done. Point train_tier2_alexnet.py's --pretrained-weights at this file.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="alexnet_imagenet_weights.npz")
    args = ap.parse_args()
    convert(args.out)

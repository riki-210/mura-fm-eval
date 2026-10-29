"""
pca_utils.py -- PCA front-end for the "AlexNet + PCA" Tier 2 configuration.

IMPORTANT INTERPRETATION NOTE (read before trusting this module):
The paper's description -- "PCA: retain components explaining 95% cumulative
variance, applied to input space before AlexNet training" -- is architecturally
ambiguous, and the original code is lost, so this is a reconstruction with an
explicit judgment call:

  There are two very different things this sentence could mean:
    (a) The classic AlexNet-paper "PCA colour jitter" (Krizhevsky et al. 2012):
        PCA over per-pixel RGB colour statistics, used only as a train-time
        colour AUGMENTATION, with the image passed through the normal
        convolutional AlexNet unchanged otherwise. This does not reduce
        dimensionality and does not explain a "cumulative variance retained"
        framing, which is why it does not seem to be what's meant here --
        but it is the most common meaning of "PCA" next to "AlexNet" in the
        literature, so it's worth ruling out explicitly.
    (b) A genuine dimensionality-reduction PCA over the (flattened, enhanced)
        image, retaining however many components explain 95% of variance,
        with THAT reduced vector used as the network's input instead of the
        raw image -- replacing AlexNet's convolutional feature-extraction
        stage with a PCA feature extractor, feeding directly into AlexNet's
        original fully-connected classifier head (4096-4096-softmax).

This module implements (b), since it is the only reading consistent with
"retain components explaining 95% cumulative variance" and "applied to the
input space" as literally written. If you have any surviving fragment of the
original code, a figure showing the AlexNet+PCA architecture diagram, or a
supplementary methods file that clarifies this, tell me and I will correct
this module -- the +7.99% accuracy contribution attributed to "PCA" in the
paper depends entirely on which of these two very different mechanisms was
actually used, and reproducing the wrong one will not reproduce that number.

To keep the flattened-vector PCA computationally tractable (a 224x224x3
image has 150,528 raw dimensions; a full covariance matrix over that many
dimensions is impractical on typical hardware), images are first downsampled
to PCA_INPUT_SIZE x PCA_INPUT_SIZE (grayscale, since the enhanced images are
single-channel replicated to 3 anyway) before PCA is fit or applied.
"""

import os
import numpy as np
import cv2
from sklearn.decomposition import IncrementalPCA
import joblib

PCA_INPUT_SIZE = 64  # downsample size before flattening for PCA (64*64=4096 raw dims)
PCA_VARIANCE_TARGET = 0.95
PCA_BATCH_SIZE = 256


def _load_for_pca(image_path: str, apply_enhancement: bool = True) -> np.ndarray:
    """Loads+enhances one image and returns a flattened (PCA_INPUT_SIZE^2,) vector."""
    from enhancement import enhance_image

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
    img = cv2.resize(img, (PCA_INPUT_SIZE, PCA_INPUT_SIZE), interpolation=cv2.INTER_AREA)
    if apply_enhancement:
        enhanced = enhance_image(img)
    else:
        enhanced = img.astype(np.float32) / 255.0
    return enhanced.flatten()


def fit_pca(image_paths, apply_enhancement: bool = True,
            variance_target: float = PCA_VARIANCE_TARGET,
            batch_size: int = PCA_BATCH_SIZE) -> IncrementalPCA:
    """
    Fits an IncrementalPCA (memory-friendly for large N) on the training-set
    images ONLY (never on val/test -- fit PCA inside the same train-only
    discipline as any other fold-fitted preprocessing step). Returns a PCA
    object retaining ALL components initially; the caller uses
    n_components_for_variance() below to determine how many components to
    actually keep to hit variance_target, then re-fits with that fixed count
    (IncrementalPCA needs n_components fixed up front, so this is a two-pass
    process: pass 1 to discover how many components are needed, pass 2 with
    that exact count).
    """
    n_features = PCA_INPUT_SIZE * PCA_INPUT_SIZE
    # Pass 1: fit with max feasible components to inspect the variance curve.
    max_components = min(n_features, len(image_paths), 512)

    # IncrementalPCA.partial_fit() requires n_samples >= n_components on
    # EVERY call (sklearn raises ValueError otherwise -- this is what broke
    # on the cluster: batch_size=256 < max_components=512 for XR_SHOULDER's
    # 6,286 training images). Force the chunking batch size to be at least
    # max_components so every partial_fit call is large enough, regardless
    # of the batch_size passed in / the PCA_BATCH_SIZE default.
    effective_batch_size = max(batch_size, max_components)
    if effective_batch_size > batch_size:
        print(f"  PCA: bumping batch size from {batch_size} to "
              f"{effective_batch_size} (must be >= n_components={max_components} "
              f"for IncrementalPCA.partial_fit).")

    ipca_probe = IncrementalPCA(n_components=max_components, batch_size=effective_batch_size)

    batch = []
    for i, p in enumerate(image_paths):
        batch.append(_load_for_pca(p, apply_enhancement))
        if len(batch) == effective_batch_size:
            ipca_probe.partial_fit(np.stack(batch))
            batch = []
        if (i + 1) % 500 == 0:
            print(f"  PCA fit pass 1: {i + 1}/{len(image_paths)} images processed")
    if len(batch) >= max_components:
        ipca_probe.partial_fit(np.stack(batch))
    elif len(batch) >= 1:
        print(f"  PCA fit pass 1: dropping final partial batch of {len(batch)} "
              f"images (< n_components={max_components}, cannot partial_fit it).")

    cum_var = np.cumsum(ipca_probe.explained_variance_ratio_)
    n_needed = int(np.searchsorted(cum_var, variance_target) + 1)
    n_needed = min(n_needed, max_components)
    print(f"PCA: {n_needed} components explain {cum_var[n_needed - 1]:.4f} "
          f"cumulative variance (target {variance_target}).")

    # Pass 2: refit with exactly n_needed components (this is the PCA object
    # actually used downstream). n_needed <= max_components <= effective_batch_size,
    # so reusing effective_batch_size here is always safe.
    ipca_final = IncrementalPCA(n_components=n_needed, batch_size=effective_batch_size)
    batch = []
    for i, p in enumerate(image_paths):
        batch.append(_load_for_pca(p, apply_enhancement))
        if len(batch) == effective_batch_size:
            ipca_final.partial_fit(np.stack(batch))
            batch = []
    if len(batch) >= n_needed:
        ipca_final.partial_fit(np.stack(batch))
    elif len(batch) >= 1:
        print(f"  PCA fit pass 2: dropping final partial batch of {len(batch)} "
              f"images (< n_needed={n_needed}, cannot partial_fit it).")

    return ipca_final


def transform_image(pca: IncrementalPCA, image_path: str,
                     apply_enhancement: bool = True) -> np.ndarray:
    """Returns the PCA-projected feature vector (n_components,) for one image."""
    vec = _load_for_pca(image_path, apply_enhancement).reshape(1, -1)
    return pca.transform(vec)[0].astype(np.float32)


def save_pca(pca: IncrementalPCA, path: str):
    joblib.dump(pca, path)


def load_pca(path: str) -> IncrementalPCA:
    return joblib.load(path)


if __name__ == "__main__":
    import argparse
    import pandas as pd

    ap = argparse.ArgumentParser(description="Fit PCA on a region's training split and save it.")
    ap.add_argument("--index-csv", required=True, help="CSV from data_utils.py (mura_ehr_index.csv)")
    ap.add_argument("--region", required=True, choices=["XR_ELBOW", "XR_HAND", "XR_SHOULDER"])
    ap.add_argument("--out", default=None, help="Output .joblib path (default: pca_<region>.joblib)")
    ap.add_argument("--max-images", type=int, default=2000,
                     help="Cap on number of training images used to fit PCA (for speed).")
    args = ap.parse_args()

    df = pd.read_csv(args.index_csv)
    train_df = df[(df["region"] == args.region) & (df["split"] == "train")]
    paths = train_df["image_path"].tolist()
    if len(paths) > args.max_images:
        rng = np.random.RandomState(42)
        paths = list(rng.choice(paths, size=args.max_images, replace=False))

    print(f"Fitting PCA on {len(paths)} training images from {args.region}...")
    pca = fit_pca(paths)

    out_path = args.out or f"pca_{args.region}.joblib"
    save_pca(pca, out_path)
    print(f"Saved fitted PCA to {out_path}")

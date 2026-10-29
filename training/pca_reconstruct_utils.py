"""
pca_reconstruct_utils.py -- PCA-RECONSTRUCTION front-end for Tier 2
("AlexNet + PCA"), implementing a third, previously-untested interpretation
of the paper's "PCA is applied to the input image space before AlexNet
training" sentence.

BACKGROUND -- why a third interpretation was needed:
pca_utils.py (the original reconstruction) implemented interpretation (b):
project each image down to a PCA feature vector (retaining components for
95% cumulative variance), then feed THAT VECTOR directly into a small
2-layer FC classifier (build_alexnet_pca_head: n_components -> 4096 -> 4096
-> softmax), replacing AlexNet's convolutional stage entirely. That
reconstruction has an architectural problem: Table 4 of the surviving PDF
lists "AlexNet+PCA" at **60M parameters** -- the size of the FULL AlexNet
(conv stack + classifier), not the ~19M a truncated PCA-vector-to-FC head
would have. The paper also describes PCA as "input-space preprocessing" /
"conditioning the input space" (Sections 1 and 3.3.2), language that reads
more naturally as denoising the IMAGE, not replacing the feature extractor.

This module implements that reading instead: PCA is fit on flattened
images, used to PROJECT then INVERSE-TRANSFORM (reconstruct) each image --
a standard PCA-based image denoising technique -- and the reconstructed
(still image-shaped) result is resized back up to the network's expected
input size and fed through the FULL AlexNet (conv stack intact), matching
the paper's 60M-parameter figure. This directly implements "PCA conditions
the input space" literally: the image itself is conditioned (denoised) by
projecting out the lower-variance components, not converted into a vector
that bypasses the convolutional stage.

The PCA fitting logic (IncrementalPCA, two-pass to hit a 95% cumulative
variance target, downsampled to PCA_INPUT_SIZE=64 grayscale for a tractable
covariance matrix) is copied unchanged from pca_utils.py -- only what
happens AFTER the projection differs: this module inverse-transforms back
to image space and upsamples, instead of returning the raw projected vector.
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
    """Loads+enhances one image and returns a flattened (PCA_INPUT_SIZE^2,) vector in [0,1]."""
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
    Identical fitting procedure to pca_utils.fit_pca (see that module for the
    detailed rationale on the two-pass IncrementalPCA approach and the
    batch-size-vs-n-components constraint) -- duplicated here rather than
    imported so this module has no import-order dependency on pca_utils.py.
    Fits on TRAINING images only.
    """
    n_features = PCA_INPUT_SIZE * PCA_INPUT_SIZE
    max_components = min(n_features, len(image_paths), 512)

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


def reconstruct_image(pca: IncrementalPCA, image_path: str, output_size: int,
                       apply_enhancement: bool = True) -> np.ndarray:
    """
    Projects the image onto the fitted PCA subspace and inverse-transforms
    it back to image space (the "denoising" step: components beyond the
    95%-variance cutoff, assumed to carry mostly normal-bone-texture noise
    per the paper's own stated rationale, are discarded and not reconstructed
    back in). Returns a (output_size, output_size, 3) float32 array in
    [0, 1], 3-channel-replicated -- same convention as
    data_utils.load_and_enhance, so it is a drop-in input for the same
    AlexNet models (build_alexnet / build_alexnet_pretrained) used elsewhere
    in this project.
    """
    vec = _load_for_pca(image_path, apply_enhancement).reshape(1, -1)
    projected = pca.transform(vec)
    reconstructed = pca.inverse_transform(projected)[0]  # (PCA_INPUT_SIZE^2,)
    reconstructed = np.clip(reconstructed, 0.0, 1.0)  # inverse_transform can overshoot [0,1] slightly
    img = reconstructed.reshape(PCA_INPUT_SIZE, PCA_INPUT_SIZE)
    img_up = cv2.resize(img, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
    img_up = np.clip(img_up, 0.0, 1.0).astype(np.float32)
    rgb = np.stack([img_up, img_up, img_up], axis=-1)
    return rgb.astype(np.float32)


def save_pca(pca: IncrementalPCA, path: str):
    joblib.dump(pca, path)


def load_pca(path: str) -> IncrementalPCA:
    return joblib.load(path)

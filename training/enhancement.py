"""
enhancement.py -- Four-stage image enhancement pipeline for MediInsight (Paper 2),
reconstructed from the methodology described in BoneFracture_Tier2_FINAL.pdf.

Stages, applied in order to each grayscale radiograph:
  1. Auto Canny edge detection    -- adaptive thresholds from the image's median intensity
  2. Gabor filtering              -- 6 orientations (0 to 150 deg in 30 deg steps),
                                      lambda=4px, sigma=2px; filtered responses are
                                      combined (max response across orientations)
  3. Watershed segmentation       -- distance-transform-based, to separate touching
                                      bone/soft-tissue regions
  4. Laplacian edge sharpening    -- final sharpening pass, added back onto the image

The output of this pipeline is a single-channel "enhanced" image, replicated to 3
channels (to match ImageNet-pretrained backbones' expected input) at the point of
use (see data_utils.py), not inside this module -- this module always returns a
single 2D float32 array in [0, 1] so it can be unit-tested and visualized on its own.

None of this has been run against the actual MURA images yet -- the original
code is lost, and this is a best-effort reconstruction from the paper's prose
description only. Once you can run this against a real sample, LOOK AT the
enhanced_preview() output for a handful of images before trusting it in training;
tune QC_* / GABOR_* / WATERSHED_* constants below if the enhanced images look wrong
(over-sharpened, edges blown out, watershed over-segmenting into many tiny regions).
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Stage 1: Auto Canny
# ---------------------------------------------------------------------------
def auto_canny(gray_u8: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """
    gray_u8: 2D uint8 grayscale image.
    Returns: 2D uint8 binary edge map (0/255), same shape as input.

    Thresholds are derived from the image's own median intensity, per the paper:
        lower = max(0,   (1 - sigma) * median)
        upper = min(255, (1 + sigma) * median)
    """
    m = float(np.median(gray_u8))
    lower = int(max(0, (1.0 - sigma) * m))
    upper = int(min(255, (1.0 + sigma) * m))
    edges = cv2.Canny(gray_u8, lower, upper)
    return edges


# ---------------------------------------------------------------------------
# Stage 2: Gabor filtering
# ---------------------------------------------------------------------------
GABOR_ORIENTATIONS_DEG = [0, 30, 60, 90, 120, 150]
GABOR_LAMBDA = 4.0     # wavelength, px
GABOR_SIGMA = 2.0      # gaussian envelope std, px
GABOR_KSIZE = 15       # kernel size (odd); large enough for lambda/sigma above
GABOR_GAMMA = 0.5      # spatial aspect ratio (standard default)
GABOR_PSI = 0.0        # phase offset


def _gabor_kernel(theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    return cv2.getGaborKernel(
        (GABOR_KSIZE, GABOR_KSIZE), GABOR_SIGMA, theta,
        GABOR_LAMBDA, GABOR_GAMMA, GABOR_PSI, ktype=cv2.CV_32F,
    )


_GABOR_KERNELS = [_gabor_kernel(t) for t in GABOR_ORIENTATIONS_DEG]


def gabor_filter_bank(gray_u8: np.ndarray) -> np.ndarray:
    """
    Applies all 6 oriented Gabor filters and takes the pixelwise maximum response
    (the standard way to combine an orientation-selective filter bank into one
    orientation-invariant response map). Returns a float32 array, same shape as
    input, normalized to [0, 1].
    """
    gray_f = gray_u8.astype(np.float32) / 255.0
    responses = [cv2.filter2D(gray_f, cv2.CV_32F, k) for k in _GABOR_KERNELS]
    combined = np.max(np.stack(responses, axis=0), axis=0)
    combined = np.abs(combined)
    if combined.max() > 0:
        combined = combined / combined.max()
    return combined


# ---------------------------------------------------------------------------
# Stage 3: Watershed segmentation
# ---------------------------------------------------------------------------
def watershed_segment(gray_u8: np.ndarray) -> np.ndarray:
    """
    Distance-transform-based watershed, used here to separate touching
    bone/soft-tissue regions rather than to produce a final fracture mask.
    Returns a float32 "segmentation-boundary-emphasized" image in [0, 1]:
    watershed boundaries are overlaid as strong edges onto a lightly blurred
    copy of the original, since the paper uses this stage for enhancement
    (making structural boundaries more distinct for later stages), not as the
    final localization output (that is the separate U-Net stage in Paper 2).
    """
    # Otsu threshold -> binary foreground mask
    _, binary = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure foreground via distance transform
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    if dist.max() > 0:
        _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    else:
        sure_fg = np.zeros_like(opening)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    gray_bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(gray_bgr, markers)

    boundaries = np.zeros_like(gray_u8, dtype=np.float32)
    boundaries[markers == -1] = 1.0

    base = cv2.GaussianBlur(gray_u8, (3, 3), 0).astype(np.float32) / 255.0
    out = np.clip(base * 0.7 + boundaries * 0.3, 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Stage 4: Laplacian edge sharpening
# ---------------------------------------------------------------------------
def laplacian_sharpen(image_float: np.ndarray, weight: float = 0.5) -> np.ndarray:
    """
    image_float: 2D float32 array in [0, 1].
    Final sharpening pass: adds a weighted Laplacian (high-frequency detail)
    back onto the image. Returns float32 in [0, 1].
    """
    img_u8 = np.clip(image_float * 255.0, 0, 255).astype(np.uint8)
    lap = cv2.Laplacian(img_u8, cv2.CV_32F, ksize=3)
    lap = lap / (np.abs(lap).max() + 1e-8)  # normalize to [-1, 1] roughly
    sharpened = image_float + weight * lap
    return np.clip(sharpened, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def enhance_image(gray_u8: np.ndarray) -> np.ndarray:
    """
    Runs the full four-stage pipeline (Auto Canny -> Gabor -> Watershed ->
    Laplacian) and returns a single float32 image in [0, 1], same H x W as input.

    The four stages are combined (rather than each stage overwriting the last)
    as a weighted blend of: the watershed-enhanced base image, the Gabor
    response map (structural/orientation detail), and the Canny edge map
    (crisp boundaries) -- then Laplacian-sharpened as the final step. This
    combination strategy is an interpretation of the paper's four-stage
    description (which does not give an explicit blending formula); adjust
    CANNY_WEIGHT / GABOR_WEIGHT below after visually inspecting output on a
    handful of real MURA images.
    """
    assert gray_u8.ndim == 2, "enhance_image expects a single-channel image"
    assert gray_u8.dtype == np.uint8, "enhance_image expects uint8 input"

    CANNY_WEIGHT = 0.15
    GABOR_WEIGHT = 0.25
    WATERSHED_WEIGHT = 0.60

    edges = auto_canny(gray_u8).astype(np.float32) / 255.0
    gabor = gabor_filter_bank(gray_u8)
    watershed = watershed_segment(gray_u8)

    combined = (
        CANNY_WEIGHT * edges
        + GABOR_WEIGHT * gabor
        + WATERSHED_WEIGHT * watershed
    )
    combined = np.clip(combined, 0.0, 1.0)

    final = laplacian_sharpen(combined, weight=0.5)
    return final.astype(np.float32)


def enhanced_preview(path: str, out_path: str = None):
    """
    Convenience function for visually sanity-checking the pipeline on one
    real image before trusting it in a training run. Saves a side-by-side
    [original | canny | gabor | watershed | final] panel to out_path (or
    displays via cv2 if out_path is None and a display is available).
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    edges = auto_canny(img)
    gabor = (gabor_filter_bank(img) * 255).astype(np.uint8)
    watershed = (watershed_segment(img) * 255).astype(np.uint8)
    final = (enhance_image(img) * 255).astype(np.uint8)

    panel = np.hstack([img, edges, gabor, watershed, final])
    if out_path:
        cv2.imwrite(out_path, panel)
        print(f"Preview panel written to {out_path}")
    else:
        return panel


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Sanity-check the enhancement pipeline on one image.")
    ap.add_argument("image_path")
    ap.add_argument("--out", default="enhancement_preview.png")
    args = ap.parse_args()
    enhanced_preview(args.image_path, args.out)

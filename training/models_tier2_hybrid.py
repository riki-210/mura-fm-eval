"""
models_tier2_hybrid.py -- a new, project-specific architecture: a
Dual-Branch CNN+Transformer fusion network for Tier 2 fracture
classification ("HybridFusionNet").

RATIONALE: CNNs (DenseNet, ConvNeXt, etc.) and Vision Transformers learn
different, complementary kinds of visual representation. A CNN's local,
translation-equivariant convolutions are well-suited to texture-level cues
(cortical bone discontinuities, trabecular pattern disruption) -- the kind
of signal a radiologist reads at high zoom. A ViT's global self-attention
instead captures long-range spatial relationships across the whole image
(overall bone alignment, joint positioning) -- the kind of signal read at
a glance. Neither representation alone is obviously sufficient for fracture
detection, which involves both a global "does this bone look right" read
and a local "is there a discontinuity here" read. HybridFusionNet
concatenates both branches' pretrained embeddings and lets a shared
classification head learn how to weigh them, rather than committing to one
representation family.

This is NOT a published architecture reproduced from a citation -- it is a
new combination assembled for this project, built from two independently
pretrained, well-established components (a CNN backbone from
models_tier2_strong.py's family, a ViT from models_tier2_vit.py's family).
Label it in the thesis/paper as a novel fusion architecture proposed for
this reproduction's accuracy-push experiments, not as "the" state of the
art -- its value is empirical (see README_accuracy_push.md's reporting
section) and should be reported as such: a controlled comparison against
the single-branch CNN and single-branch ViT results, on the same data,
same preprocessing, same training protocol.

Architecture:
    image -> [CNN branch (e.g. DenseNet169), pretrained]  -> embedding_cnn (e.g. 1920-dim)
    image -> [ViT branch (ViT-B16), pretrained]            -> embedding_vit (768-dim)
    concat(embedding_cnn, embedding_vit) -> Dense(512, relu) -> Dropout -> Dense(2, softmax)

Both branches can be frozen (phase 1: train only the fusion head) or
unfrozen (phase 2: fine-tune both branches jointly at a low LR) -- same
two-phase recipe as the other two scripts, for a fair comparison.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from models_tier2_strong import _BACKBONES
from models_tier2_vit import extract_vit_embedding_model


def build_hybrid_fusion_classifier(cnn_backbone: str = "DenseNet169", vit_variant: str = "vit_b16",
                                    img_size: int = 224, num_classes: int = 2,
                                    dropout: float = 0.5, l2_reg: float = 0.0,
                                    freeze_backbones: bool = True,
                                    fusion_hidden_dim: int = 512) -> tf.keras.Model:
    """
    Builds the dual-branch CNN+ViT fusion network described in this module's
    docstring. `freeze_backbones` controls whether BOTH branches' pretrained
    weights train or stay frozen -- train_tier2_hybrid.py uses this for its
    two-phase schedule, exactly mirroring the other two scripts.
    """
    if cnn_backbone not in _BACKBONES:
        raise ValueError(f"Unknown cnn_backbone {cnn_backbone!r}; choices: {list(_BACKBONES)}")
    cnn_cls, cnn_preprocess = _BACKBONES[cnn_backbone]

    inputs = layers.Input(shape=(img_size, img_size, 3), name="image")

    # --- CNN branch ---
    cnn_base = cnn_cls(include_top=False, weights="imagenet",
                        input_shape=(img_size, img_size, 3), pooling="avg")
    cnn_base.trainable = not freeze_backbones
    cnn_base._name = f"cnn_branch_{cnn_backbone}"
    x_cnn = cnn_preprocess(inputs * 255.0)
    embedding_cnn = cnn_base(x_cnn, training=not freeze_backbones)

    # --- ViT branch ---
    vit_embedding_model = extract_vit_embedding_model(img_size=img_size, vit_variant=vit_variant,
                                                        trainable=not freeze_backbones)
    embedding_vit = vit_embedding_model(inputs, training=not freeze_backbones)

    # --- Fusion head ---
    fused = layers.Concatenate(name="fusion_concat")([embedding_cnn, embedding_vit])
    x = layers.Dense(fusion_hidden_dim, activation="relu",
                      kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None)(fused)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(
        num_classes, activation="softmax", name="fracture_output",
        kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None,
    )(x)

    return models.Model(inputs, outputs, name=f"tier2_hybrid_{cnn_backbone}_{vit_variant}")

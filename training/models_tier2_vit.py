"""
models_tier2_vit.py -- Vision Transformer (ViT-B16) Tier 2 fracture
classifier, the modern-transformer counterpart to the CNN backbones in
models_tier2_strong.py.

Why ViT-B16 instead of CvT specifically: Paper 3's surviving manuscript
reports CvT (Convolutional vision Transformer) as its best fracture
classifier (81.46% val acc), but no CvT training code survived alongside
the manuscript -- only the published result. Rebuilding CvT exactly would
mean depending on HuggingFace `transformers`' TF CvT implementation, which
recent transformers releases have been steadily dropping TensorFlow support
for (version churn risk on a long-running cluster job). `vit-keras` is a
small, stable, pure tf.keras package providing ImageNet21k+ImageNet2012
pretrained ViT-B16/B32/L16/L32 with no cross-framework dependency -- the
safer choice for a reliable multi-hour cluster run, and still the same
*family* of model (a global-attention vision transformer) as CvT, so it
answers the same "does a transformer beat a CNN here" question Paper 3
already raised.

Install once, on a node with internet access (small pure-Python package,
no large binary download beyond the pretrained weights themselves):
    pip install vit-keras
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from vit_keras import vit


def build_vit_fracture_classifier(img_size: int = 224, num_classes: int = 2,
                                   dropout: float = 0.5, l2_reg: float = 0.0,
                                   freeze_backbone: bool = True,
                                   vit_variant: str = "vit_b16",
                                   domain_weights_path: str = None) -> tf.keras.Model:
    """
    Builds ViT(ImageNet21k+2012-pretrained, include_top=False, 768-dim CLS
    embedding) -> Dropout -> Dense(num_classes, softmax). Same
    freeze/unfreeze contract as build_strong_fracture_classifier, for the
    same two-phase fine-tuning schedule in train_tier2_vit.py.
    """
    vit_fn = {"vit_b16": vit.vit_b16, "vit_b32": vit.vit_b32,
              "vit_l16": vit.vit_l16, "vit_l32": vit.vit_l32}[vit_variant]

    if domain_weights_path:
        # GRAZPEDWRI-DX-pretrained backbone (Keras 3 native .weights.h5) from
        # pretrain_graz_vit.py: plain strict load into an identical ViT.
        # pretrained_top=False is required (vit-keras defaults it to True,
        # which asserts include_top=True and pretrained=True).
        base = vit_fn(image_size=img_size, include_top=False, pretrained=False,
                       pretrained_top=False)
        base.load_weights(domain_weights_path)
        print(f"Loaded domain-pretrained ViT weights from {domain_weights_path}")
    else:
        base = vit_fn(image_size=img_size, include_top=False, pretrained=True,
                       pretrained_top=False, weights="imagenet21k+imagenet2012")
    base.trainable = not freeze_backbone

    inputs = layers.Input(shape=(img_size, img_size, 3))
    # vit-keras's own preprocess_inputs expects [0,255] and internally does
    # the standard `mode="tf"` (x/127.5 - 1) ImageNet scaling.
    x = vit.preprocess_inputs(inputs * 255.0)
    x = base(x, training=not freeze_backbone)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(
        num_classes, activation="softmax", name="fracture_output",
        kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None,
    )(x)
    return models.Model(inputs, outputs, name=f"tier2_{vit_variant}")


def extract_vit_embedding_model(img_size: int = 224, vit_variant: str = "vit_b16",
                                 trainable: bool = False) -> tf.keras.Model:
    """
    Returns a model mapping [0,1]-range (img_size,img_size,3) images directly
    to the 768-dim pretrained ViT CLS embedding, with no classification head.
    Used by models_tier2_hybrid.py to pull the ViT branch's feature vector
    for fusion with a CNN branch's feature vector.
    """
    vit_fn = {"vit_b16": vit.vit_b16, "vit_b32": vit.vit_b32,
              "vit_l16": vit.vit_l16, "vit_l32": vit.vit_l32}[vit_variant]
    base = vit_fn(image_size=img_size, include_top=False, pretrained=True,
                   pretrained_top=False, weights="imagenet21k+imagenet2012")
    base.trainable = trainable

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = vit.preprocess_inputs(inputs * 255.0)
    embedding = base(x, training=trainable)
    return models.Model(inputs, embedding, name=f"{vit_variant}_embedding")

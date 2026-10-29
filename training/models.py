"""
models.py -- Model architectures for Paper 2 (MediInsight): Tier 1 ResNet50
region classifier, Tier 2 AlexNet (+ optional PCA front-end) fracture
classifier, and a VGGNet baseline for the ablation/comparison table.

All models built with tf.keras. ResNet50 and VGG16 use ImageNet-pretrained
weights (keras.applications), matching the paper's use of pretrained
backbones; AlexNet is trained from scratch (no standard ImageNet-pretrained
AlexNet ships with Keras), consistent with the paper treating it as the
from-scratch, PCA-augmented architecture at the center of the Tier 2 result.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


# ---------------------------------------------------------------------------
# Tier 1: ResNet50 region classifier (elbow / hand / shoulder)
# ---------------------------------------------------------------------------
def build_resnet50_region_classifier(num_classes: int = 3, img_size: int = 224,
                                      freeze_backbone: bool = False) -> tf.keras.Model:
    base = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet",
        input_shape=(img_size, img_size, 3), pooling="avg",
    )
    base.trainable = not freeze_backbone

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = tf.keras.applications.resnet50.preprocess_input(inputs * 255.0)
    x = base(x, training=not freeze_backbone)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="region_output")(x)
    return models.Model(inputs, outputs, name="tier1_resnet50")


# ---------------------------------------------------------------------------
# AlexNet (from scratch), for Tier 2
# ---------------------------------------------------------------------------
def build_alexnet(img_size: int = 224, num_classes: int = 2,
                   dropout: float = 0.5, l2_reg: float = 0.0,
                   input_channels: int = 96) -> tf.keras.Model:
    """
    input_channels: number of channels the network's first conv layer expects.
    When PCA is applied as a front-end (see PCAProjection below), the "image"
    fed to this network is actually a flattened-then-reshaped PCA projection,
    not a real RGB image -- see build_alexnet_with_pca for how the two are
    connected. When PCA is off, input_channels=3 (plain RGB) is used instead.

    dropout / l2_reg are exposed as arguments because the paper's 5-config
    ablation (Table in Section on hyperparameter search) sweeps exactly these
    two knobs (dropout in {0.5, 0.65, 0.8}, L2 in {0, 1e-4, 1e-3}).
    """
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

    inputs = layers.Input(shape=(img_size, img_size, input_channels))
    x = layers.Conv2D(96, 11, strides=4, activation="relu", kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2)(x)

    x = layers.Conv2D(256, 5, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2)(x)

    x = layers.Conv2D(384, 3, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = layers.Conv2D(384, 3, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = layers.MaxPooling2D(3, strides=2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(4096, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="fracture_output")(x)
    return models.Model(inputs, outputs, name="tier2_alexnet")


def build_alexnet_pca_head(n_pca_components: int, num_classes: int = 2,
                            dropout: float = 0.5, l2_reg: float = 0.0) -> tf.keras.Model:
    """
    The "AlexNet + PCA" Tier 2 variant (see pca_utils.py for the interpretation
    note on what "PCA" means here). Takes a pre-computed PCA feature vector
    (n_pca_components,) as input -- NOT a raw image -- and passes it through
    the same fully-connected classifier head AlexNet uses after its conv
    stack (4096-4096-softmax), with the same configurable dropout/L2 as
    build_alexnet() so the 5-config ablation grid applies identically to
    both the plain-image and PCA-front-end variants.
    """
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

    inputs = layers.Input(shape=(n_pca_components,))
    x = layers.Dense(4096, activation="relu", kernel_regularizer=reg)(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(4096, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="fracture_output")(x)
    return models.Model(inputs, outputs, name="tier2_alexnet_pca")


def build_vgg16_baseline(img_size: int = 224, num_classes: int = 2,
                          freeze_backbone: bool = False) -> tf.keras.Model:
    """VGGNet baseline for the comparison table (72.22% published figure)."""
    base = tf.keras.applications.VGG16(
        include_top=False, weights="imagenet",
        input_shape=(img_size, img_size, 3), pooling="avg",
    )
    base.trainable = not freeze_backbone

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = tf.keras.applications.vgg16.preprocess_input(inputs * 255.0)
    x = base(x, training=not freeze_backbone)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="vgg16_baseline")


def build_resnet50_baseline(img_size: int = 224, num_classes: int = 2,
                             freeze_backbone: bool = False) -> tf.keras.Model:
    """ResNet50 as a *fracture classifier* baseline for the comparison table
    (82.01% published figure) -- distinct in role from the Tier 1 region
    classifier above, though architecturally identical."""
    return build_resnet50_region_classifier(
        num_classes=num_classes, img_size=img_size, freeze_backbone=freeze_backbone
    )

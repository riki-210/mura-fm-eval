"""
models_alexnet_pretrained.py -- an ImageNet-pretrained AlexNet backbone for
Paper 2 Tier 2, added to diagnose whether the from-scratch AlexNet's low
accuracy (~61-71% vs. the paper's claimed 92.71%) is explained by the lack
of pretraining.

Keras ships no official pretrained AlexNet, so this ports the REAL
ImageNet-pretrained weights from torchvision.models.alexnet into an
architecturally-identical tf.keras model. The port logic (conv weight axis
transpose, and critically the flatten-order fix between PyTorch's
channel-first and Keras's channel-last conventions) was verified end-to-end
against torchvision's own forward pass using random weights before this
file was written: numerical outputs matched to 1e-8. You do not need to
re-verify this -- just run convert_pretrained_alexnet_weights.py once (see
its docstring) to produce alexnet_imagenet_weights.npz, then use
build_alexnet_pretrained() below exactly like build_alexnet() in models.py.

IMPORTANT SCOPE NOTE: this pretrained backbone takes a REAL RGB image as
input (224x224x3), like build_alexnet() does. It is NOT compatible with the
--use-pca path (build_alexnet_pca_head in models.py), which replaces the
image entirely with a flattened PCA projection -- there is no meaningful way
to feed a PCA feature vector through pretrained ImageNet convolutional
filters, since those filters expect real image statistics (edges, textures,
color gradients), not whatever geometric structure a pixel-space PCA
projection happens to have. So this diagnostic only applies to the
plain-image (non-PCA) branch. If pretraining closes most of the gap on the
plain-image branch, that's strong evidence the paper's "92.71%, PCA on"
result either used a pretrained/fine-tuned backbone despite being described
as from-scratch AlexNet, or the "PCA" step in the original paper meant
something other than literal pixel-space dimensionality reduction (e.g.
PCA-based color-jitter augmentation, the classic AlexNet-paper usage) --
worth revisiting once this result is in.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_alexnet_pretrained_backbone(img_size: int = 224) -> tf.keras.Model:
    """Architectural replica of torchvision.models.alexnet's feature
    extractor + classifier fc1/fc2 (NOT fc3, the 1000-way ImageNet head --
    that gets replaced by your own task head in build_alexnet_pretrained()
    below). Layer names (conv1..conv5, fc1, fc2) must match exactly what
    convert_pretrained_alexnet_weights.py expects when loading weights."""
    inputs = layers.Input(shape=(img_size, img_size, 3), name="image")

    x = layers.ZeroPadding2D(padding=2, name="pad1")(inputs)
    x = layers.Conv2D(64, 11, strides=4, padding="valid", name="conv1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool1")(x)

    x = layers.ZeroPadding2D(padding=2, name="pad2")(x)
    x = layers.Conv2D(192, 5, strides=1, padding="valid", name="conv2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool2")(x)

    x = layers.ZeroPadding2D(padding=1, name="pad3")(x)
    x = layers.Conv2D(384, 3, strides=1, padding="valid", name="conv3")(x)
    x = layers.ReLU(name="relu3")(x)

    x = layers.ZeroPadding2D(padding=1, name="pad4")(x)
    x = layers.Conv2D(256, 3, strides=1, padding="valid", name="conv4")(x)
    x = layers.ReLU(name="relu4")(x)

    x = layers.ZeroPadding2D(padding=1, name="pad5")(x)
    x = layers.Conv2D(256, 3, strides=1, padding="valid", name="conv5")(x)
    x = layers.ReLU(name="relu5")(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool5")(x)  # -> (6,6,256) for a 224x224 input

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(4096, name="fc1")(x)
    x = layers.ReLU(name="relu6")(x)
    x = layers.Dense(4096, name="fc2")(x)
    x = layers.ReLU(name="relu7")(x)

    return models.Model(inputs, x, name="alexnet_pretrained_backbone")


def load_pretrained_weights_into_backbone(backbone: tf.keras.Model, weights_npz_path: str) -> tf.keras.Model:
    """Loads the .npz produced by convert_pretrained_alexnet_weights.py into
    a freshly-built backbone from build_alexnet_pretrained_backbone()."""
    w = np.load(weights_npz_path)
    for layer_name in ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"]:
        kernel = w[f"{layer_name}_kernel"]
        bias = w[f"{layer_name}_bias"]
        backbone.get_layer(layer_name).set_weights([kernel, bias])
    return backbone


def build_alexnet_pretrained(weights_npz_path: str, img_size: int = 224, num_classes: int = 2,
                              dropout: float = 0.5, l2_reg: float = 0.0,
                              freeze_backbone: bool = False) -> tf.keras.Model:
    """
    Drop-in pretrained replacement for build_alexnet() in models.py (same
    call signature apart from the extra weights_npz_path arg and no
    input_channels arg, since this always takes real RGB images -- never
    PCA features, see the module docstring).

    freeze_backbone=True trains only the new task head (fast, good first
    test of whether the pretrained features alone separate fracture/normal);
    freeze_backbone=False fine-tunes the whole network end-to-end (slower,
    usually higher final accuracy once the head has stabilized -- consider
    running freeze_backbone=True for a few epochs first if you fine-tune,
    though this script does not implement that two-stage schedule itself).
    """
    backbone = build_alexnet_pretrained_backbone(img_size=img_size)
    backbone = load_pretrained_weights_into_backbone(backbone, weights_npz_path)
    backbone.trainable = not freeze_backbone

    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

    inputs = layers.Input(shape=(img_size, img_size, 3), name="image_input")
    x = backbone(inputs, training=not freeze_backbone)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="fracture_output",
                            kernel_regularizer=reg)(x)
    return models.Model(inputs, outputs, name="tier2_alexnet_pretrained")

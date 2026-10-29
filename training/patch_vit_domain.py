import sys

def patch(path, pairs):
    s = open(path).read()
    if "domain_weights_path" in s:
        print(f"{path}: already patched, skipping"); return
    for old, new in pairs:
        assert s.count(old) == 1, f"{path}: anchor not found exactly once:\n{old[:90]}"
        s = s.replace(old, new)
    open(path, "w").write(s)
    print(f"{path}: patched")

patch("models_tier2_vit.py", [
("""                                   freeze_backbone: bool = True,
                                   vit_variant: str = "vit_b16") -> tf.keras.Model:""",
"""                                   freeze_backbone: bool = True,
                                   vit_variant: str = "vit_b16",
                                   domain_weights_path: str = None) -> tf.keras.Model:"""),
("""    base = vit_fn(image_size=img_size, include_top=False, pretrained=True,
                   pretrained_top=False, weights="imagenet21k+imagenet2012")
    base.trainable = not freeze_backbone

    inputs = layers.Input(shape=(img_size, img_size, 3))
    # vit-keras's own""",
"""    if domain_weights_path:
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
    # vit-keras's own"""),
])

patch("train_tier2_vit.py", [
("""                      save_probs: bool = False) -> dict:""",
 """                      save_probs: bool = False, domain_weights_path: str = None) -> dict:"""),
("""l2_reg=l2_reg, freeze_backbone=True, vit_variant=vit_variant)""",
 """l2_reg=l2_reg, freeze_backbone=True, vit_variant=vit_variant,
                                           domain_weights_path=domain_weights_path)"""),
("""l2_reg=l2_reg, freeze_backbone=False, vit_variant=vit_variant)""",
 """l2_reg=l2_reg, freeze_backbone=False, vit_variant=vit_variant,
                                            domain_weights_path=domain_weights_path)"""),
("""        "model": f"ViT-{vit_variant}",""",
 """        "model": f"ViT-{vit_variant}",
        "domain_weights_path": domain_weights_path,"""),
("""    ap.add_argument("--save-probs", action="store_true")
    args""",
 """    ap.add_argument("--save-probs", action="store_true")
    ap.add_argument("--domain-weights", default=None,
                     help="GRAZPEDWRI-DX-pretrained ViT backbone (.weights.h5) from pretrain_graz_vit.py.")
    args"""),
("""        save_probs=args.save_probs,
    )""",
 """        save_probs=args.save_probs, domain_weights_path=args.domain_weights,
    )"""),
])

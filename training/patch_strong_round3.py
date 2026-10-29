path = "train_tier2_strong.py"
s = open(path).read()
if "predict_with_flip_tta" in s:
    print(f"{path}: already patched, skipping"); raise SystemExit

def sub(old, new):
    global s
    assert s.count(old) == 1, f"anchor not found exactly once:\n{old[:100]}"
    s = s.replace(old, new)

sub("def train_one_config(train_df,",
'''def predict_with_flip_tta(model, ds):
    """Average softmax over identity / h-flip / v-flip / h+v-flip of every batch.
    Returns (tta_probs, plain_probs). The training augmentation applies random
    left-right AND up-down flips, so all four views are in-distribution."""
    import tensorflow as tf
    fns = [lambda x: x,
           tf.image.flip_left_right,
           tf.image.flip_up_down,
           lambda x: tf.image.flip_up_down(tf.image.flip_left_right(x))]
    all_probs = []
    for fn in fns:
        view = ds.map(lambda x, y, fn=fn: (fn(x), y))
        all_probs.append(model.predict(view, verbose=0))
    return np.mean(all_probs, axis=0), all_probs[0]


def train_one_config(train_df,''')

sub("domain_weights_path: str = None, save_probs: bool = False) -> dict:",
    "domain_weights_path: str = None, save_probs: bool = False,\n                      tta: bool = False, seed: int = None) -> dict:")

sub("    if save_probs:\n", "    tta_info = {}\n    if save_probs or tta:\n")

sub('        print(f"Saved val/test probabilities + labels to {output_dir} for ensembling.")\n\n    result = {',
'''        print(f"Saved val/test probabilities + labels to {output_dir} for ensembling.")

    if tta:
        val_tta, val_plain = predict_with_flip_tta(model2, val_ds)
        test_tta, test_plain = predict_with_flip_tta(model2, test_ds)
        np.save(os.path.join(output_dir, "val_probs_noaug.npy"), val_plain)
        np.save(os.path.join(output_dir, "test_probs_noaug.npy"), test_plain)
        np.save(os.path.join(output_dir, "val_probs.npy"), val_tta)      # ensembling/study-level scripts read these
        np.save(os.path.join(output_dir, "test_probs.npy"), test_tta)
        test_y = test_df["fracture_label"].values
        val_y = val_df["fracture_label"].values
        tta_info = {
            "tta": "flip4",
            "test_accuracy_no_tta": float((np.argmax(test_plain, 1) == test_y).mean()),
            "test_accuracy_tta": float((np.argmax(test_tta, 1) == test_y).mean()),
            "val_accuracy_no_tta": float((np.argmax(val_plain, 1) == val_y).mean()),
            "val_accuracy_tta": float((np.argmax(val_tta, 1) == val_y).mean()),
        }
        print(f"TTA (flip x4): test_acc {tta_info['test_accuracy_no_tta']:.4f} -> {tta_info['test_accuracy_tta']:.4f}")

    result = {''')

sub('        "test_loss": float(test_loss),\n    }',
    '        "test_loss": float(test_loss),\n        "img_size": img_size,\n        "seed": seed,\n        **tta_info,\n    }')

sub('    ap.add_argument("--save-probs", action="store_true",',
'''    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=None,
                     help="Seeds TF/numpy/python and the dataset shuffle order (default: leave unseeded as before).")
    ap.add_argument("--tta", action="store_true",
                     help="Flip-x4 test-time augmentation; saves TTA probs as val/test_probs.npy and plain ones as *_noaug.npy.")
    ap.add_argument("--save-probs", action="store_true",''')

sub("    args = ap.parse_args()\n\n    df = pd.read_csv(args.index_csv)",
'''    args = ap.parse_args()

    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
        import data_utils
        data_utils.RNG_SEED = args.seed   # build_tf_dataset_v2 reads this at call time (shuffle order)
        print(f"Seed set to {args.seed}")

    df = pd.read_csv(args.index_csv)''')

sub("domain_weights_path=args.domain_weights, save_probs=args.save_probs,\n    )",
    "domain_weights_path=args.domain_weights, save_probs=args.save_probs,\n        img_size=args.img_size, tta=args.tta, seed=args.seed,\n    )")

open(path, "w").write(s)
print(f"{path}: patched")

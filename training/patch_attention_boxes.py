path = "attention_boxes.py"
s = open(path).read()
if "boxes exist, skipping" in s:
    print("already patched"); raise SystemExit

def sub(old, new):
    global s
    assert s.count(old) == 1, f"anchor not found exactly once:\n{old[:80]}"
    s = s.replace(old, new)

sub("    args = ap.parse_args()\n    tf.random.set_seed(0)\n",
    "    args = ap.parse_args()\n    if os.path.exists(args.out_npz):\n"
    "        print(f\"{args.out_npz}: boxes exist, skipping box computation\"); return\n    tf.random.set_seed(0)\n")
sub("        montage(reg[reg[\"split\"] == \"test\"], bp, args.img_size, args.montage_png)\n",
    "        try:\n            montage(reg[reg[\"split\"] == \"test\"], bp, args.img_size, args.montage_png)\n"
    "        except Exception as e:  # montage is optional; never let it kill the job\n"
    "            print(f\"montage skipped ({type(e).__name__}: {e})\")\n")
open(path, "w").write(s)
print("patched", path)

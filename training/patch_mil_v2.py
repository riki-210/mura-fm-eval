"""patch_mil_v2.py -- stabilise train_fm_mil.py: per-view LayerNorm before pooling, gradient accumulation (--accum),
and warm start of the backbone from an image-level DINOv2 checkpoint (--init-from)."""
import ast, sys
f = "train_fm_mil.py"
s = open(f).read()
if "--init-from" in s:
    print("already patched"); sys.exit(0)

def rep(old, new):
    global s
    assert s.count(old) == 1, f"pattern not found exactly once: {old[:60]!r}"
    s = s.replace(old, new)

rep("        self.D = D\n", "        self.D = D\n        self.view_norm = nn.LayerNorm(D)\n")
rep("        h[mask] = f.float()\n", "        h[mask] = self.view_norm(f.float())\n")
rep("def run_epoch(model, loader, device, amp, opt, sched, class_w, frozen):",
    "def run_epoch(model, loader, device, amp, opt, sched, class_w, frozen, accum=1):")
rep("""    for x, mask, y in loader:
        x, mask, y = x.to(device, non_blocking=True), mask.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(x, mask, frozen_backbone=frozen)
        loss = F.cross_entropy(logits.float(), y, weight=class_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()
        loss_sum += float(loss) * len(y); tot += len(y); correct += int((logits.argmax(1) == y).sum())
""", """    opt.zero_grad(set_to_none=True)
    n_it = len(loader)
    for it, (x, mask, y) in enumerate(loader):
        x, mask, y = x.to(device, non_blocking=True), mask.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(x, mask, frozen_backbone=frozen)
        loss = F.cross_entropy(logits.float(), y, weight=class_w)
        (loss / accum).backward()
        if (it + 1) % accum == 0 or (it + 1) == n_it:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            opt.zero_grad(set_to_none=True)
        loss_sum += float(loss.detach()) * len(y); tot += len(y); correct += int((logits.argmax(1) == y).sum().item())
""")
rep('    ap.add_argument("--workers", type=int, default=4)\n',
    '    ap.add_argument("--workers", type=int, default=4)\n'
    '    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps (effective batch = batch-size x accum studies)")\n'
    '    ap.add_argument("--init-from", default="", help="best_model.pt of an image-level DINOv2 run: initialise the BACKBONE from it")\n')
rep("    model = MILNet(args.model_dir, args.dropout).to(device)\n",
    """    model = MILNet(args.model_dir, args.dropout)
    if args.init_from:
        sd = torch.load(args.init_from, map_location="cpu")
        bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        assert bb, "no backbone.* keys in --init-from checkpoint"
        model.backbone.load_state_dict(bb)
        print(f"backbone initialised from {args.init_from} ({len(bb)} tensors)", flush=True)
    model = model.to(device)
""")
rep("l, a = run_epoch(model, tr_l, device, amp, opt, None, class_w, True)",
    "l, a = run_epoch(model, tr_l, device, amp, opt, None, class_w, True, args.accum)")
rep("l, a = run_epoch(model, tr_l, device, amp, opt, sched, class_w, False)",
    "l, a = run_epoch(model, tr_l, device, amp, opt, sched, class_w, False, args.accum)")
rep("    steps_ep = max(1, len(tr_l))\n", "    steps_ep = max(1, math.ceil(len(tr_l) / args.accum))\n")
ast.parse(s)
open(f, "w").write(s)
print("patched OK")

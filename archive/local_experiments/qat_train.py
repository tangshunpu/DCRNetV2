"""Per-bit Quantization-Aware Fine-Tune for v12_r (DCRNetV2-unified).

- Load existing v12_r checkpoint (pretrained, no overwrite)
- Fine-tune 50 epochs with FIXED bit-width QAT (one bit per job, no random sampling)
- STE for uniform quantizer
- Save with `v12r_qa{B}b` tag (e.g. v12r_qa6b, v12r_qa5b, ...)
"""
import argparse, os, random, math, sys, time
import torch
import torch.nn as nn
import numpy as np
sys.path.insert(0, '/home/ubuntu/Documents/project/DCRNet-V2')
from models import dcrnet_v9
# main.py does parse_args() at import; shield it from our argv
_saved_argv = sys.argv
sys.argv = [_saved_argv[0]]
from main import WarmUpCosineAnnealingLR, AverageMeter, evaluator
sys.argv = _saved_argv


def build_v12r(cr):
    return dcrnet_v9(
        reduction=cr, r_enc=512, refine_width=2, refine_K=1,
        use_film=False, complex_encoder=True, enc_fuse_mlp=64,
        hybrid_decoder=True, hybrid_direct_ranks=32, hybrid_extra_ranks=16,
        hybrid_extra_d_emb=16, hybrid_gate_bias=-2.0, hybrid_alpha_init=1e-2,
    )


def uniform_q_ste(z, bits):
    """Uniform quantize with straight-through estimator. Fixed bit-width."""
    if bits >= 32:
        return z
    levels = 2 ** bits - 1
    zmin = z.min(dim=1, keepdim=True).values.detach()
    zmax = z.max(dim=1, keepdim=True).values.detach()
    scale = (zmax - zmin).clamp(min=1e-12)
    z_norm = (z - zmin) / scale
    z_q_norm = z_norm + (torch.round(z_norm * levels) / levels - z_norm).detach()
    return z_q_norm * scale + zmin


def qat_forward(model, x, bits):
    z = model.encode(x)
    z_q = uniform_q_ste(z, bits)
    return model.decode(z_q)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--scenario", required=True, choices=["in", "out"])
    p.add_argument("--cr", type=int, required=True, choices=[4, 8, 16, 32])
    p.add_argument("--bits", type=int, required=True,
                   help="Single bit-width to train (e.g. 6, 5, 4, 3, 2)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val-freq", type=int, default=5)
    p.add_argument("--data", default="./COST2100")
    p.add_argument("--outputs", default="./outputs")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    B = args.bits
    print(f"QAT FIXED bits={B}")

    model = build_v12r(args.cr).to(device)
    pretrained_path = f"{args.outputs}/checkpoints/v9-1X-cr{args.cr}-{args.scenario}-v12r"
    assert os.path.isfile(pretrained_path), f"Need pretrained v12_r at {pretrained_path}"
    sd = torch.load(pretrained_path, map_location=device, weights_only=False)['state_dict']
    sd = {k: v for k, v in sd.items() if 'total_' not in k}
    model.load_state_dict(sd, strict=False)
    print(f"Loaded pretrained {pretrained_path}")

    from dataset import Cost2100DataLoader
    train_loader, val_loader, test_loader = Cost2100DataLoader(
        args.data, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, scenario=args.scenario,
    )()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_steps = len(train_loader) * args.epochs
    scheduler = WarmUpCosineAnnealingLR(
        optimizer, T_max=num_steps,
        T_warmup=5 * len(train_loader),
        eta_min=5e-5,
    )

    log_dir = f"{args.outputs}/log"
    os.makedirs(log_dir, exist_ok=True)
    tstamp = time.strftime("%m%d%H%M", time.localtime())
    log_path = f"{log_dir}/{tstamp}-v9-1X-{args.cr}-{args.scenario}-v12r_qa{B}b.log"
    logf = open(log_path, "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    log(f"QAT-per-bit: scenario={args.scenario} cr={args.cr} bits={B} "
        f"epochs={args.epochs} lr={args.lr}")
    log(f"Pretrained: {pretrained_path}")

    @torch.no_grad()
    def test_with_bits(BB):
        model.eval()
        iter_nmse = AverageMeter('nmse')
        for batch in test_loader:
            sparse_gt = batch[0]
            raw_gt = batch[1]
            y = qat_forward(model, sparse_gt, BB)
            rho, nmse = evaluator(y, sparse_gt, raw_gt)
            iter_nmse.update(float(nmse))
        return iter_nmse.avg

    log(f"\n=== Pretrained baseline ===")
    nmse_float0 = test_with_bits(32)
    nmse_q0 = test_with_bits(B)
    log(f"  float = {nmse_float0:+.3f} dB    {B}-bit (no QAT) = {nmse_q0:+.3f} dB")

    ckpt_path = f"{args.outputs}/checkpoints/v9-1X-cr{args.cr}-{args.scenario}-v12r_qa{B}b"
    best_nmse_q = 1e9
    best_epoch = -1
    for epoch in range(args.epochs):
        model.train()
        iter_loss = AverageMeter('loss')
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            y = qat_forward(model, x, B)  # FIXED bit
            loss = criterion(y, x)
            loss.backward()
            optimizer.step()
            scheduler.step()
            iter_loss.update(float(loss))

        if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1:
            nmse_q = test_with_bits(B)
            nmse_f = test_with_bits(32)
            improved = nmse_q < best_nmse_q
            tag = " NEW BEST" if improved else ""
            log(f"ep{epoch+1:>3}/{args.epochs} loss={iter_loss.avg:.3e}  "
                f"float={nmse_f:+.2f}  {B}b={nmse_q:+.2f}{tag}")
            if improved:
                best_nmse_q = nmse_q
                best_epoch = epoch
                torch.save({"state_dict": model.state_dict(),
                            "epoch": epoch, "bits": B,
                            "nmse_q": nmse_q, "nmse_float": nmse_f},
                           ckpt_path)

    log(f"\nDONE bits={B}. Best {B}-bit NMSE = {best_nmse_q:.3f} dB @ ep{best_epoch+1}")
    log(f"  vs pretrained {B}-bit (no QAT) = {nmse_q0:+.3f} dB  -->  gain = {nmse_q0 - best_nmse_q:+.3f} dB")
    log(f"Saved {ckpt_path}")
    logf.close()


if __name__ == "__main__":
    main()

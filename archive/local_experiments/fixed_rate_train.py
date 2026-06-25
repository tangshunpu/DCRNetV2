"""Fixed-rate training for v12_r with entropy bottleneck.

We pin γ per-dim such that quantised y = γ·z has per-dim std ≈ σ_target,
giving discrete entropy ≈ target_bpd bits/dim under a Gaussian prior:
    H_bits ≈ 0.5·log2(2πe) + log2(σ_target)  =>  σ_target = 2^(target_bpd - 2.05).

During training:
  - γ FROZEN (rate operating point locked)
  - μ, σ of factorized Gaussian prior: trainable (fits the prior to actual y distribution)
  - v12_r weights: trainable but with LOW lr to avoid float-NMSE collapse
  - Loss = MSE (no rate term — rate is pinned by γ)

Save as v12r_fbpd{N}b. Pretrained v12_r is NOT overwritten.
"""
import argparse, os, math, sys, time, random
import torch, torch.nn as nn
import numpy as np
sys.path.insert(0, '/home/ubuntu/Documents/project/DCRNet-V2')
from models import dcrnet_v9
_saved_argv = sys.argv; sys.argv = [_saved_argv[0]]
from main import WarmUpCosineAnnealingLR, AverageMeter, evaluator
sys.argv = _saved_argv


def build_v12r(cr):
    return dcrnet_v9(
        reduction=cr, r_enc=512, refine_width=2, refine_K=1,
        use_film=False, complex_encoder=True, enc_fuse_mlp=64,
        hybrid_decoder=True, hybrid_direct_ranks=32, hybrid_extra_ranks=16,
        hybrid_extra_d_emb=16, hybrid_gate_bias=-2.0, hybrid_alpha_init=1e-2,
    )


class FixedRateBottleneck(nn.Module):
    """y = γ·z with γ FROZEN; prior N(μ, σ²) per-dim is trainable."""

    def __init__(self, code_dim):
        super().__init__()
        # γ as buffer (not parameter) so it does NOT get gradients
        self.register_buffer("gamma", torch.ones(code_dim))
        self.mu = nn.Parameter(torch.zeros(code_dim))
        self.log_sigma = nn.Parameter(torch.zeros(code_dim))

    def _sigma(self):
        return self.log_sigma.clamp(min=-2.3, max=6.0).exp()

    def encode_train(self, z):
        y = z * self.gamma
        noise = torch.empty_like(y).uniform_(-0.5, 0.5)
        y_noisy = y + noise
        return y_noisy / self.gamma, y_noisy

    @torch.no_grad()
    def encode_eval(self, z):
        y = z * self.gamma
        y_int = torch.round(y)
        return y_int / self.gamma, y_int

    def bits_train(self, y_noisy):
        sigma = self._sigma()
        log_p = -0.5 * ((y_noisy - self.mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
        return -log_p / math.log(2.0)

    @torch.no_grad()
    def bits_eval(self, y_int):
        sigma = self._sigma()
        upper = (y_int + 0.5 - self.mu) / sigma
        lower = (y_int - 0.5 - self.mu) / sigma
        p = 0.5 * (torch.erf(upper / math.sqrt(2)) - torch.erf(lower / math.sqrt(2)))
        return -torch.log2(p.clamp(min=1e-12))


@torch.no_grad()
def calibrate(bn, model, loader, device, n_batches, target_y_std):
    """Set γ_d = target_y_std / std(z_d). Init prior to match data."""
    zs = []
    model.eval()
    for i, batch in enumerate(loader):
        if i >= n_batches: break
        x = batch[0]
        zs.append(model.encode(x))
    z_cat = torch.cat(zs, dim=0)
    z_std = z_cat.std(dim=0).clamp(min=1e-3)
    z_mean = z_cat.mean(dim=0)
    gamma = (target_y_std / z_std).clamp(min=0.5, max=400.)
    bn.gamma.data.copy_(gamma)
    bn.mu.data.copy_(gamma * z_mean)
    bn.log_sigma.data.copy_(torch.full_like(gamma, math.log(target_y_std)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--scenario", required=True, choices=["in", "out"])
    p.add_argument("--cr", type=int, required=True, choices=[4, 8, 16, 32])
    p.add_argument("--target-bpd", type=float, required=True,
                   help="Target discrete entropy per code dim (bits/dim).")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-5, help="model lr (kept LOW)")
    p.add_argument("--prior-lr", type=float, default=1e-3)
    p.add_argument("--rate-anchor", type=float, default=0.0,
                   help="Optional soft penalty λ·|bpd - target| to keep rate near target. "
                        "0 = off. 1e-3 = mild anchor.")
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val-freq", type=int, default=5)
    p.add_argument("--data", default="./COST2100")
    p.add_argument("--outputs", default="./outputs")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda")
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    target_y_std = 2.0 ** (args.target_bpd - 0.5 * math.log2(2 * math.pi * math.e))
    print(f"target_bpd={args.target_bpd} → target_y_std={target_y_std:.3f}")

    model = build_v12r(args.cr).to(device)
    pretrained = f"{args.outputs}/checkpoints/v9-1X-cr{args.cr}-{args.scenario}-v12r"
    sd = torch.load(pretrained, map_location=device, weights_only=False)['state_dict']
    sd = {k: v for k, v in sd.items() if 'total_' not in k}
    model.load_state_dict(sd, strict=False)
    print(f"Loaded {pretrained}")

    with torch.no_grad():
        probe = torch.zeros(1, 2, 32, 32, device=device)
        z_probe = model.encode(probe)
    code_dim = z_probe.shape[-1] if z_probe.ndim == 2 else int(np.prod(z_probe.shape[1:]))
    print(f"code_dim = {code_dim}")

    bn = FixedRateBottleneck(code_dim).to(device)

    from dataset import Cost2100DataLoader
    train_loader, val_loader, test_loader = Cost2100DataLoader(
        args.data, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, scenario=args.scenario,
    )()

    calibrate(bn, model, train_loader, device, n_batches=10, target_y_std=target_y_std)
    print(f"Init: γ_range=[{bn.gamma.min().item():.2f}, {bn.gamma.max().item():.2f}]  σ_y_target={target_y_std:.3f}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": args.lr},
        {"params": [bn.mu, bn.log_sigma], "lr": args.prior_lr},
    ])
    num_steps = len(train_loader) * args.epochs
    warmup = min(5 * len(train_loader), max(1, num_steps // 4))
    scheduler = WarmUpCosineAnnealingLR(
        optimizer, T_max=num_steps, T_warmup=warmup, eta_min=5e-6,
    )

    log_dir = f"{args.outputs}/log"; os.makedirs(log_dir, exist_ok=True)
    tstamp = time.strftime("%m%d%H%M", time.localtime())
    bpd_tag = f"{args.target_bpd:.1f}b".replace('.', 'p')
    log_path = f"{log_dir}/{tstamp}-v9-1X-{args.cr}-{args.scenario}-v12r_fbpd{bpd_tag}.log"
    logf = open(log_path, "w")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    log(f"Fixed-rate: scenario={args.scenario} cr={args.cr} target_bpd={args.target_bpd} σ_y_target={target_y_std:.3f}")
    log(f"epochs={args.epochs} model_lr={args.lr} prior_lr={args.prior_lr} rate_anchor={args.rate_anchor}")

    @torch.no_grad()
    def eval_rd():
        model.eval(); bn.eval()
        sum_nmse, sum_bits, sum_rho, count = 0., 0., 0., 0
        for batch in test_loader:
            sparse_gt, raw_gt = batch[0], batch[1]
            z = model.encode(sparse_gt)
            z_hat, y_int = bn.encode_eval(z)
            bits_pp = bn.bits_eval(y_int).sum(dim=-1)
            y = model.decode(z_hat)
            rho, nmse = evaluator(y, sparse_gt, raw_gt)
            n = z.shape[0]
            sum_nmse += float(nmse) * n
            sum_rho += float(rho) * n
            sum_bits += float(bits_pp.mean()) * n
            count += n
        return sum_nmse / count, sum_bits / count, sum_rho / count

    nmse0, bits0, rho0 = eval_rd()
    bpd0 = bits0 / code_dim
    log(f"BEFORE training: NMSE={nmse0:+.3f} dB  bits={bits0:.1f} ({bpd0:.2f} b/dim)  ρ={rho0:.4f}")

    ckpt_path = f"{args.outputs}/checkpoints/v9-1X-cr{args.cr}-{args.scenario}-v12r_fbpd{bpd_tag}"
    best_nmse = 1e9
    best_ep = -1
    for epoch in range(args.epochs):
        model.train(); bn.train()
        iter_mse = AverageMeter('mse'); iter_bits = AverageMeter('bits')
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            z = model.encode(x)
            z_hat, y_noisy = bn.encode_train(z)
            y_recon = model.decode(z_hat)
            mse = criterion(y_recon, x)
            bits_train = bn.bits_train(y_noisy).sum(dim=-1).mean()
            loss = mse
            if args.rate_anchor > 0:
                bpd_train = bits_train / code_dim
                loss = loss + args.rate_anchor * (bpd_train - args.target_bpd) ** 2
            loss.backward()
            optimizer.step()
            scheduler.step()
            iter_mse.update(float(mse))
            iter_bits.update(float(bits_train))

        if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1:
            nmse, bits_eval, rho = eval_rd()
            bpd_eval = bits_eval / code_dim
            improved = nmse < best_nmse
            tag = " NEW BEST" if improved else ""
            log(f"ep{epoch + 1:>3}/{args.epochs} mse={iter_mse.avg:.3e} bits_train={iter_bits.avg:.1f} "
                f"({iter_bits.avg/code_dim:.2f} b/dim) | "
                f"eval: NMSE={nmse:+.2f} bits={bits_eval:.1f} ({bpd_eval:.2f} b/dim) ρ={rho:.4f}{tag}")
            if improved:
                best_nmse = nmse; best_ep = epoch
                torch.save({"state_dict": model.state_dict(),
                            "bottleneck_state_dict": bn.state_dict(),
                            "epoch": epoch, "target_bpd": args.target_bpd,
                            "nmse": nmse, "bits": bits_eval, "bpd": bpd_eval,
                            "rho": rho}, ckpt_path)

    log(f"\nDONE target_bpd={args.target_bpd}. Best NMSE = {best_nmse:.3f} dB @ ep{best_ep + 1}")
    log(f"Saved {ckpt_path}")
    logf.close()


if __name__ == "__main__":
    main()

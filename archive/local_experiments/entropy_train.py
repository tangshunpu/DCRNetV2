"""Entropy-aware fine-tune for v12_r with factorized Gaussian prior on z.

z has small natural scale, so we introduce a per-dim learnable γ (scale):
    y = γ * z              (push z into quantization range)
    y_noisy = y + U(-0.5, 0.5)    (training-time relaxation)
    y_int   = round(y)            (eval-time discrete)
    z_hat   = y / γ        (inverse scale before decode)

Rate-distortion loss:
    L = MSE(decode(z_hat_noisy), x) + λ · Σ_d -log2 p_d(y_noisy)
p_d(y) = N(y; μ_d, σ_d²),  μ_d/σ_d/γ_d all trainable.

Eval: bits_d = -log2 ∫_{y_int-0.5}^{y_int+0.5} N(t; μ_d, σ_d²) dt (Gaussian CDF).
Save as v12r_ent_<lam_tag>. Pretrained v12_r is NOT overwritten.
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


class FactorizedGaussianBottleneck(nn.Module):
    """y = γ · z, then integer-quantize y; prior p(y) is Gaussian per-dim."""

    def __init__(self, code_dim):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(code_dim))
        self.log_sigma = nn.Parameter(torch.zeros(code_dim))    # σ=1 init
        self.log_gamma = nn.Parameter(torch.zeros(code_dim))    # γ=1 init

    def _gamma(self):
        return self.log_gamma.clamp(min=-1.0, max=6.0).exp()    # γ ∈ [0.37, 400]

    def _sigma(self):
        return self.log_sigma.clamp(min=-2.3, max=4.0).exp()    # σ ∈ [0.1, ~55]

    def encode_train(self, z):
        """z (B,D) → (z_hat, y_noisy). z_hat passes to decoder; y_noisy passes to bits_train."""
        gamma = self._gamma()
        y = z * gamma
        noise = torch.empty_like(y).uniform_(-0.5, 0.5)
        y_noisy = y + noise
        z_hat = y_noisy / gamma
        return z_hat, y_noisy

    @torch.no_grad()
    def encode_eval(self, z):
        """z → (z_hat, y_int). Discrete integer quantization."""
        gamma = self._gamma()
        y = z * gamma
        y_int = torch.round(y)
        z_hat = y_int / gamma
        return z_hat, y_int

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
def calibrate(bottleneck, model, loader, device, n_batches=10, target_y_std=2.0):
    """Init γ so y=γ·z has std ≈ target_y_std per dim; init μ/σ to match y stats."""
    zs = []
    model.eval()
    for i, batch in enumerate(loader):
        if i >= n_batches: break
        x = batch[0]
        zs.append(model.encode(x))
    z_cat = torch.cat(zs, dim=0)
    z_std = z_cat.std(dim=0).clamp(min=1e-3)
    z_mean = z_cat.mean(dim=0)
    gamma = (target_y_std / z_std).clamp(min=0.5, max=200.)
    bottleneck.log_gamma.data.copy_(gamma.log())
    # In y-space: y = γ z → mean(y) = γ * mean(z), std(y) ≈ target_y_std
    bottleneck.mu.data.copy_(gamma * z_mean)
    bottleneck.log_sigma.data.copy_(torch.full_like(gamma, math.log(target_y_std)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--scenario", required=True, choices=["in", "out"])
    p.add_argument("--cr", type=int, required=True, choices=[4, 8, 16, 32])
    p.add_argument("--lam", type=float, required=True, help="rate-distortion λ (mse + λ*bits)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--prior-lr", type=float, default=1e-3)
    p.add_argument("--target-y-std", type=float, default=2.0)
    p.add_argument("--freeze-model", action="store_true",
                   help="Only train the entropy bottleneck (γ,μ,σ); keep v12_r weights frozen.")
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

    bn = FactorizedGaussianBottleneck(code_dim).to(device)

    from dataset import Cost2100DataLoader
    train_loader, val_loader, test_loader = Cost2100DataLoader(
        args.data, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, scenario=args.scenario,
    )()

    calibrate(bn, model, train_loader, device, n_batches=10, target_y_std=args.target_y_std)
    print(f"Init: γ_range=[{bn._gamma().min().item():.2f}, {bn._gamma().max().item():.2f}]  "
          f"μ_range=[{bn.mu.min().item():+.2f}, {bn.mu.max().item():+.2f}]  "
          f"σ={bn._sigma().mean().item():.2f}")

    criterion = nn.MSELoss()
    if args.freeze_model:
        for p_ in model.parameters():
            p_.requires_grad = False
        optimizer = torch.optim.Adam(bn.parameters(), lr=args.prior_lr)
        print("Model frozen — training only entropy bottleneck (γ, μ, σ)")
    else:
        optimizer = torch.optim.Adam([
            {"params": model.parameters(), "lr": args.lr},
            {"params": bn.parameters(), "lr": args.prior_lr},
        ])
    num_steps = len(train_loader) * args.epochs
    warmup = min(5 * len(train_loader), max(1, num_steps // 2))
    scheduler = WarmUpCosineAnnealingLR(
        optimizer, T_max=num_steps, T_warmup=warmup, eta_min=5e-5,
    )

    log_dir = f"{args.outputs}/log"; os.makedirs(log_dir, exist_ok=True)
    tstamp = time.strftime("%m%d%H%M", time.localtime())
    lam_tag = f"{args.lam:.0e}".replace('e-0', 'em').replace('e-', 'em').replace('e+0', 'ep').replace('e+', 'ep')
    log_path = f"{log_dir}/{tstamp}-v9-1X-{args.cr}-{args.scenario}-v12r_ent_{lam_tag}.log"
    logf = open(log_path, "w")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    log(f"Entropy training: scenario={args.scenario} cr={args.cr} λ={args.lam} epochs={args.epochs} "
        f"lr={args.lr} prior_lr={args.prior_lr} y_std={args.target_y_std}")
    log(f"code_dim={code_dim}")

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
    log(f"BEFORE training: NMSE={nmse0:+.3f} dB  bits={bits0:.1f}  ρ={rho0:.4f}")

    ckpt_path = f"{args.outputs}/checkpoints/v9-1X-cr{args.cr}-{args.scenario}-v12r_ent_{lam_tag}"
    best_rd = 1e9
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
            bits = bn.bits_train(y_noisy).sum(dim=-1).mean()
            loss = mse + args.lam * bits
            loss.backward()
            optimizer.step()
            scheduler.step()
            iter_mse.update(float(mse))
            iter_bits.update(float(bits))

        if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1:
            nmse, bits_eval, rho = eval_rd()
            rd_cost = nmse + args.lam * bits_eval
            improved = rd_cost < best_rd
            tag = " NEW BEST" if improved else ""
            log(f"ep{epoch + 1:>3}/{args.epochs} mse={iter_mse.avg:.3e} bits_train={iter_bits.avg:.1f} | "
                f"eval: NMSE={nmse:+.2f} bits={bits_eval:.1f} ρ={rho:.4f}{tag}")
            if improved:
                best_rd = rd_cost; best_ep = epoch
                torch.save({"state_dict": model.state_dict(),
                            "bottleneck_state_dict": bn.state_dict(),
                            "epoch": epoch, "lam": args.lam,
                            "nmse": nmse, "bits": bits_eval, "rho": rho}, ckpt_path)

    log(f"\nDONE λ={args.lam}. Best @ ep{best_ep + 1}")
    log(f"Saved {ckpt_path}")
    logf.close()


if __name__ == "__main__":
    main()

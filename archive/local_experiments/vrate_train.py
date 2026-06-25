"""Sequential λ schedule entropy training (CompressAI-style curriculum).

Stages: easy → hard.  Easy = small λ (rate penalty weak, bits high, NMSE near float).
Hard = large λ (rate penalty strong, bits low, NMSE worse).

Within each stage:
  - Fixed λ
  - Loss = MSE + λ · bits
  - 20 epochs

Across stages:
  - Model + bottleneck weights warm-started from previous stage's final state
  - Each stage saves a checkpoint, giving K R-D operating points per (cr, scenario)

Naming: v12r_vrate_s<stage_idx>_l<lam_tag>
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


class EntropyBottleneck(nn.Module):
    """Per-dim factorized Gaussian prior with trainable per-dim γ scale."""

    def __init__(self, code_dim):
        super().__init__()
        self.log_gamma = nn.Parameter(torch.zeros(code_dim))
        self.mu = nn.Parameter(torch.zeros(code_dim))
        self.log_sigma = nn.Parameter(torch.zeros(code_dim))

    def _gamma(self):
        return self.log_gamma.clamp(min=-2.0, max=7.0).exp()

    def _sigma(self):
        return self.log_sigma.clamp(min=-2.3, max=7.0).exp()

    def encode_train(self, z):
        gamma = self._gamma()
        y = z * gamma
        noise = torch.empty_like(y).uniform_(-0.5, 0.5)
        y_noisy = y + noise
        return y_noisy / gamma, y_noisy

    @torch.no_grad()
    def encode_eval(self, z):
        gamma = self._gamma()
        y = z * gamma
        y_int = torch.round(y)
        return y_int / gamma, y_int

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
    bn.log_gamma.data.copy_(gamma.log())
    bn.mu.data.copy_(gamma * z_mean)
    bn.log_sigma.data.copy_(torch.full_like(gamma, math.log(target_y_std)))


def lam_tag(lam):
    s = f"{lam:.0e}"
    return s.replace('e-0', 'em').replace('e-', 'em').replace('e+0', 'ep').replace('e+', 'ep')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--scenario", required=True, choices=["in", "out"])
    p.add_argument("--cr", type=int, required=True, choices=[4, 8, 16, 32])
    p.add_argument("--stages", type=str, default="1e-10,1e-9,1e-8,1e-7,1e-6",
                   help="comma-separated λ values, easy → hard")
    p.add_argument("--epochs-per-stage", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--prior-lr", type=float, default=1e-3)
    p.add_argument("--init-y-std", type=float, default=10.0)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data", default="./COST2100")
    p.add_argument("--outputs", default="./outputs")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda")
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    lambdas = [float(x) for x in args.stages.split(",")]
    print(f"λ schedule (easy → hard): {lambdas}")

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

    bn = EntropyBottleneck(code_dim).to(device)

    from dataset import Cost2100DataLoader
    train_loader, val_loader, test_loader = Cost2100DataLoader(
        args.data, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, scenario=args.scenario,
    )()

    calibrate(bn, model, train_loader, device, n_batches=10, target_y_std=args.init_y_std)
    print(f"Init γ ∈ [{bn._gamma().min().item():.2f}, {bn._gamma().max().item():.2f}]  "
          f"σ_y_init={args.init_y_std:.2f}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": args.lr},
        {"params": bn.parameters(), "lr": args.prior_lr},
    ])
    total_steps = len(lambdas) * args.epochs_per_stage * len(train_loader)
    warmup = min(3 * len(train_loader), max(1, total_steps // 8))
    scheduler = WarmUpCosineAnnealingLR(
        optimizer, T_max=total_steps, T_warmup=warmup, eta_min=5e-6,
    )

    log_dir = f"{args.outputs}/log"; os.makedirs(log_dir, exist_ok=True)
    tstamp = time.strftime("%m%d%H%M", time.localtime())
    log_path = f"{log_dir}/{tstamp}-v9-1X-{args.cr}-{args.scenario}-v12r_vrate.log"
    logf = open(log_path, "w")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    log(f"V-rate sequential training: scenario={args.scenario} cr={args.cr} "
        f"stages={lambdas} ep_per_stage={args.epochs_per_stage} "
        f"model_lr={args.lr} prior_lr={args.prior_lr} init_y_std={args.init_y_std}")
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
    log(f"BEFORE training: NMSE={nmse0:+.3f}  bits={bits0:.1f} ({bits0/code_dim:.2f} b/dim)  ρ={rho0:.4f}")

    rd_log = []  # (stage, λ, nmse, bits, bpd, rho)
    for stage_idx, lam in enumerate(lambdas):
        log(f"\n=== Stage {stage_idx}/{len(lambdas)-1} : λ={lam:.0e} ===")
        for epoch in range(args.epochs_per_stage):
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
                loss = mse + lam * bits
                loss.backward()
                optimizer.step()
                scheduler.step()
                iter_mse.update(float(mse))
                iter_bits.update(float(bits))
            global_ep = stage_idx * args.epochs_per_stage + epoch + 1
            log(f"  s{stage_idx} ep{epoch+1:>2}/{args.epochs_per_stage} (global {global_ep}) "
                f"mse={iter_mse.avg:.3e}  bits_train={iter_bits.avg:.1f} "
                f"({iter_bits.avg/code_dim:.2f} b/dim)")

        # End-of-stage eval + save
        nmse, bits_eval, rho = eval_rd()
        bpd = bits_eval / code_dim
        log(f"  ⇒ stage {stage_idx} done: NMSE={nmse:+.3f} bits={bits_eval:.1f} ({bpd:.2f} b/dim) ρ={rho:.4f}")
        rd_log.append((stage_idx, lam, nmse, bits_eval, bpd, rho))

        ckpt_path = (f"{args.outputs}/checkpoints/v9-1X-cr{args.cr}-{args.scenario}-"
                     f"v12r_vrate_s{stage_idx}_l{lam_tag(lam)}")
        torch.save({"state_dict": model.state_dict(),
                    "bottleneck_state_dict": bn.state_dict(),
                    "stage_idx": stage_idx, "lam": lam,
                    "nmse": nmse, "bits": bits_eval, "bpd": bpd, "rho": rho}, ckpt_path)
        log(f"  saved {ckpt_path}")

    log(f"\n=== R-D summary for cr={args.cr} {args.scenario} ===")
    log(f"  stage | λ       | NMSE     | bits   | b/dim | ρ")
    for s, lam, n, b, bp, r in rd_log:
        log(f"  s{s}    | {lam:.0e} | {n:+8.3f} | {b:6.1f} | {bp:5.2f} | {r:.4f}")
    logf.close()


if __name__ == "__main__":
    main()

import torch
from torch import nn
from dataset import (Cost2100DataLoader, RiceFDDDataLoader,
                     DeepMIMODataLoader, RTCSIDataLoader)
import argparse
import models as _models
import os
from tqdm import tqdm
import math
from torch.optim.lr_scheduler import _LRScheduler
import time
import thop
import sys
from colorama import Fore
import logging
import random
import numpy as np
import inspect
# arg
parser = argparse.ArgumentParser(description='PyTorch CSI feeback')

parser.add_argument('--data', default='./COST2100', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default='cost2100',
                    choices=['cost2100', 'rice', 'deepmimo', 'rtcsi'],
                    help='which dataset to train on. cost2100 = original '
                         '(2,32,32) angular-delay. rice = RENEW FDD indoor, '
                         'raw frequency-domain (2,64,52), no DFT/IFFT. '
                         'deepmimo = DeepMIMO O1_3p4 outdoor NLoS angular-delay '
                         '(2,32,32), 20k/10k/10k split. rtcsi = Sionna RT '
                         'angular-delay (2,32,32) from RT_CSI scenes.')
parser.add_argument('--rtcsi-scene', type=str, default='zju',
                    choices=['zju', 'munich', 'etoile', 'florence', 'canyon',
                             'sutd', 'shenzhen', 'zjubs0', 'combined5'],
                    help='rtcsi only: which scene npz to load')
parser.add_argument('--rice-channel', type=str, default='ch14',
                    choices=['ch1', 'ch14'],
                    help='rice only: ch1 = UL (channel 1), ch14 = DL (channel 14)')
parser.add_argument('--rice-domain', type=str, default='freq',
                    choices=['freq', 'angdelay'],
                    help='rice only: input representation. freq = raw '
                         'frequency-domain CSI (2,96,52). angdelay = '
                         'fft(ant)+ifft(sub), lossless, no crop. v5 needs '
                         'angdelay to make its low-rank prior work.')
parser.add_argument('--rice-split', type=str, default='location',
                    choices=['location', 'random'],
                    help='rice only: location = train/val/test on disjoint '
                         'sets of locations (zero-shot to new sites, hard). '
                         'random = pool all 24 locations and split frames '
                         '70/15/15 — the standard CSI-feedback evaluation.')
parser.add_argument('--rice-train-frac', type=float, default=0.70,
                    help='rice random-split only: train fraction. '
                         '0.70 → 70/15/15 (default). 0.80 → 80/20 (no separate '
                         'val; val mirrors test, matches CSINet/CRNet protocol).')
parser.add_argument('--scenario', type=str, default="in", choices=["in", "out"],
                    help="the channel scenario")
parser.add_argument('--cr', metavar='N', type=int, default=4,
                    help='compression ratio')
parser.add_argument('--outputs', default='./outputs',
                    help='folder to output model checkpoints')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=2500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default=None, type=str,
                    help='GPU id to use.')
parser.add_argument('-b', '--batch-size', default=200, type=int,
                    metavar='N', help='mini-batch size (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', dest='weight_decay',
                    default=0.0, type=float,
                    help='Adam weight decay (default 0). Use 1e-4 for v1 cr=4 to '
                         'counter overfit on small multi-BS RT_CSI scenes.')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on test set')
parser.add_argument('--expansion', default=1, type=int,
                   help='expansion rate of dcrnet')
parser.add_argument('--pretrained', type=str, default=None,
                    help='using locally pre-trained model. The path of pre-trained model should be given')
parser.add_argument('--init-from-v5', dest='init_from_v5', type=str, default=None,
                    help='path to a v5 checkpoint. Loads encoder->enc_lowrank and '
                         'decoder->decoder as a warm start. Use --freeze-v5 to also '
                         'freeze those modules.')
parser.add_argument('--freeze-v5', dest='freeze_v5', action='store_true',
                    help='with --init-from-v5, freeze the loaded enc_lowrank+decoder '
                         'so only the new modules (enc_dilate, fuse_enc, dec_refine) '
                         'are trained.')
parser.add_argument('--backbone-lr', dest='backbone_lr', type=float, default=None,
                    help='with --init-from-v5 (no freeze), use this lr for the loaded '
                         'enc_lowrank+decoder while keeping --lr for the new modules. '
                         'Use a small value (e.g. 5e-5) to protect the v5 warm start.')
parser.add_argument('--val-freq', '-v', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--model', type=str, default='v8',
                    choices=['v1', 'v5', 'v8', 'v9', 'v13', 'v14', 'v20', 'v21a', 'v21b', 'v22', 'v23', 'v24', 'v25', 'v25b', 'v25c', 'v25d', 'v26', 'v27', 'v28', 'v28b', 'v29', 'v29ng', 'v29svd', 'v29svd2', 'v29out', 'v29h', 'v29t', 'v29p', 'v29m', 'v29mx', 'v30', 'v30a',
                             'crnet', 'clnet', 'transnet', 'tclnet'],
                    help='which model to use. v1=DCRNet paper; v5/v8/v9/v13/v14='
                         'our DCRNetV2 variants; crnet/clnet/transnet='
                         'external baselines (factory of the same name).')
parser.add_argument('--ranks', type=int, default=None,
                    help='decoder rank R (v3/v4/v5/v6/v7); None = factory default')
parser.add_argument('--r-enc', dest='r_enc', type=int, default=None,
                    help='encoder matched-filter count R_enc (v5/v7); None = factory default')
parser.add_argument('--d-emb', dest='d_emb', type=int, default=None,
                    help='v9 rank-embedding dim; None = factory default')
parser.add_argument('--refine-width', dest='refine_width', type=int, default=None,
                    help='v9 refine block width (channels in DCRDecoderRefine); '
                         'None = factory default')
parser.add_argument('--refine-k', dest='refine_K', type=int, default=None,
                    help='v9 number of stacked DCRDecoderRefine stages; None = factory default')
parser.add_argument('--no-dilate-path', dest='no_dilate_path', action='store_true',
                    help='v9 only: disable the dilate-conv encoder path (pure neural low-rank)')
parser.add_argument('--coarse-loss-weight', dest='coarse_loss_weight', type=float, default=0.0,
                    help='v9 only: λ for auxiliary loss on the coarse rank-R reconstruction. '
                         'Total loss = MSE(refined, gt) + λ·MSE(coarse, gt). λ=0 disables (default).')
parser.add_argument('--coarse-anneal', dest='coarse_anneal', action='store_true',
                    help='v9 only: anneal coarse-loss-weight — first 40%% of epochs use peak λ '
                         '(coarse-loss-weight), then linear decay to 0 over next 40%%, last 20%% = 0.')
parser.add_argument('--no-train-clamp', dest='no_train_clamp', action='store_true',
                    help='v9 only: skip clamp(0,1) during training forward; still clamp at eval.')
parser.add_argument('--rank-attn-blocks', dest='rank_attn_blocks', type=int, default=0,
                    help='v9 only: number of RankTokenAttention transformer blocks on the R '
                         'bilinear measurements. 0 disables (default).')
parser.add_argument('--rank-attn-dim', dest='rank_attn_dim', type=int, default=32,
                    help='v9 only: d_model for RankTokenAttention. (default 32)')
parser.add_argument('--rank-attn-heads', dest='rank_attn_heads', type=int, default=4,
                    help='v9 only: number of heads in RankTokenAttention MHA. (default 4)')
parser.add_argument('--no-film', dest='no_film', action='store_true',
                    help='v9 only: disable FiLM-modulated NeuralLowRankEncoder, use v5/v8\'s '
                         'static LowRankEncoder (direct u, v parameters). Strictly more '
                         'expressive at same R, no rank-bottleneck through D-dim embeddings.')
parser.add_argument('--neural-decoder', dest='neural_decoder', action='store_true',
                    help='v9 only: replace LowRankDecoder (direct Linear(code_dim, R·128)) with '
                         'NeuralLowRankDecoder (FiLM-modulated rank embeddings → factor heads). '
                         'Symmetric to NeuralLowRankEncoder; much cheaper at high R.')
parser.add_argument('--dec-d-emb', dest='dec_d_emb', type=int, default=32,
                    help='v9 only: d_emb for NeuralLowRankDecoder rank embedding (default 32).')
parser.add_argument('--hybrid-decoder', dest='hybrid_decoder', action='store_true',
                    help='v9 only: use HybridRankDecoder = direct LowRankDecoder(R_d) + '
                         'GatedFactoredResidualDecoder(R_e). Cost-neutral capacity boost vs '
                         'plain LowRankDecoder(R=R_d+R_e).')
parser.add_argument('--hybrid-direct-r', dest='hybrid_direct_ranks', type=int, default=16,
                    help='v9 hybrid: direct LowRankDecoder ranks (default 16).')
parser.add_argument('--hybrid-extra-r', dest='hybrid_extra_ranks', type=int, default=64,
                    help='v9 hybrid: GatedFactoredResidualDecoder ranks (default 64).')
parser.add_argument('--hybrid-extra-d', dest='hybrid_extra_d_emb', type=int, default=16,
                    help='v9 hybrid: d_emb in factored residual decoder (default 16).')
parser.add_argument('--hybrid-gate-bias', dest='hybrid_gate_bias', type=float, default=-2.0,
                    help='v9 hybrid: initial rank_gate bias; sigmoid(bias) is initial open fraction '
                         '(default -2.0 → ~0.12).')
parser.add_argument('--hybrid-alpha-init', dest='hybrid_alpha_init', type=float, default=1e-2,
                    help='v9 hybrid: initial global residual scale alpha (default 1e-2).')
parser.add_argument('--enc-fuse-1x1', dest='enc_fuse_1x1', action='store_true',
                    help='v9 only: fuse encoder branches (LowRank + Dilate) via 1x1 conv MLP '
                         'instead of plain sum.')
parser.add_argument('--enc-fuse-mlp', dest='enc_fuse_mlp', type=int, default=0,
                    help='v9 only: fuse encoder branches via concat + 2-layer MLP (hidden=N). '
                         '0 disables. Output = (z_a+z_b) + MLP(concat(z_a, z_b)) — residual fusion.')
parser.add_argument('--enc-fuse-linear', dest='enc_fuse_linear', action='store_true',
                    help='v9 only: fuse encoder branches via single Linear(2·code_dim → code_dim), '
                         'identity-init so output equals z_a + z_b at step 0.')
parser.add_argument('--complex-encoder', dest='complex_encoder', action='store_true',
                    help='v9 only: use true complex Hermitian bilinear encoder '
                         '(u_re, u_im, v_re, v_im as parameters; 4 cross-terms preserved).')
parser.add_argument('--dec-dilate', dest='dec_dilate', action='store_true',
                    help='v9 only: add a parallel DilateDecoderPath (z → Linear → DCR block) '
                         'next to the low-rank decoder. Output fused via 1x1 conv (4→2).')
parser.add_argument('--dec-dilate-width', dest='dec_dilate_width', type=int, default=8,
                    help='v9 only: DCRDecoderBlock width inside DilateDecoderPath (default 8).')
parser.add_argument('--seed', dest='seed', type=int, default=None,
                    help='Random seed for reproducibility. None = no seeding.')
parser.add_argument('--warm-restart', dest='warm_restart', action='store_true',
                    help='With --resume: load weights only, reset optimizer + '
                         'scheduler state (cosine warm restart at args.lr).')
parser.add_argument('--scheduler', type=str, default='warmup_cosine',
                    choices=['warmup_cosine', 'const'],
                    help='warmup_cosine = WarmUpCosineAnnealingLR (default). '
                         'const = constant lr at args.lr (matches TransNet paper).')
parser.add_argument('--tag', dest='tag', type=str, default='',
                    help='extra tag appended to BOTH log file and checkpoint filename, '
                         'e.g. "tinyO" → log=...-{cr}-{scenario}-tinyO.log, '
                         'ckpt=...-cr{cr}-{scenario}-tinyO. Required to avoid collision '
                         'when multiple configs run at the same cr.')
args = parser.parse_args()

# Resolve model factory by --model.
#   v1                       -> dcrnet
#   v5/v8/v9/v13/v14         -> dcrnet_v{N}
#   crnet/clnet/transnet     -> {name} (external baselines)
if args.model == 'v1':
    _factory_name = 'dcrnet'
elif args.model in ('crnet', 'clnet', 'transnet', 'tclnet'):
    _factory_name = args.model
else:
    _factory_name = f'dcrnet_{args.model}'
dcrnet = getattr(_models, _factory_name)
if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)





def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # logger.addHandler(sh)

    return logger
class ConstLR(_LRScheduler):
    """No-op scheduler: lr stays at base_lr every step. Matches TransNet paper's
    FakeLR (Treedy2020/TransNet). Used when --scheduler const."""
    def get_lr(self):
        return list(self.base_lrs)


class WarmUpCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, T_warmup, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            return [base_lr * self.last_epoch / self.T_warmup for base_lr in self.base_lrs]
        else:
            k = 1 + math.cos(math.pi * (self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup))
            return [self.eta_min + (base_lr - self.eta_min) * k / 2 for base_lr in self.base_lrs]

def _einsum_macs(model, batch_size=1):
    """MACs for einsum-based ops in our custom modules.

    thop only hooks standard nn.Module subclasses (Conv2d, Linear, BN, ...).
    LowRankEncoder / LowRankDecoder / TiedLowRankCodec call torch.einsum
    directly, which thop sees as zero MACs. For v3/v5/v6/v8/v9 this is a
    sizable undercount (~40-75% of true total at high cr).
    """
    extra = 0
    for m in model.modules():
        cls = m.__class__.__name__
        if cls == 'LowRankEncoder':
            # encode: 'bcij,rj->bcir' (B·C·H·W·R) + 'ri,bcir->bcr' (B·C·H·R)
            R, H, W = m.r_enc, m.h, m.w
            C = m.mix.in_features // R
            extra += batch_size * C * H * W * R
            extra += batch_size * C * H * R
        elif cls == 'LowRankDecoder':
            # 4 outer-product einsums 'brh,brw->bhw' (B·R·H·W each)
            R, H, W = m.ranks, m.h, m.w
            extra += 4 * batch_size * R * H * W
        elif cls == 'GatedFactoredResidualDecoder':
            # code_proj + rank_gate are nn.Linear → already counted by thop.
            # Bare einsum: 2·R·d·H + 2·R·d·W for factor heads, 4·R·H·W for outer.
            R, H, W, D = m.extra_ranks, m.h, m.w, m.d_emb
            extra += 2 * batch_size * R * D * H
            extra += 2 * batch_size * R * D * W
            extra += 4 * batch_size * R * H * W
        elif cls == 'NeuralLowRankDecoder':
            # V2: code_proj is nn.Linear (caught by thop). The 4 factor-head
            # matmuls F_per_sample @ W and the 4 outer-product einsums are bare
            # tensor ops -- add here.
            R, H, W, D = m.ranks, m.h, m.w, m.d_emb
            extra += 2 * batch_size * R * D * H
            extra += 2 * batch_size * R * D * W
            extra += 4 * batch_size * R * H * W
        elif cls == 'TiedLowRankCodec':
            # encode: 4×'bij,rj->bri' (B·R·H·W each) + 4×'ri,bri->br' (B·R·H each)
            # decode: 4×'bri,rj->bij' (B·R·H·W each)
            R, H, W = m.r, m.h, m.w
            extra += 4 * batch_size * R * H * W
            extra += 4 * batch_size * R * H
            extra += 4 * batch_size * R * H * W
        elif cls == 'AxialAttention':
            # axis='w': sequences=H, seq_len=W. QK: H·W·W·D, AV: same.
            # axis='h': sequences=W, seq_len=H. QK: W·H·H·D, AV: same.
            # qkv proj and output proj are nn.Linear → already hooked by thop.
            D, H, W = m.dim, m.h, m.w
            if m.axis == 'w':
                extra += 2 * batch_size * D * H * W * W
            else:
                extra += 2 * batch_size * D * H * H * W
        elif cls == 'NeuralLowRankEncoder':
            # FiLM mod: γ·E (B·R·D MACs, β is just add — not counted)
            # E_mod @ W_U / W_V: 2 matmuls, each B·R·D·H/W MACs
            # bilinear einsums: bcij,brj->bcir (B·C·H·W·R) + bri,bcir->bcr (B·C·H·R)
            # self.mix is nn.Linear → already counted by thop.
            R, H, W, D = m.r_enc, m.h, m.w, m.d_emb
            C = m.in_channels
            extra += batch_size * R * D
            extra += batch_size * R * D * H
            extra += batch_size * R * D * W
            extra += batch_size * C * H * W * R
            extra += batch_size * C * H * R
        elif cls == 'ComplexLowRankEncoder' or cls == 'SVDComplexLowRankEncoder':
            # 4 einsums "bij,rj->bri" (B·R·H·W each) for {H_re/im} × {v_re/im}
            # + 4 einsums "ri,bri->br" (B·R·H each) for u terms
            # self.mix is nn.Linear → already by thop.
            # SVDComplexLowRankEncoder shares same einsum pattern; unit-norm is element-wise.
            R, H, W = m.r_enc, m.h, m.w
            extra += 4 * batch_size * R * H * W
            extra += 4 * batch_size * R * H
        elif cls == 'SVDLowRankDecoder' or cls == 'SVDLowRankDecoderSched':
            # Same einsum cost as LowRankDecoder: 4 outer-product einsums.
            R, H, W = m.ranks, m.h, m.w
            extra += 4 * batch_size * R * H * W
        elif cls == 'RankTokenAttention':
            # nn.MultiheadAttention's qkv/out_proj Linears are caught by thop.
            # The bare attention matmul (Q·K^T and AV) is not.
            # Per block: Q·K^T: B·R·R·D, A·V: B·R·R·D → 2·B·R²·D.
            R, D = m.r_enc, m.d_model
            n_blocks = len(m.blocks)
            extra += n_blocks * 2 * batch_size * R * R * D
    return extra


def load_model(args, input_shape=(2, 32, 32)):
    # Pass ranks / r_enc only to factories that accept them.
    sig = inspect.signature(dcrnet)
    kwargs = {'reduction': args.cr, 'expansion': args.expansion}
    # Plumb the input geometry (H, W) into factories that accept it.
    # Default for COST2100 is 32x32; for Rice it's 64x52.
    _, H, W = input_shape
    if 'h' in sig.parameters:
        kwargs['h'] = H
    if 'w' in sig.parameters:
        kwargs['w'] = W
    if args.ranks is not None and 'ranks' in sig.parameters:
        kwargs['ranks'] = args.ranks
    if args.r_enc is not None and 'r_enc' in sig.parameters:
        kwargs['r_enc'] = args.r_enc
    if args.d_emb is not None and 'd_emb' in sig.parameters:
        kwargs['d_emb'] = args.d_emb
    if args.refine_width is not None and 'refine_width' in sig.parameters:
        kwargs['refine_width'] = args.refine_width
    if args.refine_K is not None and 'refine_K' in sig.parameters:
        kwargs['refine_K'] = args.refine_K
    if args.no_dilate_path and 'use_dilate_path' in sig.parameters:
        kwargs['use_dilate_path'] = False
    if args.rank_attn_blocks > 0 and 'rank_attn_blocks' in sig.parameters:
        kwargs['rank_attn_blocks'] = args.rank_attn_blocks
        kwargs['rank_attn_dim'] = args.rank_attn_dim
        kwargs['rank_attn_heads'] = args.rank_attn_heads
    if args.no_film and 'use_film' in sig.parameters:
        kwargs['use_film'] = False
    if args.neural_decoder and 'neural_decoder' in sig.parameters:
        kwargs['neural_decoder'] = True
        kwargs['dec_d_emb'] = args.dec_d_emb
    if args.hybrid_decoder and 'hybrid_decoder' in sig.parameters:
        kwargs['hybrid_decoder'] = True
        kwargs['hybrid_direct_ranks'] = args.hybrid_direct_ranks
        kwargs['hybrid_extra_ranks'] = args.hybrid_extra_ranks
        kwargs['hybrid_extra_d_emb'] = args.hybrid_extra_d_emb
        kwargs['hybrid_gate_bias'] = args.hybrid_gate_bias
        kwargs['hybrid_alpha_init'] = args.hybrid_alpha_init
    if args.enc_fuse_1x1 and 'enc_fuse_1x1' in sig.parameters:
        kwargs['enc_fuse_1x1'] = True
    if args.enc_fuse_mlp > 0 and 'enc_fuse_mlp' in sig.parameters:
        kwargs['enc_fuse_mlp'] = args.enc_fuse_mlp
    if args.enc_fuse_linear and 'enc_fuse_linear' in sig.parameters:
        kwargs['enc_fuse_linear'] = True
    if args.complex_encoder and 'complex_encoder' in sig.parameters:
        kwargs['complex_encoder'] = True
    if args.dec_dilate and 'dec_dilate' in sig.parameters:
        kwargs['dec_dilate'] = True
        kwargs['dec_dilate_width'] = args.dec_dilate_width
    model = dcrnet(**kwargs)
    image = torch.randn([1, *input_shape])
    flops_thop, _ = thop.profile(model, inputs=(image,), verbose=False)
    flops = flops_thop + _einsum_macs(model, batch_size=1)
    # thop also misses bare nn.Parameter (u, v in v5/v6, scale/bias, gates);
    # sum(p.numel()) is authoritative and avoids the discrepancy.
    params = sum(p.numel() for p in model.parameters())
    flops_str, params_str = thop.clever_format([flops, params], "%.3f")
    flops_thop_str, _ = thop.clever_format([flops_thop, 0], "%.3f")
    return model, flops_str, params_str, flops_thop_str

def main():


    try:
        os.makedirs(args.outputs+"/log")
        os.makedirs(args.outputs + "/checkpoints")

    except OSError:
        pass
    t=time.strftime("%m%d%H%M", time.localtime())
    tag_suffix = f"-{args.tag}" if args.tag else ""
    logger = get_logger(args.outputs + f"/log/{t}-{args.model}-{args.expansion}X-{args.cr}-{args.scenario}{tag_suffix}.log")
    # init device
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        device="cuda"
        pin_memory=True
        torch.backends.cudnn.benchmark = True
        print(f"Use gpu {args.gpu}")
    else:
        device="cpu"
        pin_memory = False

    # Build the data loader first so we know the input shape, then size the
    # model and the thop dummy accordingly.
    if args.dataset == 'cost2100':
        loader_factory = Cost2100DataLoader(
            root=args.data, batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=pin_memory,
            scenario=args.scenario)
        input_shape = (2, 32, 32)
    elif args.dataset == 'rice':
        loader_factory = RiceFDDDataLoader(
            root=args.data, batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=pin_memory,
            scenario=args.scenario, channel=args.rice_channel,
            domain=args.rice_domain, split=args.rice_split,
            train_frac=args.rice_train_frac)
        input_shape = loader_factory.shape   # (2, 96, 52)
    elif args.dataset == 'deepmimo':
        loader_factory = DeepMIMODataLoader(
            root=args.data, batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=pin_memory)
        input_shape = loader_factory.shape   # (2, 32, 32)
    else:  # rtcsi
        loader_factory = RTCSIDataLoader(
            root=args.data, batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=pin_memory,
            scene=args.rtcsi_scene)
        input_shape = loader_factory.shape   # (2, 32, 32)

    model,flops,params,flops_thop=load_model(args, input_shape=input_shape)
    # Detect whether this model's forward accepts the v9-only kwargs
    # (clamp, return_coarse). v1/v5/v8 do not and would TypeError if passed.
    _fwd_params = inspect.signature(model.forward).parameters
    model._supports_clamp = 'clamp' in _fwd_params
    model._supports_coarse = 'return_coarse' in _fwd_params
    logger.info(f'Model Name: DCRNet-{args.model} [pretrained: {args.pretrained}]')
    logger.info(f'Model Config: dataset={args.dataset} input={input_shape} '
                f'compression ratio=1/{args.cr} expansion={args.expansion} '
                f'ranks={args.ranks} r_enc={args.r_enc}')
    if args.dataset == 'rice':
        logger.info(f'Rice split[{loader_factory.split_mode}]: '
                    f'train={loader_factory.n_train} ({loader_factory.n_train_locs} locs)  '
                    f'val={loader_factory.n_val} ({loader_factory.n_val_locs} locs)  '
                    f'test={loader_factory.n_test} ({loader_factory.n_test_locs} locs)  '
                    f'norm scale={loader_factory.scale:.4g}  '
                    f'channel={args.rice_channel}  domain={args.rice_domain}')
    logger.info(f'Model Flops: {flops}  (thop alone: {flops_thop} — einsum-corrected)')
    logger.info(f'Model Params Num: {params}\n')
    model.to(device)
    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        state_dict = torch.load(args.pretrained,
                                map_location=torch.device(device))['state_dict']
        print("pretrained model loaded from {}".format(args.pretrained))
        model.load_state_dict(state_dict,strict=False)

    if args.init_from_v5:
        assert os.path.isfile(args.init_from_v5), args.init_from_v5
        v5_sd = torch.load(args.init_from_v5,
                           map_location=torch.device(device))['state_dict']
        # v5 uses encoder.* / decoder.*; v9 uses enc_lowrank.* / decoder.*.
        remap = {}
        for k, v in v5_sd.items():
            if k.startswith('total_ops') or k.startswith('total_params'):
                continue
            if '.total_ops' in k or '.total_params' in k:
                continue
            if k.startswith('encoder.'):
                remap['enc_lowrank.' + k[len('encoder.'):]] = v
            elif k.startswith('decoder.'):
                remap['decoder.' + k[len('decoder.'):]] = v
        missing, unexpected = model.load_state_dict(remap, strict=False)
        loaded_keys = [k for k in remap if k not in unexpected]
        if args.freeze_v5:
            for n, p in model.named_parameters():
                if n.startswith('enc_lowrank.') or n.startswith('decoder.'):
                    p.requires_grad = False
        n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'init from v5: {args.init_from_v5}  (freeze={args.freeze_v5})')
        logger.info(f'  loaded {len(loaded_keys)} tensors into enc_lowrank+decoder, '
                    f'unexpected={len(unexpected)}')
        logger.info(f'  frozen={n_frozen/1e3:.1f}K params, trainable={n_train/1e3:.1f}K params')

    # Create the data loader (factory was constructed above)
    train_loader, val_loader, test_loader = loader_factory()
    criterion=nn.MSELoss()
    if args.init_from_v5 and args.backbone_lr is not None and not args.freeze_v5:
        backbone_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and (n.startswith('enc_lowrank.')
                                                    or n.startswith('decoder.'))]
        new_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and not (n.startswith('enc_lowrank.')
                                                   or n.startswith('decoder.'))]
        param_groups = [
            {'params': backbone_params, 'lr': args.backbone_lr},
            {'params': new_params,      'lr': args.lr},
        ]
        logger.info(f'param groups: backbone lr={args.backbone_lr} '
                    f'({sum(p.numel() for p in backbone_params)/1e3:.1f}K params) | '
                    f'new lr={args.lr} '
                    f'({sum(p.numel() for p in new_params)/1e3:.1f}K params)')
        optimizer = torch.optim.Adam(param_groups,
                                     weight_decay=args.weight_decay)
    else:
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=args.lr,
                                     weight_decay=args.weight_decay)
    num_steps = len(train_loader) * args.epochs
    if args.scheduler == 'const':
        scheduler = ConstLR(optimizer=optimizer)
    else:
        scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                                 T_max=num_steps,
                                                 T_warmup=30 * len(train_loader),
                                                 eta_min=5e-5)
    #warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=10)
    #scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=2500,eta_min=5e-5)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu:
                checkpoint = torch.load(args.resume)
            else:
                # Load GPU model on CPU
                checkpoint = torch.load(args.resume)
            args.start_epoch = 0 if args.warm_restart else checkpoint['epoch']
            best_nmse = 10 if args.warm_restart else checkpoint['best_nmse']
            best_epoch = -1 if args.warm_restart else checkpoint.get('best_epoch', checkpoint['epoch'] - 1)
            missing, unexpected = model.load_state_dict(
                checkpoint['state_dict'], strict=not args.warm_restart)
            if args.warm_restart:
                logger.info(f'warm-restart: fresh optimizer + scheduler, '
                            f'weights loaded (missing={len(missing)} new, '
                            f'unexpected={len(unexpected)})')
            else:
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            print("loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("no checkpoint found at '{}'".format(args.resume))
    else:
        best_nmse = 10
        best_epoch = -1

    if args.evaluate:
        print("=> evaluating...")
        test(test_loader, device,model,criterion, dataset=args.dataset)
        return
    for epoch in range(args.start_epoch,args.epochs):

        # Optional sigma_bias schedule (v29svd2/v29svd3 etc.)
        if hasattr(model, 'update_sigma_bias'):
            model.update_sigma_bias(epoch, args.epochs)
        training_loss=train(train_loader,device, model, criterion, optimizer,scheduler,epoch)
        logger.info(f"Epoch: {epoch + 1} traning loss={training_loss:3e}")
        if (epoch+1) % args.val_freq==0:
            print("=> val:")
            nmse=test(test_loader,device,model,criterion,epoch, dataset=args.dataset)
            improved = nmse <= best_nmse
            if improved:
                best_nmse=nmse
                best_epoch=epoch
            logger.info(f"Test Epoch: [{epoch+1}/{args.epochs}] nmese={nmse:.6f} (dB) "
                        f"[best={best_nmse:.6f} @ep{best_epoch+1}] {'NEW BEST' if improved else ''}")
            if improved:
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_nmse': best_nmse,
                    'best_epoch': best_epoch
                }
                torch.save(state, os.path.join(args.outputs+'/checkpoints',f"{args.model}-{args.expansion}X-cr{args.cr}-{args.scenario}{tag_suffix}" ))
            print(f"=> best nmse:{best_nmse:3e} (db), epoch:{best_epoch+1}\n\n")






def train(train_loader,device, model, criterion, optimizer, scheduler,epoch):
    iter_loss = AverageMeter('Iter loss')
    iter_time = AverageMeter('Iter time')
    time_tmp = time.time()
    model.train()
    t = tqdm(train_loader, file=sys.stdout, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET),ncols=150)
    # Time-varying coarse loss weight (peak first 40%, linear decay 40%-80%, 0 after)
    if args.coarse_anneal and args.coarse_loss_weight > 0:
        e0 = 0.4 * args.epochs
        e1 = 0.8 * args.epochs
        if epoch < e0:
            cur_coarse_w = args.coarse_loss_weight
        elif epoch < e1:
            cur_coarse_w = args.coarse_loss_weight * (1.0 - (epoch - e0) / (e1 - e0))
        else:
            cur_coarse_w = 0.0
    else:
        cur_coarse_w = args.coarse_loss_weight
    clamp_train = not args.no_train_clamp

    for batch_idx, (sparse_gt,) in enumerate(t):
        sparse_gt = sparse_gt.to(device)

        optimizer.zero_grad()
        # v9 supports return_coarse/clamp; v1/v5/v8 do not — gate by signature.
        if cur_coarse_w > 0 and model._supports_coarse:
            sparse_pred, coarse = model(sparse_gt, return_coarse=True, clamp=clamp_train)
            loss = criterion(sparse_pred, sparse_gt) \
                   + cur_coarse_w * criterion(coarse, sparse_gt)
        elif model._supports_clamp:
            sparse_pred = model(sparse_gt, clamp=clamp_train)
            loss = criterion(sparse_pred, sparse_gt)
        else:
            sparse_pred = model(sparse_gt)
            loss = criterion(sparse_pred, sparse_gt)
        loss.backward()
        optimizer.step()
        scheduler.step()
        # Log and visdom update
        iter_loss.update(loss)
        iter_time.update(time.time() - time_tmp)
        time_tmp = time.time()

        t.set_description(f"Epoch:[{epoch+1}/{args.epochs}]")
        t.set_postfix({"lr":f"{scheduler.get_last_lr()[0]:.2e}",
            "MSE loss":f"{iter_loss.avg:.3e}",
        })
        # if (batch_idx + 1) % args.print_freq == 0:
        #     print(f'Epoch: [{epoch}/{args.epochs}]'
        #                 f'[{batch_idx + 1}/{len(train_loader)}] '
        #                 f'lr: {optimizer.param_groups[0]["lr"]:.2e} | '
        #                 f'MSE loss: {iter_loss.avg:.3e} | '
        #                 f'time: {iter_time.avg:.3f}')

    return iter_loss.avg

def test(data_loader,device,model,criterion,epoch=1, dataset='cost2100'):
    iter_rho = AverageMeter('Iter rho')
    iter_nmse = AverageMeter('Iter nmse')
    iter_loss = AverageMeter('Iter loss')
    t = tqdm(data_loader, file=sys.stdout, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET), ncols=150)
    model.eval()
    with torch.no_grad():
        for batch_idx, (sparse_gt, raw_gt) in enumerate(t):
            sparse_gt = sparse_gt.to(device)
            sparse_pred = model(sparse_gt)
            loss = criterion(sparse_pred, sparse_gt)
            if dataset == 'cost2100':
                rho, nmse = evaluator(sparse_pred, sparse_gt, raw_gt)
            else:
                # Rice / DeepMIMO: no separate raw full-band tensor in the
                # same sense as COST2100's HtestF mat (Rice has only 52 sub
                # = full bandwidth; DeepMIMO is already angular-delay).
                # Report NMSE only and pin ρ to NaN so the average is clean.
                nmse = _nmse_only(sparse_pred, sparse_gt)
                rho = torch.tensor(float('nan'))
            # Log and visdom update
            iter_loss.update(loss)
            iter_rho.update(rho)
            iter_nmse.update(nmse)
            t.set_description(f"Testing: Epoch:[{epoch + 1}]")
            t.set_postfix({"MSE loss": f"{iter_loss.avg:.3e}",
                           "NMSE(db)": f"{iter_nmse.avg:.3e}",
                           "rho":f"{iter_rho.avg:.3e}"
                           })
    return iter_nmse.avg


def _nmse_only(sparse_pred, sparse_gt):
    """NMSE in dB on the (2, H, W) tensor, with the same de-centering
    convention as evaluator() but no FFT/ρ step. Works for any H, W."""
    if os.environ.get('DCRNET_CENTERED', '0') != '1':
        sparse_gt = sparse_gt - 0.5
        sparse_pred = sparse_pred - 0.5
    power_gt = sparse_gt[:, 0] ** 2 + sparse_gt[:, 1] ** 2
    diff = sparse_gt - sparse_pred
    mse = diff[:, 0] ** 2 + diff[:, 1] ** 2
    return 10 * torch.log10((mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean())
class AverageMeter(object):
    r"""Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, name):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"==> For {self.name}: sum={self.sum}; avg={self.avg}"

def evaluator(sparse_pred, sparse_gt, raw_gt):
    r""" Evaluation of decoding implemented in PyTorch Tensor
         Computes normalized mean square error (NMSE) and rho.
    """

    with torch.no_grad():
        # Basic params
        nt = 32
        nc = 32
        nc_expand = 257

        # De-centralize
        sparse_gt = sparse_gt - 0.5
        sparse_pred = sparse_pred - 0.5

        # Calculate the NMSE
        power_gt = sparse_gt[:, 0, :, :] ** 2 + sparse_gt[:, 1, :, :] ** 2
        difference = sparse_gt - sparse_pred
        mse = difference[:, 0, :, :] ** 2 + difference[:, 1, :, :] ** 2
        nmse = 10 * torch.log10((mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean())

        # Calculate the Rho
        n = sparse_pred.size(0)
        sparse_pred = sparse_pred.permute(0, 2, 3, 1)  # Move the real/imaginary dim to the last
        zeros = sparse_pred.new_zeros((n, nt, nc_expand - nc, 2))
        sparse_pred = torch.cat((sparse_pred, zeros), dim=2)
        raw_pred = torch.view_as_real(torch.fft.fft(torch.view_as_complex(sparse_pred.contiguous()), dim=-1))[:, :, :125, :]

        norm_pred = raw_pred[..., 0] ** 2 + raw_pred[..., 1] ** 2
        norm_pred = torch.sqrt(norm_pred.sum(dim=1))

        norm_gt = raw_gt[..., 0] ** 2 + raw_gt[..., 1] ** 2
        norm_gt = torch.sqrt(norm_gt.sum(dim=1))

        real_cross = raw_pred[..., 0] * raw_gt[..., 0] + raw_pred[..., 1] * raw_gt[..., 1]
        real_cross = real_cross.sum(dim=1)
        imag_cross = raw_pred[..., 0] * raw_gt[..., 1] - raw_pred[..., 1] * raw_gt[..., 0]
        imag_cross = imag_cross.sum(dim=1)
        norm_cross = torch.sqrt(real_cross ** 2 + imag_cross ** 2)

        rho = (norm_cross / (norm_pred * norm_gt)).mean()

        return rho, nmse

if __name__ == '__main__':
    main()
"""Generate DeepMIMO O1 (Outdoor1, 3.4 GHz) CSI feedback dataset.

User-supplied protocol:
    - Carrier 3.4 GHz   -> scenario 'O1_3p4'
    - Bandwidth 10 MHz
    - Nf = 1024  (OFDM subcarriers)
    - Nc = 32    (delay taps kept after IFFT)
    - Nt = 32    (BS ULA antennas)
    - Num paths = 10
    - Outdoor NLoS only
    - 20k train / 10k val / 10k test

Pipeline:
    DeepMIMOv3.generate_data -> per-UE H ∈ C^{32_ant x 1024_sub}
    -> angular-delay: DFT on ant, IFFT on sub, keep first 32 delay taps
    -> stack (re, im), dataset-wide max-abs normalize to [0, 1]
    -> (2, 32, 32) float32

Run inside the v3 venv (/tmp/deepmimov3_venv) because DeepMIMOv4 doesn't
read the legacy raw-raytracing .mat layout.

Usage:
    source /tmp/deepmimov3_venv/bin/activate
    python tools/prepare_deepmimo_o1.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenarios-dir',
                    default='/home/ubuntu/Documents/dataset/DeepMIMO',
                    help='parent dir containing O1_3p4/*.mat')
    ap.add_argument('--scenario', default='O1_3p4')
    ap.add_argument('--bs', type=int, default=16,
                    help='active BS (BS 16 is on a side-street and has '
                         '~75% NLoS UEs across the user grid; BS 3-12 have '
                         '100% LoS in O1_3p4)')
    ap.add_argument('--n-ant', type=int, default=32, help='Nt (ULA size)')
    ap.add_argument('--n-sub', type=int, default=1024, help='Nf (subcarriers)')
    ap.add_argument('--n-delay', type=int, default=32, help='Nc (delay taps kept)')
    ap.add_argument('--n-paths', type=int, default=10)
    ap.add_argument('--bw-mhz', type=float, default=10.0, help='OFDM bandwidth in MHz')
    ap.add_argument('--user-rows', type=int, default=2700,
                    help='number of user grid rows from row 1 (each row ~181 UEs). '
                         'BS 16 + 2700 rows + NLoS filter gives ~365K UEs to draw from.')
    ap.add_argument('--user-subsampling', type=float, default=0.1,
                    help='Fraction of UEs to sample (0,1]. Default 0.1 caps '
                         'channel-gen memory at ~12GB. At BS 16, 488K UEs × '
                         '0.1 × 73%% NLoS ≈ 36K NLoS UEs (> 30K we need).')
    ap.add_argument('--n-train', type=int, default=20000)
    ap.add_argument('--n-val',   type=int, default=0,
                    help='0 = no separate val (val mirrors test in loader)')
    ap.add_argument('--n-test',  type=int, default=10000)
    ap.add_argument('--nlos-only', action='store_true', default=True)
    ap.add_argument('--dst', default='/home/ubuntu/Documents/dataset/DeepMIMO_O1_3p4')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    import DeepMIMOv3 as DM

    p = DM.default_params()
    p['dataset_folder'] = args.scenarios_dir
    p['scenario'] = args.scenario
    p['num_paths'] = args.n_paths
    p['active_BS'] = [args.bs]
    p['user_rows'] = list(range(1, args.user_rows + 1))
    p['user_subsampling'] = args.user_subsampling
    p['enable_BS2BS'] = 0
    p['OFDM_channels'] = 1                                  # freq-domain channel matrix
    p['bs_antenna']['shape'] = np.array([args.n_ant, 1])    # 32x1 ULA
    p['bs_antenna']['spacing'] = 0.5
    p['ue_antenna']['shape'] = np.array([1, 1])
    p['OFDM']['subcarriers'] = args.n_sub
    p['OFDM']['selected_subcarriers'] = np.arange(args.n_sub)
    p['OFDM']['bandwidth'] = args.bw_mhz / 1000.0           # DeepMIMOv3 uses GHz

    print(f'generating channels for scenario={args.scenario} BS={args.bs} '
          f'rows=1..{args.user_rows}  n_ant={args.n_ant}  n_sub={args.n_sub}  '
          f'paths={args.n_paths}  bw={args.bw_mhz}MHz')
    t0 = time.time()
    ds = DM.generate_data(p)
    print(f'  generated in {time.time()-t0:.1f}s')

    # ds is a list (one per active BS). ds[0]['user']['channel'] is per-UE
    # complex channel with shape (n_ue, n_rx_ant, n_tx_ant, n_sub).
    user_data = ds[0]['user']
    H = user_data['channel']            # (n_ue, 1, n_ant, n_sub) usually
    los = user_data['LoS']              # (n_ue,)  +1 LoS, 0 NLoS, -1 no path
    print(f'  channel shape: {H.shape}, los shape: {los.shape}')
    H = np.asarray(H, dtype=np.complex64)
    if H.ndim == 4 and H.shape[1] == 1:
        H = H[:, 0, :, :]               # (n_ue, n_ant, n_sub)
    assert H.shape[1] == args.n_ant and H.shape[2] == args.n_sub, (
        f'shape mismatch: got {H.shape}, expected (*, {args.n_ant}, {args.n_sub})')

    los = np.asarray(los)
    print(f'  LoS={int((los==1).sum())}  NLoS={int((los==0).sum())}  '
          f'no_path={int((los==-1).sum())}')

    if args.nlos_only:
        mask = (los == 0)
        H = H[mask]
        print(f'  after NLoS filter: N={len(H)}')
    else:
        mask = (los != -1)
        H = H[mask]
        print(f'  after no-path filter: N={len(H)}')

    n_need = args.n_train + args.n_val + args.n_test
    if len(H) < n_need:
        sys.exit(f'  not enough UEs ({len(H)} < {n_need}); '
                 f'increase --user-rows or --no-nlos-only')

    # Angular-delay transform: DFT on ant axis, IFFT on sub axis, crop to N_c delay taps
    print('applying angular-delay transform ...')
    Ha = np.fft.fft(H, axis=1, norm='ortho')                # (N, Na, Nf)
    Had = np.fft.ifft(Ha, axis=2, norm='ortho')             # (N, Na, Nf)
    Had = Had[:, :, :args.n_delay]                          # (N, Na, Nc)
    print(f'  Had shape: {Had.shape}  dtype: {Had.dtype}')

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(Had))
    idx_tr = perm[:args.n_train]
    idx_te = perm[args.n_train:args.n_train + args.n_test]
    if args.n_val > 0:
        idx_va = perm[args.n_train + args.n_test:
                      args.n_train + args.n_test + args.n_val]
    else:
        idx_va = idx_te                 # val mirrors test (user spec: 20k/10k split)

    # Per-sample max-abs normalization.
    #
    # WHY not global: DeepMIMO outdoor NLoS at 3.4 GHz has 100-1000x path-loss
    # variation between near and far UEs. A global max-abs scale is dominated
    # by the strongest UE, squishing 99% of samples to mean=0.5 ± 0.004 —
    # any reconstruction error then dwarfs the signal power and NMSE explodes
    # to +20 dB. Per-sample normalization preserves each UE's full dynamic
    # range and matches what CSINet/CRNet do for DeepMIMO.
    #
    # Each H is scaled so max(|H.re|, |H.im|) → 0.5, then +0.5 → [0, 1].
    # Eps guards UEs with all-zero channels.
    def to_2chw_per_sample(H):
        scales = np.maximum(np.abs(H.real).reshape(len(H), -1).max(axis=1),
                            np.abs(H.imag).reshape(len(H), -1).max(axis=1))
        scales = np.maximum(scales, 1e-12)[:, None, None]
        re = (H.real / scales) * 0.5 + 0.5
        im = (H.imag / scales) * 0.5 + 0.5
        return np.stack([re, im], axis=1).astype(np.float32)

    x_tr = to_2chw_per_sample(Had[idx_tr])
    x_va = to_2chw_per_sample(Had[idx_va])
    x_te = to_2chw_per_sample(Had[idx_te])

    # Report train-split stats so we can verify the data really fills [0,1].
    xc = x_tr - 0.5
    print(f'  train post-norm:  re std={xc[:,0].std():.4f}  '
          f'im std={xc[:,1].std():.4f}  signal_power median='
          f'{((xc[:,0]**2+xc[:,1]**2).sum(axis=(1,2)).mean()):.4f}')
    print(f'  train={x_tr.shape}  val={x_va.shape} (mirrors test)  test={x_te.shape}')
    scale = float(np.nan)  # not a single scalar anymore

    os.makedirs(args.dst, exist_ok=True)
    out = os.path.join(args.dst, 'deepmimo_o1_3p4.npz')
    np.savez(out,
             x_train=x_tr, x_val=x_va, x_test=x_te,
             scale=np.float32(scale),
             config=dict(scenario=args.scenario, bs=args.bs,
                         n_ant=args.n_ant, n_sub=args.n_sub,
                         n_delay=args.n_delay, n_paths=args.n_paths,
                         bw_mhz=args.bw_mhz, seed=args.seed))
    print(f'wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()

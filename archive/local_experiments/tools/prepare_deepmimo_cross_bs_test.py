"""Generate a DeepMIMO O1_3p4 test set from a DIFFERENT BS than the one
used in training, to measure cross-BS generalization (no spatial leakage
from dense user grid).

Default: train was BS 16, test now from BS 18. Both are side-street BSs
with ~75% NLoS UEs, so geometry is comparable.

Output: deepmimo_o1_3p4_bs{N}_test.npz with key `x_test` shape (10000, 2, 32, 32).
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
                    default='/home/ubuntu/Documents/dataset/DeepMIMO')
    ap.add_argument('--scenario', default='O1_3p4')
    ap.add_argument('--test-bs', type=int, default=18,
                    help='BS to use for the cross-BS test set')
    ap.add_argument('--n-ant', type=int, default=32)
    ap.add_argument('--n-sub', type=int, default=1024)
    ap.add_argument('--n-delay', type=int, default=32)
    ap.add_argument('--n-paths', type=int, default=10)
    ap.add_argument('--bw-mhz', type=float, default=10.0)
    ap.add_argument('--user-rows', type=int, default=2700)
    ap.add_argument('--user-subsampling', type=float, default=0.1)
    ap.add_argument('--n-test', type=int, default=10000)
    ap.add_argument('--nlos-only', action='store_true', default=True)
    ap.add_argument('--dst', default='/home/ubuntu/Documents/dataset/DeepMIMO_O1_3p4')
    ap.add_argument('--seed', type=int, default=1)         # different seed from train
    args = ap.parse_args()

    import DeepMIMOv3 as DM
    p = DM.default_params()
    p['dataset_folder'] = args.scenarios_dir
    p['scenario'] = args.scenario
    p['num_paths'] = args.n_paths
    p['active_BS'] = [args.test_bs]
    p['user_rows'] = list(range(1, args.user_rows + 1))
    p['user_subsampling'] = args.user_subsampling
    p['enable_BS2BS'] = 0
    p['OFDM_channels'] = 1
    p['bs_antenna']['shape'] = np.array([args.n_ant, 1])
    p['bs_antenna']['spacing'] = 0.5
    p['ue_antenna']['shape'] = np.array([1, 1])
    p['OFDM']['subcarriers'] = args.n_sub
    p['OFDM']['selected_subcarriers'] = np.arange(args.n_sub)
    p['OFDM']['bandwidth'] = args.bw_mhz / 1000.0

    print(f'generating channels for {args.scenario} BS={args.test_bs} '
          f'subsampling={args.user_subsampling}')
    t0 = time.time()
    ds = DM.generate_data(p)
    print(f'  generated in {time.time()-t0:.1f}s')

    H = np.asarray(ds[0]['user']['channel'], dtype=np.complex64)
    los = np.asarray(ds[0]['user']['LoS'])
    if H.ndim == 4 and H.shape[1] == 1:
        H = H[:, 0, :, :]
    print(f'  channel shape: {H.shape}  los: LoS={int((los==1).sum())} '
          f'NLoS={int((los==0).sum())}  no_path={int((los==-1).sum())}')

    if args.nlos_only:
        H = H[los == 0]
        print(f'  after NLoS filter: N={len(H)}')

    if len(H) < args.n_test:
        sys.exit(f'  not enough UEs: {len(H)} < {args.n_test}')

    # Angular-delay transform
    Ha = np.fft.fft(H, axis=1, norm='ortho')
    Had = np.fft.ifft(Ha, axis=2, norm='ortho')
    Had = Had[:, :, :args.n_delay]

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(Had))[:args.n_test]

    # Per-sample max-abs normalization (matches training preprocessing)
    H_test = Had[idx]
    scales = np.maximum(np.abs(H_test.real).reshape(len(H_test), -1).max(axis=1),
                        np.abs(H_test.imag).reshape(len(H_test), -1).max(axis=1))
    scales = np.maximum(scales, 1e-12)[:, None, None]
    re = (H_test.real / scales) * 0.5 + 0.5
    im = (H_test.imag / scales) * 0.5 + 0.5
    x_test = np.stack([re, im], axis=1).astype(np.float32)

    print(f'  x_test: {x_test.shape}  '
          f'signal_power median={((x_test - 0.5)**2).sum(axis=(1,2,3)).mean():.4f}')

    os.makedirs(args.dst, exist_ok=True)
    out = os.path.join(args.dst, f'deepmimo_o1_3p4_bs{args.test_bs}_test.npz')
    np.savez(out, x_test=x_test,
             config=dict(scenario=args.scenario, test_bs=args.test_bs,
                         n_ant=args.n_ant, n_sub=args.n_sub,
                         n_delay=args.n_delay, n_paths=args.n_paths,
                         bw_mhz=args.bw_mhz, seed=args.seed))
    print(f'wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()

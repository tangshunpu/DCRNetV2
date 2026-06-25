# RT_CSI Multi-Scene Benchmark

NMSE (dB) results on Sionna-RT generated CSI feedback datasets across cities.
All scenes: 3.5 GHz, 1024 subcarriers, 32-ant ULA, angular-delay (2, 32, 32),
per-sample max-abs normalization to [0, 1].

Generated from `/home/ubuntu/Documents/project/RT_CSI/outputs/csi/csi_<scene>_3GHz_32x1024.npz`.

## Scene specs

| Scene    | train | val  | test | BS | UE std |
|----------|-------|------|------|----|--------|
| zju      | 24000 | 8000 | 8000 | 5  | 0.0161 |
| munich   | 24000 | 8000 | 8000 | 5  | 0.0172 |
| etoile   | 24000 | 8000 | 8000 | 5  | 0.0163 |
| florence | 24000 | 8000 | 8000 | 5  | 0.0163 |
| sutd     | 24000 | 8000 | 8000 | 5  | 0.0163 |
| shenzhen | 17012 | 5671 | 5670 | 5  | 0.0149 |
| zjubs0   | 3808  | 1232 | 1261 | 1  | (diagnostic: ZJU restricted to BS=0) |

## Training recipes (lr=2e-3, val-freq=5, batch=200)

- **v1** (`--model v1`): 2500 ep, default. After 2026-05-21 edit: Kaiming-normal Linear init (was normal std=0.001).
- **v5** (`--model v5 --r-enc 512 --ranks 16`): 2500 ep.
- **v9 unified** (`--model v9 --r-enc 512 --refine-width 2 --refine-k 1 --no-film --enc-fuse-mlp 64 --complex-encoder --hybrid-decoder --hybrid-direct-r 32 --hybrid-extra-r 16 --hybrid-extra-d 16 --hybrid-gate-bias -2.0 --hybrid-alpha-init 1e-2`): 1500 ep. **cr=4 must use lr=1e-3** (lr=2e-3 diverges on multi-BS data).
- **CRNet** (`--model crnet`): 2500 ep.
- **TransNet** (`--model transnet`, d_model=64): 2500 ep. **Fails on multi-BS RT_CSI** (Sigmoid saturation + LayerNorm collapse with small-std data).

---

## ZJU (5-BS, 24k)

Best NMSE (dB) on test set.

| CR | v1                  | v5     | v9 unified | CRNet  | TransNet |
|----|---------------------|--------|------------|--------|----------|
| 4  | −9.49               | −12.54 | **−15.42** (lr=1e-3) | −11.50 | −0.01    |
| 8  | −8.66 (seed=42)     | −10.92 | **−12.92** | −10.95 | −0.04    |
| 16 | −10.38              | −9.28  | **−10.95** | −10.61 | −0.10    |
| 32 | killed              | −7.36  | **−8.73**  | not run | not run  |

Notes:
- v1 cr=4/8/16/32 first attempt with normal(std=0.001) init: stuck at +6 dB / 0 dB. Switched to Kaiming-normal.
- v1 cr=8 needed `--seed 42` to escape an initial plateau.
- v9 unified cr=4 at lr=2e-3 diverged at ep70 (train loss kept falling, test NMSE went to 0); relaunched at lr=1e-3 (final −15.42).
- v5/v1/CRNet/TransNet runs use 2500 ep; v9 unified uses 1500 ep.

## munich (5-BS, 24k)

| CR | v1 | v5     | v9 unified | CRNet  | TransNet |
|----|----|--------|------------|--------|----------|
| 4  | TBD | −13.96 | **−17.23** @ep300 (lr=1e-3, killed ep329/1500) | −15.37 @ep1130 (killed ep1190/2500) | TBD |
| 8  | TBD | −10.62 | −13.40 @ep225 (killed ep328) | −13.18 @ep1145 (killed ep1193) | TBD |
| 16 | TBD |  −9.00 | −10.12 @ep145 (killed ep328) | **−11.17** @ep975 (killed ep1189) | TBD |

Status (2026-05-22): killed mid-training to test Frobenius normalization on ZJU. v9u took cr=4/8, CRNet cr=16. v9u best epochs all in first 13% of schedule (plateaued early — same pattern as ZJU 5-BS).

## etoile, florence, sutd, shenzhen, canyon

(Pending — to be run after munich completes.)

---

## Diagnostic: single-BS vs multi-BS (ZJU)

| CR | ZJU 5-BS (24k) | ZJU BS=0 only (3.8k) | Δ |
|----|----------------|-----------------------|---|
| 4  | −12.54 | −7.46 | −5.1 |
| 8  | −10.92 | −7.53 | −3.4 |
| 16 |  −9.28 | −6.35 | −2.9 |

**Conclusion:** the multi-BS distribution helps, not hurts. More data > less data even
with mixed-BS BatchNorm drift. Single-BS is data-starved at 3.8k frames.

## Normalization sweep on ZJU (2026-05-22)

Tested 5 output-layer × data-norm combinations on v1 + CRNet × cr=4/8/16. Best NMSE (dB):

| Model | CR | max-abs+Sigmoid | Frob+Sigmoid | Frob+FrobLayer | Frob+FrobLayer+α(fixed) | Frob+FrobLayer+α(z) | centered+Identity | centered+Tanh |
|-------|----|-----------------|---------------|----------------|------------------------|--------------------|--------------------|----------------|
| v1    | 4  | −9.49           | (DNF/saturated) | **−10.00**     | −10.42                 | −9.05              | −2.85 ❌           | −9.54          |
| v1    | 8  | −8.66           | (DNF)         | −9.36          | −9.45                  | **−10.16**         | −2.40 ❌           | −4.92 ❌       |
| v1    | 16 | **−10.38**      | (DNF)         | −9.26          | −9.23                  | −7.91              | −1.95 ❌           | −9.94          |
| CRNet | 4  | −11.50          | (DNF)         | **−12.27**     | −11.61                 | −11.38             | −11.61             | −11.35         |
| CRNet | 8  | −10.95          | (DNF)         | **−11.05**     | −10.67                 | −9.83              | −10.73             | −11.18         |
| CRNet | 16 | **−10.61**      | (DNF)         | −9.21          | −10.54                 | −10.26             | −9.33              | −10.34         |

**Verdict:** stick with **max-abs + Sigmoid (the original recipe)** as the default. The Frob layer variants converge dramatically faster at low cr but the final NMSE is comparable or worse at high cr. No clean "single config wins everywhere".

Implementation gates (in `models/dcrnet.py` + `models/crnet.py`, `dataset/rtcsi.py`, `main.py`):

- `DCRNET_FROBENIUS_OUTPUT=1` — replace Sigmoid with `FrobeniusNormOutput` (project to ||·||_F=0.5, shift +0.5). Use with Frobenius-normed npz at `--data /…/csi_frob/`.
- `DCRNET_FROBENIUS_ALPHA_Z=1` — adds a Linear head predicting shrinkage α ∈ [0.5, 1] from the codeword z.
- `DCRNET_CENTERED=1` — loader shifts data to centered [-1, 1] domain (s = 2(x-0.5)), model uses Identity output, evaluator skips −0.5. Combine with `DCRNET_TANH=1` to use Tanh output instead of Identity.

**Default (no env vars set): max-abs + Sigmoid.**

## Counter-intuitive findings

1. **Per-sample normalization, not global** — RT_CSI angular-delay tensors have
   highly skewed magnitudes; global max-abs makes 99% of samples have std≈0.004
   and models collapse to constant-0.5 predictions. Per-sample normalization
   restores std≈0.016, which is enough gradient signal.

2. **v1's `nn.init.normal_(m.weight, std=0.001)` is fatal on small-std data.**
   The bottleneck FC passes near-zero signal through; Sigmoid+small-std means
   output sticks at 0.5 for 100+ epochs. Switching Linear init to `kaiming_normal_`
   fixes it (single-line edit in `models/dcrnet.py:170`). Sigmoid kept (removing
   it makes the model collapse to "output data mean = 0.5" trivially).

3. **TransNet doesn't work on RT_CSI.** d_model=64 + Sigmoid + small-std data
   = train loss frozen at exactly data variance for all 2500 epochs.

4. **CRNet > v1 on RT_CSI.** Xavier-uniform init (vs Kaiming-normal) + LeakyReLU
   (vs PReLU) make CRNet escape plateaus faster on multi-BS data.

5. **v9 unified cr=4 needs lr=1e-3, not lr=2e-3.** The hybrid (direct=32 +
   extra=16) decoder has too much capacity for lr=2e-3 at the multi-BS
   distribution — diverges mid-training. lr=1e-3 stable.

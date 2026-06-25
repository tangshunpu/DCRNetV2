# DCRNet-V2 Family — Configs & Performance

A family of CSI feedback compression models for COST2100, derived from
DCRNet-V1. All share the same core: ComplexLowRankEncoder (Hermitian
bilinear matched-filter) + LowRankDecoder (rank-1 outer products) +
DCRDecoderRefine. Differences are in `r_enc`, decoder rank scheme,
and (for hybrid variants) α / gate tuning.

## TL;DR — 5 Production Configs

| Variant | FLOPs cr=4 | Use case |
|---------|-----------|----------|
| **DCRNetV2-mini** | 3.01 M | Ultra-edge / mobile |
| **DCRNetV2-small** | 4.07 M | Edge, sub-DCRNet-1× FLOPs |
| **DCRNetV2-base** | 5.41 M | Balanced server |
| **DCRNetV2-unified** | 6.77 M | One config across all cr × scenarios |
| **DCRNetV2-large** | 7.00 M | Per-cr SOTA (outdoor/indoor specialists) |

---

## 1. DCRNetV2-mini (= `v11cmh10e8`)

Hybrid decoder with mini-tier encoder.

```bash
python main.py --gpu 0 --model v9 --scenario {in|out} --cr {4|8|16|32} \
  --lr 2e-3 --val-freq 5 --epochs 1500 \
  --r-enc 128 \
  --refine-width 2 --refine-k 1 \
  --no-film --enc-fuse-mlp 32 --complex-encoder \
  --hybrid-decoder \
  --hybrid-direct-r 10 --hybrid-extra-r 8 --hybrid-extra-d 8 \
  --hybrid-gate-bias -2.0 --hybrid-alpha-init 1e-2 \
  --tag mini
```

### Performance

| cr | FLOPs (real) | Outdoor NMSE | Indoor NMSE |
|----|--------------|--------------|--------------|
| 4  | 3.01 M | −10.825 | −31.004 |
| 8  | 2.05 M | −7.201  | −19.233 |
| 16 | 1.57 M | −4.834  | −14.178 |
| 32 | 1.33 M | −3.139  | **−9.548** ★ |

★ indoor cr=32 −9.548 is Pareto SOTA — beats v11c-sm (−9.354) at 44% of its FLOPs.

---

## 2. DCRNetV2-small (= `v11c-tiny`)

Reduced `r_enc=256` for sub-DCRNet-1× FLOPs.

```bash
python main.py --gpu 0 --model v9 --scenario {in|out} --cr {4|8|16|32} \
  --lr 2e-3 --val-freq 5 --epochs 1500 \
  --r-enc 256 \
  --ranks 16 \
  --refine-width 2 --refine-k 1 \
  --no-film --enc-fuse-mlp 64 --complex-encoder \
  --tag small
```

### Performance

| cr | FLOPs (real) | Outdoor NMSE | Indoor NMSE |
|----|--------------|--------------|--------------|
| 4  | 4.07 M | −11.445 | −31.122 |
| 8  | 2.84 M | −7.697  | −20.511 |
| 16 | 2.23 M | −5.175  | −14.637 |
| 32 | 1.92 M | −3.314  | −9.155  |

---

## 3. DCRNetV2-base (= `v11c-sm`)

Paper baseline equivalent. Original `v11c-small` config.

```bash
python main.py --gpu 0 --model v9 --scenario {in|out} --cr {4|8|16|32} \
  --lr 2e-3 --val-freq 5 --epochs 1500 \
  --r-enc 512 \
  --ranks 16 \
  --refine-width 2 --refine-k 1 \
  --no-film --enc-fuse-mlp 64 --complex-encoder \
  --tag base
```

### Performance

| cr | FLOPs (real) | Outdoor NMSE | Indoor NMSE |
|----|--------------|--------------|--------------|
| 4  | 5.41 M | −11.890 | **−32.749** |
| 8  | 4.05 M | −7.832  | −20.073 |
| 16 | 3.37 M | −5.506  | **−14.731** |
| 32 | 3.03 M | −3.561  | −9.354  |

---

## 4. DCRNetV2-unified (= `v12_r`)

Hybrid decoder with direct=32 anchor + 16 cheap z-conditioned extras.
**Single config 5/8 wins across cr × scenarios.**

```bash
python main.py --gpu 0 --model v9 --scenario {in|out} --cr {4|8|16|32} \
  --lr 2e-3 --val-freq 5 --epochs 1500 \
  --r-enc 512 \
  --refine-width 2 --refine-k 1 \
  --no-film --enc-fuse-mlp 64 --complex-encoder \
  --hybrid-decoder \
  --hybrid-direct-r 32 --hybrid-extra-r 16 --hybrid-extra-d 16 \
  --hybrid-gate-bias -2.0 --hybrid-alpha-init 1e-2 \
  --tag unified
```

### Performance

| cr | FLOPs (real) | Outdoor NMSE | Indoor NMSE |
|----|--------------|--------------|--------------|
| 4  | 6.77 M | −12.987 | −32.837 |
| 8  | 4.81 M | −8.649  | **−20.364** |
| 16 | 3.83 M | **−5.846** | −14.617 |
| 32 | 3.35 M | **−3.793** | **−9.689** ★ |

★ indoor cr=32 SOTA across all variants.

---

## 5. DCRNetV2-large

**Three scenario-specific variants** at same 7.00M FLOPs (cr=4),
differing only in `--hybrid-gate-bias` and (for FTC) post-training reset.

### 5a. -large-out (= `v12_rX`) — Outdoor SOTA

```bash
python main.py --gpu 0 --model v9 --scenario out --cr {4|8|16|32} \
  --lr 2e-3 --val-freq 5 --epochs 1500 \
  --r-enc 512 --refine-width 2 --refine-k 1 \
  --no-film --enc-fuse-mlp 64 --complex-encoder \
  --hybrid-decoder \
  --hybrid-direct-r 32 --hybrid-extra-r 32 --hybrid-extra-d 16 \
  --hybrid-gate-bias -2.0 --hybrid-alpha-init 0.05 \
  --tag large_out
```

### 5b. -large-in4 (= `v12rXg`) — Indoor cr=4 SOTA

Only `--hybrid-gate-bias` differs (−4 vs −2):

```bash
python main.py --gpu 0 --model v9 --scenario in --cr 4 \
  ...same as above...
  --hybrid-gate-bias -4.0 --hybrid-alpha-init 0.05 \
  --tag large_in4
```

### 5c. -large-in16 (= `v12rXgFTC`) — Indoor cr=16 SOTA (fine-tune)

Two-stage: train `v12rXg` then reset `rank_gate.bias` to −1 + low-lr FT.

```bash
# Stage 1: train v12rXg (above) on indoor cr=16
# Stage 2: reset gate then fine-tune
python -c "
import torch
ckpt = torch.load('outputs/checkpoints/v9-1X-cr16-in-large_in4')
ckpt['state_dict']['decoder.extra.rank_gate.bias'].fill_(-1.0)
torch.save(ckpt, '/tmp/large_in16_init.pt')
"

python main.py --gpu 0 --model v9 --scenario in --cr 16 \
  --pretrained /tmp/large_in16_init.pt \
  --lr 1e-4 --val-freq 5 --epochs 200 \
  --r-enc 512 --refine-width 2 --refine-k 1 \
  --no-film --enc-fuse-mlp 64 --complex-encoder \
  --hybrid-decoder \
  --hybrid-direct-r 32 --hybrid-extra-r 32 --hybrid-extra-d 16 \
  --hybrid-gate-bias -1.0 --hybrid-alpha-init 0.05 \
  --tag large_in16
```

### Performance — DCRNetV2-large family

| cr/sc | Variant | FLOPs | NMSE | vs paper |
|-------|---------|-------|------|----------|
| 4 out  | large-out | 7.00 M | **−13.354** | DCRNet-1× −12.58 (**+0.77**) |
| 8 out  | large-out | 4.98 M | **−8.926** | DCRNet-1× −7.95 (**+0.98**) |
| 16 out | large-out | 3.97 M | **−5.914** | DCRNet-1× −5.60 (+0.31) |
| 32 out | large-out | 3.46 M | **−3.825** | DCRNet-1× −3.47 (+0.36) |
| 4 in   | large-in4 | 7.00 M | **−33.197** | TransNet −32.38 (**+0.82**) ★ |
| 16 in  | large-in16 | 3.97 M | **−14.794** | LRP-1× −13.91 (+0.88) |

★ Indoor cr=4 beats TransNet paper baseline at 1/5 the FLOPs.

---

## DCRNet-V1 paper baselines (reference)

| cr | FLOPs | Indoor | Outdoor |
|----|-------|--------|---------|
| 4  | 4.01 M | −28.04 | −12.58 |
| 8  | 2.96 M | −16.26 | −7.95  |
| 16 | 2.44 M | −11.74 | −5.60  |
| 32 | 2.18 M | −9.05  | −3.47  |

## DCRNetV2 vs DCRNet-V1 — Indoor

| cr | V1 paper | DCRNetV2-mini | -small | -base | -unified | -large |
|----|----------|---------------|--------|-------|----------|--------|
| 4  | −28.04 | −31.00 (+2.96) | −31.12 (+3.08) | −32.75 (+4.71) | −32.84 (+4.80) | **−33.20** (+5.16) |
| 8  | −16.26 | −19.23 (+2.97) | −20.51 (+4.25) | −20.07 (+3.81) | **−20.36** (+4.10) | n/a |
| 16 | −11.74 | −14.18 (+2.44) | −14.64 (+2.90) | −14.73 (+2.99) | −14.62 (+2.88) | **−14.79** (+3.05) |
| 32 | −9.05  | **−9.55** (+0.50) | −9.16 (+0.11) | −9.35 (+0.30) | −9.69 (+0.64) | n/a |

**DCRNetV2 全部击穿 DCRNet-V1 indoor 4 cr by 0.1-5.2 dB.**

## DCRNetV2 vs DCRNet-V1 — Outdoor

| cr | V1 paper | DCRNetV2-mini | -small | -base | -unified | -large |
|----|----------|---------------|--------|-------|----------|--------|
| 4  | −12.58 | −10.83 (−1.75) | −11.44 (−1.14) | −11.89 (−0.69) | −12.99 (+0.41) | **−13.35** (+0.77) |
| 8  | −7.95  | −7.20 (−0.75) | −7.70 (−0.25) | −7.83 (−0.12) | **−8.65** (+0.70) | **−8.93** (+0.98) |
| 16 | −5.60  | −4.83 (−0.77) | −5.17 (−0.43) | −5.51 (−0.09) | −5.85 (+0.25) | **−5.91** (+0.31) |
| 32 | −3.47  | −3.14 (−0.33) | −3.31 (−0.16) | −3.56 (+0.09) | −3.79 (+0.32) | **−3.83** (+0.36) |

**DCRNetV2-base/unified/large outdoor 4 cr 都击穿 DCRNet-V1.**
**DCRNetV2-mini/-small outdoor 略输 paper**（FLOPs 也更小, Pareto trade-off）。

---

## Key Innovations Summary

1. **Complex Hermitian Bilinear Encoder** (`ComplexLowRankEncoder`)
   - u^H · H · v with all 4 cross-terms preserved (re/re, re/im, im/re, im/im)
   - +2-3 dB over real bilinear baseline at cr=4

2. **Encoder Concat-MLP Fusion** (`EncoderFuseMLP`)
   - Replaces v8's plain sum z_a + z_b with concat + 2-layer MLP residual
   - +1-2 dB over naive sum

3. **Hybrid Rank Decoder** (`HybridRankDecoder`, v12 family)
   - Direct LowRankDecoder (R=32, anchor) + GatedFactoredResidualDecoder
     (R=16/32, cheap z-conditioned)
   - Breaks cr=8/16/32 outdoor stall (where direct-only loses to baselines)

4. **α / gate as scenario hyperparam**
   - Outdoor multipath-rich: α=0.05 + gate=−2 (aggressive extras)
   - Indoor cr=4 LOS-dominant: α=0.05 + gate=−4 (extras start closed)
   - Indoor cr=16 (post-train fine-tune): α=0.05 + gate=−1 reset
   - Unified safe: α=0.01 + gate=−2

5. **r_enc as FLOPs lever**
   - 128 (mini/small variants), 256 (small), 512 (base/unified/large)
   - Each step halves encoder cost, ~0.3-0.6 dB NMSE drop

---

## File Reference

- **Models**: `models/dcrnet_v9.py` (all v9 variants including hybrid)
- **Training**: `main.py` (CLI entry, all flags above documented)
- **Full experiment log**: `V9_EXPERIMENT_20260511.md` (2589 lines,
  timeline + ablations)

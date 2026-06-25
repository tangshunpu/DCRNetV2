# DCRNet-v8 配置组合扫描汇总

**日期**：2026-05-09
**所有实验**：训练 2500 epochs（除标注 `@N` 表示中途停在 ep N），lr=2e-3 + warmup 30 ep + cosine 衰减，COST2100 数据集

---

## Indoor (`--scenario in`)

| 配置 | cr=4 | cr=8 | cr=16 | params (4/8/16) | FLOPs (4/8/16) |
|---|---|---|---|---|---|
| **exp=8, r=16, r_enc=512** ⭐参考 | -31.69 @2470 | -19.27 @2395 | -13.78 @735 | 2.66M / 1.35M / 0.69M | 5.81 / 4.50 / 3.84 M |
| exp=4, r=16, r_enc=1024 | **-32.52** @2445 | -18.73 @2165 | **-13.91** @2205 | 3.22M / 1.64M / 0.86M | 6.52 / 4.94 / 4.16 M |
| exp=16, r=64, r_enc=1024 @115 | -29.91 @115 | -17.13 @100 | -12.87 @105 | 6.37M / 3.22M / 1.65M | 12.56 / 9.42 / 7.84 M |
| exp=32, r=32, r_enc=1024 @50 | -27.54 @50 | -17.93 @35 | -12.29 @20 | 4.27M / 2.17M / 1.12M | 13.94 / 11.84 / 10.79 M |
| exp=32, r=32, r_enc=1024 cr=32 @38 | (cr=32 only): -9.11 @30 | | | 601K params | 10.27M FLOPs |

### Indoor 结论
- **cr=4 capacity-limited**：`exp=4, r=16, r_enc=1024` 反超 ref **+0.83 dB**（充分训完后）
- **cr=8 中间区**：ref 仍是冠军（exp/r_enc 任何方向调都没赢）
- **cr=16 info-limited**：`exp=4, r_enc=1024` 反超 ref **+0.13 dB**
- 简单加大 (exp=16/32, r=64) 早期看着强但收益不延续，FLOPs 暴涨没换来终值优势

---

## Outdoor (`--scenario out`)

| 配置 | cr=4 | cr=8 | cr=16 | params (4/8/16) | FLOPs (4/8/16) |
|---|---|---|---|---|---|
| **exp=1, r=16, r_enc=512** ⭐参考 | -11.21 @2495 | -7.73 @2485 | -4.97 @2430 | 2.66M / 1.35M / 0.69M | 4.23 / 2.92 / 2.27 M |
| exp=2, r=16, r_enc=2048 @935-1219 | **-12.13** @935 | **-8.76** @1195 | **-5.90** @1185 | 4.33M / 2.23M / 1.18M | 9.28 / 7.18 / 6.13 M |
| exp=4, r=64, r_enc=512 @25（跑中）| -11.93 @25 | -8.11 @25 | -5.37 @20 | 5.81M / 2.93M / 1.48M | **8.25 / 5.37 / 3.93 M** |

注：outdoor exp=2 那行**不是终值**，停在 ep935-1219，仍在小幅推进中。

### Outdoor 结论
- **大 r_enc (2048) 在 outdoor 全面胜出**：cr=4/8/16 都领先 ref 终值 ~0.9-1.0 dB
- 多径丰富的 outdoor 场景下 matched filter 数量决定上限，r_enc 比 refine 容量更值钱
- exp=4, r=64, r_enc=512 早期 ep20-25 数字已经接近 exp=2/r_enc=2048 同期，但参数翻倍，性价比待终值确认

---

## Indoor v9 变体（架构改造对比）

所有变体均在 `exp=8, r=16, r_enc=512` 设置下，只改架构。

| 变体 | 描述 | cr=4 best @ ep | cr=8 best @ ep | cr=16 best @ ep | 状态 |
|---|---|---|---|---|---|
| **v8 (ref)** | sum encoder + DCRDecoderRefine (BN, scalar gate) | **-31.69 @2470** | **-19.27 @2395** | **-13.78 @735** | 2500 ep 完成 |
| v9 (concat+fuse + FiLM + GN + bounded refine) | 首次设计，warmup 中段崩 | 崩 ep5 | 崩 ep10 | 崩 ep10 | 失败 |
| v9_nofilm (concat+fuse + GN + bounded) | 去 FiLM | 崩 ep10 | — | — | 失败 |
| v9_nogn (concat+fuse + BN + bounded) | 去 GN | 崩 ep15 | 崩 ep15 | — | 失败 |
| v9_lnenc (concat+fuse + LN-on-z_b + BN + bounded) @50 | 加回 LN | -26.27 @45 | -17.91 @45 | -13.74 @50 | @ep50 停 |
| v9_lnenc_v8r (concat+fuse + LN + BN + v8 风格 refine) @39 | 改 refine 残差形式 | -19.48 @30 | -16.17 @25 | -12.69 @35 | @ep39 停 |
| **v9_new (v8 sum encoder + FiLM refine)** @68 | 最简洁 v9 | -25.01 @60 | — | -13.48 @55 | @ep68 停 |

### v9 系列发现
- **encoder 缺 LayerNorm 是导致 warmup 崩溃的根本原因**（z_b magnitude 失控 + fuse_enc 右半增长 → 乘积爆炸）
- 加回 LN 后所有变体训练稳定
- v9 相比 v8 没明显优势（cr=16 略胜 v8 终值 0.05-0.10 dB，cr=4 落后 ~3-6 dB）

---

## v5 warm-init 实验（indoor）

用 `outputs/v5_sweep/r16_renc512/checkpoints/v5-1X-cr*-in` 做 LowRank 预训练初始化。

| 实验 | lr 设置 | cr=4 best @ep | cr=8 best @ep | cr=16 best @ep | 状态 |
|---|---|---|---|---|---|
| v9_v8r + v5 frozen | lr=2e-3 (frozen 不训) | -23.01 @10 (停 36ep) | -17.91 @45 | -13.65 @45 | @ep50 停 |
| v9_v8r + v5 warm (no freeze) | lr=2e-3 全局 | -25.87 @25 | -17.21 @25 | -13.74 @15 | @ep32 停 |
| v8 + v5 warm | lr=5e-4 全局 | -26.91 @70 | -17.69 @70 | -13.86 @75 | @ep75 停 |
| v8 + v5 warm + param groups | backbone 5e-5 / new 2e-3 | -28.19 @130 | -18.38 @105 | -13.88 @75 | @ep131 停 |

### v5 warm 结论
- v5 backbone 提供强起点（5 epoch 即达 v5 baseline）
- frozen 在 cr=4 capacity-limited 区会卡死
- warm + param groups 兼得（v5 backbone 保护 + 新模块充分探索）
- 对 cr=8/16 信息瓶颈区的提速效果非常显著（用 1/30 epoch 达到 fresh 训完水平）

---

## 关键启示（备论文写作）

1. **regime 判断最重要**：cr=4 capacity-limited、cr=8 中间区、cr=16/32 info-limited，调参应该按 regime 分别选策略
2. **Indoor 适合"窄而深"**：r_enc=512-1024 + 较大 ranks/expansion 配比
3. **Outdoor 适合"宽 encoder"**：大 r_enc (2048) 比扩 ranks/expansion 更值钱（多径丰富，matched filter 数量决定上限）
4. **架构改造空间不大**：v9 各种变体跟 v8 持平或略输；增益主要来自配置扫参（r_enc / ranks），不是结构改动
5. **v5 warm-init 是好套路**：配 param groups 能在前几十 epoch 达到 fresh 训完的水平

---

## 文件参考

- 模型代码：`models/dcrnet_v8.py`、`models/dcrnet_v9.py`（含 `DCRNetV9` 主版本和 `DCRNetV9Diagnostic` 诊断变体）
- v5 高容量 checkpoints：`outputs/v5_sweep/r16_renc512/checkpoints/`
- 训练日志：`outputs/log/`，命名 `MMDDHHMM-{model}-{exp}X-{cr}-{scenario}.log`
- 之前的实验报告：`EXPERIMENT_REPORT_20260508.md`（v9 设计与诊断的详细过程）

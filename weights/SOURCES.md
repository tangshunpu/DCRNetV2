# Weight source mapping

The public checkpoint filenames are normalized to DCRNetV2/LRP names. Some DCRNetV2 variants were trained under older experiment tags; the curated files in this directory were rebuilt from those local checkpoints and strict-load against the release models.

## DCRNetV2

| Public variant | Scenario(s) | Public files | Historical source tag |
|---|---|---|---|
| `dcrnetv2-mini` | `in`, `out` | `weights/dcrnetv2-mini/{in,out}/cr*.pt` | `v11cmh10e8` |
| `dcrnetv2-small` | `in`, `out` | `weights/dcrnetv2-small/{in,out}/cr*.pt` | `v11ctiny` |
| `dcrnetv2-base` | `in`, `out` | `weights/dcrnetv2-base/{in,out}/cr*.pt` | `v11c` |
| `dcrnetv2-unified` | `in`, `out` | `weights/dcrnetv2-unified/{in,out}/cr*.pt` | `v12r` |
| `dcrnetv2-large-out` | `out` | `weights/dcrnetv2-large-out/out/cr*.pt` | `v12rX` |
| `dcrnetv2-large-in4` | `in` | `weights/dcrnetv2-large-in4/in/cr4.pt` | `v12rXg` |
| `dcrnetv2-large-in4` | `in` | `weights/dcrnetv2-large-in4/in/cr{8,16,32}.pt` | `v12rXgFTC` |

During conversion, historical `enc_dilate.block.conv2.0.*` keys are normalized to release `enc_dilate.block.conv2.*` keys and `thop` bookkeeping keys are removed.

## Sionna-RT-Mix5

| Public files | Historical source tag | Training data |
|---|---|---|
| `weights/rtcsi-mix5/dcrnetv2-unified/cr{4,8,16}.pt` | `combined5_v9u_cr{4,8,16}` | Five-scene Mix5 training split |

The three published RT-CSI files contain only the current model state and small provenance metadata. Early encoder activations were saved as PReLU parameters with slope 0.3, preserving the fixed-slope inference behavior of the source weights. The files strict-load with `models/dcrnetv2.py`.

## LRP

| Public variant | Scenario(s) | Public files | Historical source directory |
|---|---|---|---|
| `lrp-r4-d128` | `in`, `out` | `weights/lrp-r4-d128/{in,out}/cr*.pt` | `outputs/v5_sweep/r4_renc128/` |
| `lrp-r4-d512` | `in`, `out` | `weights/lrp-r4-d512/{in,out}/cr*.pt` | `outputs/v5_sweep/r4_renc512/` |
| `lrp-r8-d512` | `in`, `out` | `weights/lrp-r8-d512/{in,out}/cr*.pt` | `outputs/v5_sweep/r8_renc512/` |
| `lrp-r16-d512` | `in`, `out` | `weights/lrp-r16-d512/{in,out}/cr*.pt` | `outputs/v5_sweep/r16_renc512/` |

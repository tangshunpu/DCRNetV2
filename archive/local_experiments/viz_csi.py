"""Visualize COST2100 CSI samples with a mostly-white color scheme."""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

mat_in = sio.loadmat('COST2100/DATA_Htestin.mat')['HT']
mat_out = sio.loadmat('COST2100/DATA_Htestout.mat')['HT']

def to_complex_img(row):
    x = row.reshape(2, 32, 32)
    return x[0] - 0.5 + 1j * (x[1] - 0.5)

rng = np.random.default_rng(7)
idx_in = rng.choice(mat_in.shape[0], 4, replace=False)
idx_out = rng.choice(mat_out.shape[0], 4, replace=False)

samples = [(mat_in[i], 'indoor', i) for i in idx_in] + \
          [(mat_out[i], 'outdoor', i) for i in idx_out]

# White-dominant colormap: pure white background -> soft blue -> deep navy
white_blue = LinearSegmentedColormap.from_list(
    'white_blue',
    [(1.0, 1.0, 1.0), (0.85, 0.92, 0.97), (0.45, 0.66, 0.85), (0.10, 0.25, 0.50)],
    N=256,
)

fig, axes = plt.subplots(2, 4, figsize=(13, 6.8), facecolor='white')
fig.suptitle('COST2100 CSI · angular-delay magnitude',
             fontsize=14, color='#333', y=0.98)

for ax, (row, scenario, idx) in zip(axes.flat, samples):
    H = to_complex_img(row)
    mag = np.abs(H)
    # robust normalization so the tail stays near-white
    vmax = np.quantile(mag, 0.995) + 1e-12
    im = ax.imshow(mag, cmap=white_blue, vmin=0, vmax=vmax,
                   interpolation='nearest', aspect='equal')
    ax.set_title(f'{scenario} #{idx}', fontsize=10, color='#444', pad=4)
    ax.set_xticks([0, 16, 31]); ax.set_yticks([0, 16, 31])
    ax.tick_params(colors='#888', labelsize=8, length=2)
    for s in ax.spines.values():
        s.set_color('#ddd'); s.set_linewidth(0.6)

cbar_ax = fig.add_axes([0.92, 0.12, 0.012, 0.76])
cb = fig.colorbar(im, cax=cbar_ax)
cb.outline.set_edgecolor('#ddd'); cb.outline.set_linewidth(0.6)
cb.ax.tick_params(colors='#888', labelsize=8, length=2)
cb.set_label('|H|', color='#555', fontsize=9)

fig.text(0.5, 0.02,
         'rows: delay τ (0–31)   ·   cols: angle θ (0–31)   ·   '
         f'in={mat_in.shape[0]} samples, out={mat_out.shape[0]} samples',
         ha='center', color='#666', fontsize=9)

plt.subplots_adjust(left=0.04, right=0.90, top=0.91, bottom=0.08,
                    wspace=0.18, hspace=0.28)
out = 'outputs/csi_samples_white.png'
plt.savefig(out, dpi=160, facecolor='white')
print(f'saved -> {out}')

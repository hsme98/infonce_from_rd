from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METRICS = ROOT / 'results' / 'paper' / 'markov' / 'metrics.json'
REPRESENTATIVE = ROOT / 'results' / 'paper' / 'markov' / 'representative_seed003_arrays.npz'
OUT = ROOT / 'assets'
OUT.mkdir(parents=True, exist_ok=True)

with METRICS.open('r', encoding='utf-8') as handle:
    data = json.load(handle)
curves = data['aggregate']['curves']
rep = np.load(REPRESENTATIVE)
true_transition = np.asarray(rep['transition'], dtype=float)
recovered_transition = np.asarray(rep['recovered_channel'], dtype=float)

plt.rcParams.update({
    'font.size': 7.1,
    'axes.titlesize': 7.9,
    'axes.labelsize': 7.2,
    'legend.fontsize': 6.0,
    'xtick.labelsize': 6.3,
    'ytick.labelsize': 6.3,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Fixed-position layout: the left/right plotting boxes are identical in both rows.
# This prevents the shared colorbar and the square heatmap aspect from shrinking or
# shifting only the top row.
fig = plt.figure(figsize=(7.15, 4.90))
left_x, right_x, panel_w = 0.095, 0.555, 0.330
heat_y, heat_h = 0.490, 0.482
plot_y, plot_h = 0.075, 0.292

ax_true = fig.add_axes([left_x, heat_y, panel_w, heat_h])
ax_rec = fig.add_axes([right_x, heat_y, panel_w, heat_h])
ax_hier = fig.add_axes([left_x, plot_y, panel_w, plot_h])
ax_base = fig.add_axes([right_x, plot_y, panel_w, plot_h])
cax = fig.add_axes([0.910, heat_y, 0.014, heat_h])

vmin = 0.0
vmax = float(max(true_transition.max(), recovered_transition.max()))
for ax, matrix, title in [
    (ax_true, true_transition, '(a) True transition channel'),
    (ax_rec, recovered_transition, '(b) InfoNCE recovery'),
]:
    im = ax.imshow(matrix, origin='upper', aspect='equal', vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_title(title, pad=2.0)
    ax.set_xlabel('next state', labelpad=1.5)
    ticks = [0, 16, 32, 48, 63]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    for boundary in range(4, 64, 4):
        width = 1.0 if boundary % 16 == 0 else 0.32
        alpha = 0.92 if boundary % 16 == 0 else 0.45
        ax.axhline(boundary - 0.5, color='white', linewidth=width, alpha=alpha)
        ax.axvline(boundary - 0.5, color='white', linewidth=width, alpha=alpha)
ax_true.set_ylabel('current state', labelpad=1.8)
ax_rec.set_yticklabels([])
ax_rec.set_ylabel('')
cb = fig.colorbar(im, cax=cax, orientation='vertical')
cb.set_label(r'$W(y\mid x)$', labelpad=1.5)
cb.ax.tick_params(pad=1.0)

# Panel (c): learned coarse-to-fine hierarchy.
info = curves['InfoNCE']
rate = np.asarray(info['rate_grid'], dtype=float)
series = [
    ('macro_fraction', 'macro-community'),
    ('micro_increment', 'micro beyond macro'),
    ('state_increment', 'state beyond micro'),
]
for metric, label in series:
    stats = info[metric]
    median = np.asarray(stats['median'], dtype=float)
    q10 = np.asarray(stats['q10'], dtype=float)
    q90 = np.asarray(stats['q90'], dtype=float)
    line = ax_hier.plot(rate, median, linewidth=1.35, label=label)[0]
    ax_hier.fill_between(rate, q10, q90, alpha=0.10)
ax_hier.set_xlim(0.0, 4.15)
ax_hier.set_ylim(0.0, 1.02)
ax_hier.set_xlabel(r'rate $I(X;\widetilde X)$ [nats]', labelpad=1.6)
ax_hier.set_ylabel('normalized information retained', labelpad=2.0)
ax_hier.set_title('(c) InfoNCE-induced hierarchy', pad=2.0)
ax_hier.grid(True, alpha=0.22)
ax_hier.legend(frameon=False, loc='lower right', handlelength=2.0)

# Panel (d): macro-community efficiency against unrelated geometries.
line_styles = {
    'oracle': '--',
    'InfoNCE': '-',
    'Hamming': '-.',
    'random sphere': ':',
}
for method in ['oracle', 'InfoNCE', 'Hamming', 'random sphere']:
    d = curves[method]
    x = np.asarray(d['rate_grid'], dtype=float)
    y = np.asarray(d['macro_fraction']['median'], dtype=float)
    ax_base.plot(x, y, linewidth=1.35, linestyle=line_styles[method], label=method)
ax_base.axvline(np.log(4.0), linewidth=0.72, linestyle=':')
ax_base.text(np.log(4.0) + 0.035, 0.028, r'$\log 4$', fontsize=6.0, rotation=90, va='bottom')
ax_base.set_xlim(0.0, 4.15)
ax_base.set_ylim(0.0, 1.02)
ax_base.set_xlabel(r'rate $I(X;\widetilde X)$ [nats]', labelpad=1.6)
ax_base.set_ylabel('macro information retained', labelpad=2.0)
ax_base.set_title('(d) Macro-information efficiency', pad=2.0)
ax_base.grid(True, alpha=0.22)
ax_base.legend(frameon=False, loc='lower right', handlelength=2.0)

for path in [OUT / 'markov_full.pdf', OUT / 'markov_full.png']:
    if path.suffix == '.pdf':
        fig.savefig(path)
    else:
        fig.savefig(path, dpi=300)
plt.close(fig)

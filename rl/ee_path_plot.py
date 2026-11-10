"""(11-09) Figure of ee_path.py traces (OUT.npz): x y z and speed of both wrists, reference dashed.  python3 ee_path_plot.py OUT.npz OUT.png"""
import sys
import numpy as np


def draw(npz, out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    z = np.load(npz); labs = list(z['labels']); FS = 50.; pal = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
    fig, ax = plt.subplots(4, 2, figsize=(15, 11), sharex=True)
    for j, s in enumerate(('right', 'left')):
        for i, lab in enumerate(labs):
            L, sp = z[f'{s}|{lab}|pos'], z[f'{s}|{lab}|speed']; ref = lab == 'ref'; tt = np.arange(len(L))/FS
            kw = dict(color='k' if ref else pal[(i-1) % len(pal)], lw=2.2 if ref else 1.1, ls='--' if ref else '-', label='reference (re-targeted demo)' if ref else lab)
            for k in range(3): ax[k, j].plot(tt, L[:, k]*1e3, **kw)
            ax[3, j].plot(tt, sp*100, **kw)
        for k, nm in enumerate('xyz'): ax[k, j].set_ylabel(f'{nm} (mm)')
        ax[3, j].set_ylabel('speed (cm/s)'); ax[3, j].set_xlabel('t (s)'); ax[0, j].set_title(f'{s} wrist, world {int(z["world"])} (low-passed {float(z["lp"]):g} Hz)')
    ax[0, 0].legend(fontsize=8); fig.tight_layout(); fig.savefig(out, dpi=110); print('plot', out)


if __name__ == '__main__': draw(sys.argv[1], sys.argv[2])

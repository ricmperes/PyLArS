import numpy as np
import matplotlib.pyplot as plt


def plot_waveform(waveform_array, full_y=False, full_x=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    n_samples = len(waveform_array)
    x = np.arange(0, n_samples)
    ax.plot(x, waveform_array)
    baseline_rough = np.median(waveform_array[:50])
    std_rough = np.std(waveform_array[:50])
    ax.axhline(baseline_rough, ls='--', c='gray')
    ax.axhline(baseline_rough - std_rough, ls='--', c='orange')
    ax.axhline(baseline_rough + std_rough, ls='--', c='orange')
    ax.axhline(baseline_rough - std_rough * 5, ls='--', c='red')
    ax.axhline(baseline_rough + std_rough * 5, ls='--', c='red')
    if full_x != True:
        ax.set_xlim(full_x)
    if full_y:
        ax.set_ylim(0, 2**14)
    else:
        ax.set_ylim(
            min(waveform_array) -
            std_rough *
            6,
            max(waveform_array) +
            std_rough *
            6)

    return ax

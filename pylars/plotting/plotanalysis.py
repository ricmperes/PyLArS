import matplotlib.pyplot as plt
import numpy as np
from pylars.utils.common import Gaussean, func_linear
from matplotlib.figure import Figure as pltFigure
from .plotprocessed import *

# LED ON


def plot_area_LED(bv_dataset, voltage, LED_position=300,
                  log_y=True, full_x=False, ax=None,
                  color=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    df = bv_dataset.data[voltage]

    cut_mask = ((df['position'] > (LED_position - 10)) &
                (df['position'] < (LED_position + 20)) &
                (df['length'] > 3))

    ax = plot_hist_area(df[cut_mask], ax=ax, color=color)

    if full_x:
        ax.set_xlim(0, 2**14 * 10 * 300)
    if log_y:
        ax.set_yscale('log')

    ax.set_title((f'LED ON\n module {bv_dataset.module} | '
                  f'channel {bv_dataset.channel[-1]}')
                 )

    med = np.median(df[cut_mask]['area'])
    std = np.std(df[cut_mask]['area'])
    med_err = std / np.sqrt(len(df[cut_mask]))

    return med, med_err, ax


def plot_LED_all_voltages(bv_dataset, cmap='winter', ax=None,):
    cm = plt.get_cmap(cmap)  # type: ignore
    N_lines = len(bv_dataset.voltages)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for i, _v in enumerate(bv_dataset.voltages):

        med, med_err, ax = plot_area_LED(bv_dataset, 50,
                                         color=cm(i / N_lines),
                                         ax=ax)
        ax.axvline(med, color=cm(i / N_lines))
        ax.set_title('')

    plt.show()


# LED OFF


def plot_DCR_curve(plot, area_hist_x, DCR_values, _x, _y, min_area_x, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5), facecolor='white')
    ax.plot(area_hist_x, DCR_values,
            marker='x', ls='',
            c='k', label='Data points')
    ax.set_yscale('log')
    ax.set_xlabel('Area [integrated ADC counts]')
    ax.set_ylabel('# events')

    ax3 = ax.twinx()
    ax3.plot(_x, _y, c='r')
    ax3.tick_params(axis='y', labelcolor='r')
    ax3.axvline(min_area_x, c='r', ls='--', alpha=0.8,
                label='1$^{st}$ der. (smoothed)')
    ax3.set_ylabel('1$^{st}$ derivative')

    if 'fig' in locals():
        fig.legend()  # type: ignore

    if isinstance(plot, str):
        fig.savefig(f'figures/{plot}_stepplot.png')  # type: ignore
        plt.close()
    else:
        return ax


def plot_SPE_fit(df, length_cut, plot, area_hist_x,
                 min_area_x, A, mu, sigma, ax=None):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5), facecolor='white')
    bin_size = area_hist_x[1] - area_hist_x[0]
    ax.hist(df[df['length'] > length_cut]['area'],
            bins=np.linspace(min_area_x - 2000, min_area_x + 2000, 300),
            color='gray', alpha=0.8)
    _x = np.linspace(area_hist_x[0], area_hist_x[-1], 200)
    ax.plot(_x, Gaussean(_x, A, mu, sigma), color='red')
    ax.set_xlabel('Area [integrated ADC counts]')
    ax.set_ylabel('# events')
    for i in range(1, 4):
        ax.axvline(mu * i, color='red', ls='--', alpha=0.7)
        # plt.yscale('log')

    if isinstance(plot, str):
        plt.savefig(f'figures/{plot}_1pe_fit.png')
        plt.close()
    else:
        return ax


def plot_BV_fit(plot, temperature, voltages, gains,
                a, b, _breakdown_v, ax=None):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3), facecolor='white')

    ax.plot(gains, voltages,
            ls='', marker='x', c='k',
            label=f'{temperature}K: {_breakdown_v:.2f} V')
    _x = np.linspace(min(gains), max(gains), 100)
    ax.plot(_x, func_linear(_x, a, b), c='r', alpha=0.9)
    ax.set_xlabel('Area')
    ax.set_ylabel('V')
    ax.legend()
    ax.set_title(plot)

    if isinstance(plot, str):
        plt.tight_layout()
        plt.savefig(f'figures/{plot}_{temperature}_BV_fit.pdf')
        plt.close()
    else:
        return ax


def plot_peaks(area_x, area_y, area_filt, area_peaks_x,
               plot=None, ax=None):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5), facecolor='white')

    ax.plot(area_x, area_y, color='k', alpha=0.7)  # , range = (100,1e4))

    ax.plot(area_x, area_filt, color='blue', ls='-', lw=1, alpha=1)

    ax.vlines(area_x[area_peaks_x], 0, 1e6, color='green', alpha=0.5)
    ax.set_xlabel('Area')
    ax.set_yscale('log')

    if isinstance(plot, str):
        plt.savefig(f'figures/{plot}_paeks_and_valeys.png')
        plt.close()
    elif plot is True:
        plt.show()

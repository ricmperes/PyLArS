import matplotlib.pyplot as plt
import numpy as np
from pylars.utils.common import Gaussean, func_linear

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
    cm = plt.get_cmap(cmap)
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
    fig.legend()

    if isinstance(plot, str):
        fig.savefig(f'figures/{plot}_stepplot.png')
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
        fig, ax = plt.subplots(1, 1, figsize=(12, 5), facecolor='white')

    ax.plot(voltages, gains,
            ls='', marker='x', c='k',
            label=f'{temperature}K: {_breakdown_v:.2f} V')
    _x = np.linspace(min(voltages), max(voltages), 100)
    ax.plot(_x, func_linear(_x, a, b), c='r', alpha=0.9)
    ax.set_ylabel('Gain')
    ax.set_xlabel('V')
    ax.legend()

    if isinstance(plot, str):
        plt.savefig(f'figures/{plot}_{temperature}_BV_fit.png')
        plt.close()
    else:
        return ax

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylars


def plot_rates()


fig, ax = plt.subplots(
    1, 1, figsize=(
        6, 4), facecolor='white', gridspec_kw={
        'hspace': 0, 'wspace': 0})

ax.bar(
    np.arange(25),
    200 /
    result_df['livetime'].dt.total_seconds(),
    alpha=1,
    label='Double coin (2xPMT)')
ax.bar(
    np.arange(25),
    result_df['rate'],
    alpha=1,
    label='Triple coin (2xPMT + SiPM)')
ax.set_ylabel('Rate [Hz]')
ax.set_xlabel('Run #')
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax.set_yscale('log')
ax.legend(loc='lower left', bbox_to_anchor=(0, 1, 1, 0.5))

#ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
ax_histy = ax.inset_axes([1., 0, 0.25, 1], sharey=ax)
ax_histy.hist(200 / result_df['livetime'].dt.total_seconds(), bins=np.logspace(-5, -
                                                                               2, 20), alpha=1, histtype='step', color='C0', orientation='horizontal')
ax_histy.hist(result_df['rate'], bins=np.logspace(-5, -2, 20),
              alpha=1, histtype='step', color='C1', orientation='horizontal')

med_2coin = np.median(200 / result_df['livetime'].dt.total_seconds())
med_3coin = np.median(result_df['rate'])
ax_histy.axhline(med_2coin, ls='--', color='C0',
                 label=f'2coin median: {med_2coin:.2e} Hz')
ax_histy.axhline(np.median(result_df['rate']), ls='--', color='C1',
                 label=f'3coin median: {med_3coin:.2e} Hz')
#ax_histx.tick_params(axis="x", labelbottom=False)
ax_histy.tick_params(axis="y", labelleft=False)
ax_histy.set_xlabel('Rate [Hz]')
ax_histy.legend(loc='lower right', bbox_to_anchor=(0, 1, 1, 0.5))
# plt.subplot(122)
# plt.hist(result_df['rate'], bins = 10, alpha = 1, histtype='step', color = 'C1')
# plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
# plt.xlabel('Rate [Hz]')
plt.subplots_adjust(
    left=None,
    bottom=None,
    right=None,
    top=None,
    wspace=0,
    hspace=0)
plt.savefig(
    'coin_rates.jpeg',
    dpi=120,
    bbox_inches='tight',
    pad_inches=0.1)
plt.show()


def plot_baseline_all_runs():
    fig, ax = plt.subplots(4, 3, figsize=(20, 10), sharey=True, sharex=True,
                           gridspec_kw={'hspace': 0, 'wspace': 0},
                           constrained_layout=False)
    ax = ax.flatten()
    ax[0].set_ylim(14750, 15300)
    for i, ch in enumerate(['wf1', 'wf2', 'wf3', 'wf4', 'wf5']):
        ax[i].errorbar(np.arange(25), baseline_all_runs[f'0_{ch}'],
                       yerr=std_all_runs[f'0_{ch}'],
                       ls='-', marker='o', capsize=2, label=labels['mod0'][ch])
        ax[i].text(0.05, 0.95, s=f'Mod 0 | ' + labels[f'mod0'][ch],
                   transform=ax[i].transAxes, va='top', ha='left',
                   bbox=dict(facecolor='white', alpha=0.5))

    for j, ch in enumerate(['wf1', 'wf2', 'wf3', 'wf4', 'wf5', 'wf6', 'wf7']):
        ax[i + j + 1].errorbar(np.arange(25), baseline_all_runs[f'1_{ch}'],
                               yerr=std_all_runs[f'1_{ch}'],
                               ls='-', marker='o', capsize=2, label=labels['mod1'][ch])
        ax[i + j + 1].text(0.05, 0.95, s=f'Mod 1 | ' + labels[f'mod1'][ch],
                           transform=ax[i + j + 1].transAxes, va='top', ha='left',
                           bbox=dict(facecolor='white', alpha=0.5))

    plt.subplots_adjust(
        left=None,
        bottom=None,
        right=None,
        top=None,
        wspace=0,
        hspace=0)

    # Put x and y label in the center
    big_ax = fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big subplot
    big_ax.tick_params(
        labelcolor='none',
        top=False,
        bottom=False,
        left=False,
        right=False)

    big_ax.set_xlabel(
        'Number of dataset',
        labelpad=10,
    )  # set the common x label
    big_ax.set_ylabel(
        'Baseline [ADCcounts]',
        labelpad=25)  # set the common y label

    plt.savefig(
        'baselines_all_runs.jpeg',
        dpi=120,
        bbox_inches='tight',
        pad_inches=0.1)
    plt.show()


def plot_baseline_channel(data_dict, mod, ch, figax=None):
    if figax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    else:
        fig, ax = figax
    baselines, std = get_baseline_channel(data_dict, mod, ch)

    ax.errorbar(np.arange(200), baselines, yerr=std, ls='',
                marker='o', capsize=2)
    # ax.set_xlabel('Waveform #')
    #ax.set_ylabel('Baseline [ADC]')
    ax.text(0.05, 0.95, s=f'Mod {mod} | ' + labels[f'mod{mod}'][ch],
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.5))
    if figax is None:
        plt.show()
    else:
        return fig, ax


def plot_baseline_all_channels(data_dict, figax=None):
    if figax is None:
        fig, axs = plt.subplots(3, 4, figsize=(20, 10),
                                sharey=True, sharex=True,
                                gridspec_kw={'hspace': 0, 'wspace': 0},
                                constrained_layout=False)
        axs = axs.flatten()
    else:
        fig, axs = figax

    for i, ch in enumerate(['wf1', 'wf2', 'wf3', 'wf4', 'wf5']):
        fig, axs[i] = plot_baseline_channel(data_dict, 0, ch,
                                            figax=(fig, axs[i]))  # type: ignore

    for j, ch in enumerate(['wf1', 'wf2', 'wf3', 'wf4', 'wf5', 'wf6', 'wf7']):
        fig, axs[i + j + 1] = plot_baseline_channel(data_dict, 1, ch,
                                                    figax=(fig, axs[i + j + 1]))  # type: ignore
    axs[0].set_ylim(14750, 15300)
    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=0, hspace=0)

    # Put x and y label in the center
    big_ax = fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big subplot
    big_ax.tick_params(
        labelcolor='none',
        top=False,
        bottom=False,
        left=False,
        right=False)

    big_ax.set_xlabel(
        'Waveform number',
        labelpad=10,
    )  # set the common x label
    big_ax.set_ylabel(
        'Baseline [ADCcounts]',
        labelpad=25)  # set the common y label

    if figax is None:
        plt.savefig(
            'baselines_run001.jpeg',
            dpi=120,
            bbox_inches='tight',
            pad_inches=0.1)
        plt.show()
    else:
        return fig, axs

    def plot_mu_waveform_array(data_dict, n_wf, plot=True, save_fig=False):
        labels = {'mod0': {'wf1': 'wf1 | Tile H', 'wf2': 'wf2 | Tile J',
                           'wf3': 'wf3 | Tile K', 'wf4': 'wf4 | Tile L',
                           'wf5': 'wf5 | Tile M',
                           'wf6': 'wf6 | Muon detector 1',
                           'wf7': 'wf7 | Muon detector 2'},
                  'mod1': {'wf1': 'wf1 | Tile A', 'wf2': ' wf2 | Tile B',
                           'wf3': 'wf3 | Tile C', 'wf4': 'wf4 | Tile D',
                           'wf5': 'wf5 | Tile E', 'wf6': 'wf6 | Tile F',
                           'wf7': 'wf7 | Tile G'}
                  }

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
        _x = np.arange(process.raw_data.n_samples)
        plt.subplots_adjust(hspace=0)

        for _ch in ['wf6', 'wf7']:
            axs[0].plot(_x, data_dict['mod0'][_ch][n_wf],
                        label=labels['mod0'][_ch])
        # axs[0].legend()

        for _ch in ['wf1', 'wf2', 'wf3', 'wf4', 'wf5', ]:
            axs[2].plot(_x, data_dict['mod0'][_ch][n_wf],
                        label=labels['mod0'][_ch])
        # axs[1].legend()

        for _ch in ['wf1', 'wf2', 'wf3', 'wf4', 'wf5', 'wf6', 'wf7']:
            axs[1].plot(_x, data_dict['mod1'][_ch][n_wf],
                        label=labels['mod1'][_ch])
        # axs[2].legend()

        fig.suptitle(f'Evt # {n_wf}', y=1.5, x=0.5, horizontalalignment='center',
                     verticalalignment='center', transform=axs[0].transAxes)
        axs[1].set_xlabel('Sample #')
        #axs[0].set_xlim(400, 600)
        #axs[1].set_xlim(400, 600)
        fig.legend(ncol=5, loc='lower center',
                   bbox_to_anchor=(0, 0.9, 1, 0))
        if save_fig != False:
            if isinstance(save_fig, str):
                os.makedirs(f'figures/{save_fig}', exist_ok=True)
                plt.savefig(
                    f'figures/{save_fig}/{save_fig}_{n_wf}.png', dpi=80)
            else:
                plt.savefig(f'figures/{n_wf}_{int(time.time())}.png', dpi=80)

        if plot:
            plt.show()
        plt.close()

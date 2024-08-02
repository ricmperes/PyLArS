from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylars

def plot_LED_window(waveform: np.ndarray, led_window, figax = None):
    """Plot the LED window on top of the waveform.
    """
    if figax is None:
        fig, ax = plt.subplots(1,1,figsize = (6,4), dpi = 60)
    else:
        fig, ax = figax
    
    ax = pylars.plotting.plot_waveform(waveform, ax = ax)
    ax.fill_betweenx([0,16000], led_window[0],led_window[1], 
                     alpha = 0.1, color = 'C1')
    
    return fig, ax


def plot_LED_window_from_file(filename: str, 
                              led_window: Tuple[int, int],
                              module: Optional[int] = None,
                              channel: Optional[str] = None,
                              wf_number: Optional[int] = None,
                              figax: Optional[Tuple] = None):
    """Plot the LED window on top of the waveform directly from a file.
    """

    process = pylars.processing.fixwindowprocessor.window_processor(
    baseline_samples=50, led_window=(105,155))
    if module is None:
        module = 0

    print(f"Loading raw data from {filename}")
    process.load_raw_data(path_to_raw=filename, module=module)

    if channel is None:
        channel = str(np.random.choice(process.raw_data.channels))
        print(f"No channel specified, choosing random channel: {channel}")

    if wf_number is None:
        wf_number = int(np.random.choice(process.raw_data.n_waveforms))
        print(f"No waveform number specified, choosing random waveform:"
              f"{wf_number}")
        
    channel_data = process.raw_data.get_channel_data(channel)

    if figax is None:
        fig, ax = plt.subplots(1,1,figsize = (6,4), dpi = 60)
    else:
        fig, ax = figax
    
    fig, ax = plot_LED_window(channel_data[wf_number], 
                              led_window=led_window, 
                              figax = (fig, ax))

    if figax is None:
        plt.show()
    else:
        return fig, ax
    

def plot_LED_all_channels(df_processed, figax = None):
    """Plot max ADC counts, amplitude and area for all channels.
    """
    if figax is None:
        fig, axs = plt.subplots(1,3,figsize=(13,4), sharey=True,dpi = 60)
    else:
        fig, axs = figax

    plt.subplots_adjust(wspace=0)
    for ch in np.unique(df_processed['channel']):
        axs[0].hist(df_processed[df_processed['channel'] == ch]['led_ADCcounts'], 
                bins=100, histtype='step',label = f'{ch}')
    
        axs[1].hist(df_processed[df_processed['channel'] == ch]['led_amplitude'], 
                bins=100, histtype='step',)
    
        axs[2].hist(df_processed[df_processed['channel'] == ch]['led_area'], 
                bins=100, histtype='step',)

    axs[0].set_xlabel('max ADC counts')
    axs[0].set_ylabel('Counts')
    axs[1].set_xlabel('Amplitude [ADC counts]')
    axs[2].set_xlabel('Area [integrated yADC counts]')

    fig.legend(ncol=6, loc='upper center', bbox_to_anchor=(0.5, 1.05))

    [_ax.grid(color='lightgray', linestyle='--', linewidth=0.5) for _ax in axs]

    if figax is None:
        plt.show()
    else:
        return fig, axs
    

def plot_light_levels(led_processed_df: pd.DataFrame,
                      led_width: int,
                      channel: Optional[str] = None,
                      module: Optional[int] = None,
                      figax: Optional[Tuple] = None):

    select_mask = ((led_processed_df['LEDwidth'] == led_width) &
                     (led_processed_df['channel'] == channel) &
                     (led_processed_df['module'] == module))
    _df = led_processed_df[select_mask]

    if figax is None:
        fig, axs = plt.subplots(2,1,figsize = (6,8), dpi = 60)
    else:
        fig, axs = figax

    _ledvoltages = np.unique(_df['LEDvoltage'])
    _median_amplitudes = []
    _std_amplitudes = []
    for _v in _ledvoltages:
        _mask = _df['LEDvoltage'] == _v
        axs[0].hist(_df[_mask]['led_area'], 
                bins=np.linspace(-1000, 25000,500), histtype='step', label = f'{_v} V')
        _amp = np.median(_df[_mask]['led_area'])
        _std = np.std(_df[_mask]['led_area'])
        _median_amplitudes.append(_amp)
        _std_amplitudes.append(_std)
    axs[0].set_yscale('log')
    
    axs[1].errorbar(_ledvoltages, _median_amplitudes,
                    yerr = _std_amplitudes, marker = 'o',
                    ls = '-')
    axs[1].set_xlabel('LED voltage [V]')
    axs[1].set_ylabel('Median Area [ADC counts]')
    axs[1].grid(color='lightgray', linestyle='--', linewidth=0.5)

    axs[0].set_xlabel('Area [ADC counts]')
    axs[0].set_ylabel('Counts')
    axs[0].legend(loc = 'center left', bbox_to_anchor=(1., 0.5), 
              title = 'LED voltage')
    axs[0].grid(color='lightgray', linestyle='--', linewidth=0.5)

    if figax is None:
        plt.show()
    else:
        return fig, axs


def plot_1_pe_fit_led(df_processed, led_voltage, module, channel, A, mu, sigma, figax = None):
        if figax == None:
            fig, ax = plt.subplots(1,1, figsize = (5,3), dpi = 120)   
        else:
            fig, ax = figax

        df_processed_mask = ((df_processed['module'] == module) & 
                             (df_processed['channel'] == channel) &
                             (df_processed['LEDvoltage'] == led_voltage))
        hist = ax.hist(df_processed[df_processed_mask]['led_area'], 
                       bins = np.linspace(-2000, 20000,300), histtype = 'step')

        _x = np.linspace(mu*0.5, mu*1.5, 1000)
        ax.fill_between(_x, pylars.utils.common.Gaussian(_x, A, mu, sigma),
                color = 'C1', alpha = 1, zorder = -1)#, ls = '--', linewidth = '2')
        [ax.axvline(mu*n, color = 'C1', ls = '--', alpha = 1) for n in range(5)]

        #ax.set_yscale('log')
        #ax.set_ylabel('Counts')
        #ax.set_xlabel('Area (window)')
        ax.set_ylim(0,6000)
        if figax == None:
            plt.show()
        else:
            return fig, ax
        
def plot_gains_occ(df_gains, figaxs = None):
    if figaxs == None:
        fig, axs = plt.subplots(2,1, figsize = (4,4), 
                        dpi = 120, sharex = True,
                        gridspec_kw = {'hspace':0, 'wspace':0},
                        constrained_layout = False)
        axs = axs.flatten()

    _x = np.arange(len(df_gains))
    axs[0].errorbar(_x, df_gains['gain'], yerr = df_gains['gain_err'],
                    ls = '', capsize = 4, marker = '.')

    axs[1].errorbar(_x, df_gains['occ'], yerr = df_gains['occ_err'],
                    ls = '', capsize = 4, marker = '.')


    axs[0].set_xticks(_x, df_gains['tile'])
    #axs[0].set_ylim(0,1.5)
    axs[0].set_ylabel('Gain [$10^6$]')
    axs[1].set_ylabel('Occupancy')
    #axs[1].set_ylim(-0.1,3.9)
    if figaxs == None:
        plt.show()

    else:
        return (fig, axs)
    

def plot_gain_evolution(gain_evolution: pd.DataFrame, 
                        tile_list: list, mod, 
                        figax = None):
    
    if figax == None:
        fig, ax = plt.subplots(1,1, figsize=(6,4), dpi = 120)
    else:
        fig, ax = figax

    for _tile in tile_list:
        _df = gain_evolution[gain_evolution['tile'] == _tile]
        ax.errorbar(_df['start'], _df['gain'], yerr = _df['gain_err'], 
                     ls='--', 
                    marker = '.', label=_tile)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
            title='Tile', 
            labelspacing=1.0)
    ax.set_ylim(0,None) # type:ignore
    ax.set_title(f'Gain evolution - ADC #{mod}')
    ax.tick_params(axis = 'x', rotation=45)
    ax.set_ylabel('SiPM Gain [M]')
    # plt.twinx()
    # plt.plot(tt09['datetime'], tt09['TT09'], 'k--', alpha = 0.3,
    #          zorder = -10,label='Gast temperature')
    
    # plt.axvline(np.datetime64('2024-06-14T17:00:00'),ls = '--', color = 'gray', alpha = 0.8)
    # plt.text(np.datetime64('2024-06-14T08:00:00'), 0.78, 
    #          'Filling complete', color = 'gray',rotation=90)
    
    
    if figax == None:
        fig.savefig(f'21062024_mod{mod}.png')
        plt.show()
    else:
        return fig, ax

def plot_gains_occ_ledvoltage(LED_calib, module, channel, labels):
    fig, ax = plt.subplots(1,1, figsize = (6,3))
    _df = LED_calib.results_df
    _df = _df[(_df['module'] == module) & 
              (_df['channel'] == channel) &
              (_df['LEDvoltage'] > 1)]
    ax.errorbar(_df['LEDvoltage'], _df['gain'], yerr = _df['gain_err'], fmt = 'o')
    ax.set_xlabel('LED voltage [V]')
    ax.set_ylabel('Gain [ADC/PE/10^6]')
    ax.set_title(labels[f"mod{module}"][channel])
    
    ax1 = ax.twinx()
    ax1.errorbar(_df['LEDvoltage'], _df['occ'], 
                 yerr = _df['occ_err'], fmt = 'o',
                 color = 'C1')
    ax1.set_ylabel('Occupancy [PE]', color = 'C1')
    ax.grid()
    plt.show()
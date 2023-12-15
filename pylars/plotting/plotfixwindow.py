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
        axs[0].hist(_df[_mask]['led_amplitude'], 
                bins=100, histtype='step', label = f'{_v} V')
        _amp = np.median(_df[_mask]['led_amplitude'])
        _std = np.std(_df[_mask]['led_amplitude'])
        _median_amplitudes.append(_amp)
        _std_amplitudes.append(_std)
    
    axs[1].errorbar(_ledvoltages, _median_amplitudes,
                    yerr = _std_amplitudes, marker = 'o',
                    ls = '-')
    axs[1].set_xlabel('LED voltage [V]')
    axs[1].set_ylabel('Median amplitude [ADC counts]')
    axs[1].grid(color='lightgray', linestyle='--', linewidth=0.5)

    axs[0].set_xlabel('Amplitude [ADC counts]')
    axs[0].set_ylabel('Counts')
    axs[0].legend(loc = 'center left', bbox_to_anchor=(1., 0.5), 
              title = 'LED voltage')
    axs[0].grid(color='lightgray', linestyle='--', linewidth=0.5)

    if figax is None:
        plt.show()
    else:
        return fig, axs


import numpy as np
import matplotlib.pyplot as plt


def plot_waveform(waveform_array:np.array, full_y:bool=False,
                  full_x:bool=True, ax:plt.Axes=None) -> plt.Axes:
    """Plot a waveform from its array.

    Args:
        waveform_array (np.array): waveform array. Elements of the array
    correspond to the sample of the waveform and its value the recorded ADC
    counts.
        full_y (bool, optional): plot full range of ADC amplitude. Defaults
    to False.
        full_x (bool, optional): plot full range of ADC samples. Defaults
    to True.
        ax (plt.Axes, optional): axes to plot into. Defaults to None.

    Returns:
        plt.Axes: axes with plot.
    """
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
    ax.set_xlabel('Sample number')
    ax.set_ylabel('ADC counts')
    return ax

def plot_pulses(waveform:np.array, pulse_list: list,
                ax:plt.Axes=None) -> plt.Axes:
    """Plot the identified pulses in a waveform. 

    Args:
        waveform (np.array): waveform array. Elements of the array
    correspond to the sample of the waveform and its value the recorded ADC
    counts.
        pulse_list (list): list of pulses, as the output of
    pulse_processing.find_pulses_simple.
        ax (plt.Axes, optional): axes to plot into. Defaults to None.

    Returns:
        plt.Axes: axes with plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax = plot_waveform(waveform, ax = ax)
    for pulse in pulse_list:
        if len(pulse)>1:
            ax.fill_betweenx(y = np.linspace(0,16000,100),
                             x1 = pulse[0], x2 = pulse[-1],
                             alpha = 0.2, color = 'cyan')
    return ax
        

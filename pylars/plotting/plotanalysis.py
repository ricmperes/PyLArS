import numpy as np
import matplotlib.pyplot as plt
from .plotprocessed import *


def plot_area_LED(bv_dataset, voltage, LED_position = 300, 
                  log_y = True, full_x = False, ax = None,
                  color = None):
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize = (12,6))
    
    df = bv_dataset.data[voltage]

    cut_mask = ((df['position'] > (LED_position - 10)) &
                (df['position'] < (LED_position + 20)) &
                (df['length'] > 3))

    ax = plot_hist_area(df[cut_mask], ax = ax, color = color)

    if full_x:
        ax.set_xlim(0,2**14*10*300)
    if log_y:
        ax.set_yscale('log')

    ax.set_title((f'LED ON\n module {bv_dataset.module} | '
                  f'channel {bv_dataset.channel[-1]}')
                )

    med = np.median(df[cut_mask]['area'])
    std = np.std(df[cut_mask]['area'])
    med_err = std/np.sqrt(len(df[cut_mask]))
    
    return med, med_err, ax


def plot_LED_all_voltages(bv_dataset, cmap = 'winter', ax = None,):
    cm = plt.get_cmap(cmap)
    N_lines = len(bv_dataset.voltages)

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize = (12,6))
    for i, _v in enumerate(bv_dataset.voltages):
        
        med, med_err, ax = plot_area_LED(bv_dataset, 50, color = cm(i/N_lines))
        ax.axvline(med, color = cm(i/N_lines))
        ax.set_title('')
        
    plt.show()
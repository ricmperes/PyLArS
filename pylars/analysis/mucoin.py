import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pylars
import os
import datetime 

from matplotlib.gridspec import GridSpec


array_layout = np.loadtxt('/disk/gfs_atp/xenoscope/tpc/tiles_layout.txt')
array_labels = ['A', 'B', 'C', 'D', 'E', 'F',
                'G', 'H', 'J', 'K', 'L', 'M']
## Loading

def get_file(n_run, module, run_name, run_period = 'commissioning'):
    if run_period == 'commissioning':
        run_path = '/disk/gfs_atp/xenoscope/tpc/commissioning/pressure_studies/2bar/mucoin/'
    elif run_period == 'filling':
        run_path = '/disk/gfs_atp/xenoscope/tpc/filling/mucoin/'
    elif run_period == 'auto':
        run_path = '/disk/gfs_atp/xenoscope/tpc/filling/mucoin/'
        file_path = f'{run_name}/Module{module}/{run_name}_Module_{module}_0.root'
        return run_path + file_path
    elif run_period == 'test':
        run_path = '/disk/gfs_atp/xenoscope/tpc/filling/mucoin/'
        file_path = f'{run_name}/Module{module}/{run_name}_Module_{module}_0.root'
        return run_path + file_path
    elif run_period == 'ramp_up':
        run_path = '/disk/gfs_atp/xenoscope/tpc/ramp_up/'
        file_path = f'{run_name}/Module{module}/{run_name}_Module_{module}_0.root'
        return run_path + file_path
    elif run_period == 'largewindow':
        run_path = '/disk/gfs_atp/xenoscope/tpc/ramp_up/largewindows/'
        file_path = f'{run_name}/Module{module}/{run_name}_Module_{module}_0.root'
        return run_path + file_path
    
    file_path = f'{run_name}_{n_run:0>3}/Module{module}/{run_name}_{n_run:0>3}_Module_{module}_0.root'
    return run_path + file_path


def get_data_dict(n_run, run_name, run_period):
    process = pylars.processing.rawprocessor.simple_processor(
        sigma_level=3, baseline_samples=50)
    file_mod0 = get_file(n_run, 0, run_name, run_period)
    file_mod1 = get_file(n_run, 1, run_name, run_period)
    
    
    data_dict = {'mod0' : {}, 'mod1' : {}}

    process.load_raw_data(file_mod0, 47, 300, module = 0)
    for _ch in process.raw_data.channels:
        data_dict['mod0'][_ch] = process.raw_data.get_channel_data(_ch)

    process.load_raw_data(file_mod1, 47, 300, module = 0)
    for _ch in process.raw_data.channels:
        data_dict['mod1'][_ch] = process.raw_data.get_channel_data(_ch)
        
    return data_dict

def load_layout():
    array_layout = np.loadtxt('tiles_layout.txt')
    array_labels = ['A', 'B', 'C', 'D', 'E', 'F',
                    'G', 'H', 'J', 'K', 'L', 'M']
        
    return array_layout, array_labels


## Plot waveforms

def plot_s2_waveform_array_mitpattern(data_dict, n_wf, plot = True, 
                           limits_peak = None, limits_wf = None,
                           hitp_var = 'amp', x_unit = 'samples',
                           save_fig = False):
    labels = {'mod0' : {'wf1': 'wf1 | Tile H', 'wf2': 'wf2 | Tile J', 
                        'wf3': 'wf3 | Tile K', 'wf4': 'wf4 | Tile L',
                        'wf5': 'wf5 | Tile M', 
                        'wf6': 'wf6 | Muon detector 1', 
                        'wf7': 'wf7 | Muon detector 2'},
              'mod1' : {'wf1': 'wf1 | Tile A', 'wf2': ' wf2 | Tile B', 
                        'wf3': 'wf3 | Tile C', 'wf4': 'wf4 | Tile D',
                        'wf5': 'wf5 | Tile E', 'wf6': 'wf6 | Tile F', 
                        'wf7': 'wf7 | Tile G' }
             }
    array_layout, array_labels = load_layout()
    fig = plt.figure(figsize = (12,5))
    gs = GridSpec(2, 2, width_ratios=[1, 0.8], height_ratios=[1, 1])

    
    ax_mod1 = fig.add_subplot(gs[0,0])
    ax_mod0 = fig.add_subplot(gs[1,0], sharex = ax_mod1)
    ax_hitp = fig.add_subplot(gs[:,1])
    
    test_mod = list(data_dict.keys())[0] 
    test_ch = list(data_dict[test_mod])[0]
    n_samples = data_dict[test_mod][test_ch].shape[1]
    _x = np.arange(n_samples)
    plt.subplots_adjust(hspace=0)
    
    for _ch in ['wf1','wf2','wf3', 'wf4', 'wf5',]:
        ax_mod0.plot(_x, data_dict['mod0'][_ch][n_wf], 
                    label = labels['mod0'][_ch])

    for _ch in ['wf1','wf2','wf3', 'wf4', 'wf5', 'wf6', 'wf7']:
        ax_mod1.plot(_x, data_dict['mod1'][_ch][n_wf], 
                    label = labels['mod1'][_ch])
    if limits_peak is not None:
        ax_mod1.fill_between(limits_peak,15000,14980, 
                             color = 'gray', alpha = 0.2)
        ax_mod0.fill_between(limits_peak,15000,14980, 
                             color = 'gray', alpha = 0.2)
        
    if limits_wf is not None:
        ax_mod1.set_xlim(limits_wf)
        ax_mod0.set_xlim(limits_wf)

    
    if x_unit == 'time':
        ax_mod0.set_xticks(ax_mod0.get_xticks(), ax_mod0.get_xticks()*10/1000)
        ax_mod0.set_xlabel('Time [us]')
        
    else:
        ax_mod0.set_xlabel('Sample #')

    ax_mod1.set_xticks([])
    
    #Make hitpattern
    hitp_amplitude, hitp_area = make_hitpatterns(data_dict, 
                                                 limits=limits_peak,
                                                 n_wf = n_wf)
    if hitp_var == 'amp':
        ax_hitp, _map = plot_hitpattern(hitpattern = hitp_amplitude, 
                    array_layout = array_layout, 
                    array_labels = array_labels, 
                    ax = ax_hitp)
        fig.colorbar(_map, label = 'Pulse Amplitude')
    else:
        ax_hitp, _map = plot_hitpattern(hitpattern = hitp_area, 
                    array_layout = array_layout, 
                    array_labels = array_labels, 
                    ax = ax_hitp)
        fig.colorbar(_map, label = 'Pulse Area')
        
    ax_hitp.set_xlabel('x [mm]')
    ax_hitp.set_ylabel('x [mm]')
    
    fig.legend(ncol = 5, loc = 'lower center', 
               bbox_to_anchor  = (0,0.9,1,0))
    fig.suptitle(f'Evt # {n_wf}', y = 1.02, x = 0.5, horizontalalignment='center',
                 verticalalignment='center', transform=fig.transFigure)
    if save_fig != False: 
        if isinstance(save_fig, str):
            os.makedirs(f'figures/{save_fig}', exist_ok=True)
            plt.savefig(f'figures/{save_fig}/{save_fig}_{n_wf}.png', dpi = 80)
        else:
            plt.savefig(f'figures/{n_wf}_{int(time.time())}.png', dpi = 80)
    
    if plot:
        plt.show()
    plt.close()


def plot_mu_waveform_array_mitpattern_s1_s2(data_dict, n_wf, plot = True, 
                           limits = None, hitp_var = 'amp',save_fig = False):
    labels = {'mod0' : {'wf1': 'wf1 | Tile H', 'wf2': 'wf2 | Tile J', 
                        'wf3': 'wf3 | Tile K', 'wf4': 'wf4 | Tile L',
                        'wf5': 'wf5 | Tile M', 
                        'wf6': 'wf6 | Muon detector 1', 
                        'wf7': 'wf7 | Muon detector 2'},
              'mod1' : {'wf1': 'wf1 | Tile A', 'wf2': ' wf2 | Tile B', 
                        'wf3': 'wf3 | Tile C', 'wf4': 'wf4 | Tile D',
                        'wf5': 'wf5 | Tile E', 'wf6': 'wf6 | Tile F', 
                        'wf7': 'wf7 | Tile G' }
             }
    array_layout, array_labels = load_layout()
    fig = plt.figure(figsize = (12,6))
    gs = GridSpec(3, 2, width_ratios=[1, 0.8], height_ratios=[1, 1, 1])

    ax_pmt = fig.add_subplot(gs[0,0])
    ax_mod1 = fig.add_subplot(gs[1,0], sharex = ax_pmt)
    ax_mod0 = fig.add_subplot(gs[2,0], sharex = ax_pmt)
    ax_hitp = fig.add_subplot(gs[:,1])
    
    test_mod = list(data_dict.keys())[0] 
    test_ch = list(data_dict[test_mod])[0]
    n_samples = data_dict[test_mod][test_ch].shape[1]
    _x = np.arange(n_samples)
    plt.subplots_adjust(hspace=0)
    
    for _ch in ['wf6','wf7']:
        ax_pmt.plot(_x, data_dict['mod0'][_ch][n_wf], 
                    label = labels['mod0'][_ch]) 
    
    for _ch in ['wf1','wf2','wf3', 'wf4', 'wf5',]:
        ax_mod0.plot(_x, data_dict['mod0'][_ch][n_wf], 
                    label = labels['mod0'][_ch])

    for _ch in ['wf1','wf2','wf3', 'wf4', 'wf5', 'wf6', 'wf7']:
        ax_mod1.plot(_x, data_dict['mod1'][_ch][n_wf], 
                    label = labels['mod1'][_ch])
    if limits is not None:
        ax_mod1.fill_between(limits,15000,14500, color = 'gray', alpha = 0.2)
        ax_mod0.fill_between(limits,15000,14500, color = 'gray', alpha = 0.2)
        
    
    ax_mod0.set_xlabel('Sample #')

    
    #Make hitpattern
    hitp_amplitude, hitp_area = make_hitpatterns(data_dict, 
                                                 limits=limits,n_wf = n_wf)
    if hitp_var == 'amp':
        ax_hitp, _map = plot_hitpattern(hitpattern = hitp_amplitude, 
                    array_layout = array_layout, 
                    array_labels = array_labels, 
                    ax = ax_hitp)
        fig.colorbar(_map, label = 'Pulse Amplitude')
    else:
        ax_hitp, _map = plot_hitpattern(hitpattern = hitp_area, 
                    array_layout = array_layout, 
                    array_labels = array_labels, 
                    ax = ax_hitp)
        fig.colorbar(_map, label = 'Pulse Area')
        
    ax_hitp.set_xlabel('x [mm]')
    ax_hitp.set_ylabel('x [mm]')
    
    fig.legend(ncol = 5, loc = 'lower center', 
               bbox_to_anchor  = (0,0.9,1,0))
    fig.suptitle(f'Evt # {n_wf}', y = 1.02, x = 0.5, horizontalalignment='center',
                 verticalalignment='center', transform=fig.transFigure)
    if save_fig != False: 
        if isinstance(save_fig, str):
            os.makedirs(f'figures/{save_fig}', exist_ok=True)
            plt.savefig(f'figures/{save_fig}/{save_fig}_{n_wf}.png', dpi = 80)
        else:
            plt.savefig(f'figures/{n_wf}_{int(time.time())}.png', dpi = 80)
    
    if plot:
        plt.show()
    plt.close()


def plot_mu_waveform_array_mitpattern(data_dict, n_wf, plot = True, 
                           limits_peak = None, limits_wf = None,
                           hitp_var = 'amp', x_unit = 'samples',
                           save_fig = False):
    labels = {'mod0' : {'wf1': 'wf1 | Tile H', 'wf2': 'wf2 | Tile J', 
                        'wf3': 'wf3 | Tile K', 'wf4': 'wf4 | Tile L',
                        'wf5': 'wf5 | Tile M', 
                        'wf6': 'wf6 | Muon detector 1', 
                        'wf7': 'wf7 | Muon detector 2'},
              'mod1' : {'wf1': 'wf1 | Tile A', 'wf2': ' wf2 | Tile B', 
                        'wf3': 'wf3 | Tile C', 'wf4': 'wf4 | Tile D',
                        'wf5': 'wf5 | Tile E', 'wf6': 'wf6 | Tile F', 
                        'wf7': 'wf7 | Tile G' }
             }
    array_layout, array_labels = load_layout()
    fig = plt.figure(figsize = (12,6))
    gs = GridSpec(3, 2, width_ratios=[1, 0.8], height_ratios=[1, 1, 1])

    ax_pmt = fig.add_subplot(gs[0,0])
    ax_mod1 = fig.add_subplot(gs[1,0], sharex = ax_pmt)
    ax_mod0 = fig.add_subplot(gs[2,0], sharex = ax_pmt)
    ax_hitp = fig.add_subplot(gs[:,1])
    
    test_mod = list(data_dict.keys())[0] 
    test_ch = list(data_dict[test_mod])[0]
    n_samples = data_dict[test_mod][test_ch].shape[1]
    _x = np.arange(n_samples)
    plt.subplots_adjust(hspace=0)
    
    for _ch in ['wf6','wf7']:
        ax_pmt.plot(_x, data_dict['mod0'][_ch][n_wf], 
                    label = labels['mod0'][_ch]) 
    
    for _ch in ['wf1','wf2','wf3', 'wf4', 'wf5',]:
        ax_mod0.plot(_x, data_dict['mod0'][_ch][n_wf], 
                    label = labels['mod0'][_ch])

    for _ch in ['wf1','wf2','wf3', 'wf4', 'wf5', 'wf6', 'wf7']:
        ax_mod1.plot(_x, data_dict['mod1'][_ch][n_wf], 
                    label = labels['mod1'][_ch])
    if limits_peak is not None:
        ax_mod1.fill_between(limits_peak,15000,14980, 
                             color = 'gray', alpha = 0.2)
        ax_mod0.fill_between(limits_peak,15000,14980, 
                             color = 'gray', alpha = 0.2)
        
    if limits_wf is not None:
        ax_mod1.set_xlim(limits_wf)
        ax_mod0.set_xlim(limits_wf)
        ax_pmt.set_xlim(limits_wf)
    
    if x_unit == 'time':
        ax_mod0.set_xticks(ax_mod0.get_xticks(), ax_mod0.get_xticks()*10/1000)
        ax_mod0.set_xlabel('Time [us]')
        
    else:
        ax_mod0.set_xlabel('Sample #')

    
    #Make hitpattern
    hitp_amplitude, hitp_area = make_hitpatterns(data_dict, 
                                                 limits=limits_peak,
                                                 n_wf = n_wf)
    if hitp_var == 'amp':
        ax_hitp, _map = plot_hitpattern(hitpattern = hitp_amplitude, 
                    array_layout = array_layout, 
                    array_labels = array_labels, 
                    ax = ax_hitp)
        fig.colorbar(_map, label = 'Pulse Amplitude')
    else:
        ax_hitp, _map = plot_hitpattern(hitpattern = hitp_area, 
                    array_layout = array_layout, 
                    array_labels = array_labels, 
                    ax = ax_hitp)
        fig.colorbar(_map, label = 'Pulse Area')
        
    ax_hitp.set_xlabel('x [mm]')
    ax_hitp.set_ylabel('x [mm]')
    
    fig.legend(ncol = 5, loc = 'lower center', 
               bbox_to_anchor  = (0,0.9,1,0))
    fig.suptitle(f'Evt # {n_wf}', y = 1.02, x = 0.5, horizontalalignment='center',
                 verticalalignment='center', transform=fig.transFigure)
    if save_fig != False: 
        if isinstance(save_fig, str):
            os.makedirs(f'figures/{save_fig}', exist_ok=True)
            plt.savefig(f'figures/{save_fig}/{save_fig}_{n_wf}.png', dpi = 80)
        else:
            plt.savefig(f'figures/{n_wf}_{int(time.time())}.png', dpi = 80)
    
    if plot:
        plt.show()
    plt.close()


def plot_hitpattern(hitpattern, array_layout, array_labels, ax = None):
    if ax is None:
        fig, ax = plt.subplots(1,1)

    ax, _map = pylars.plotting.plot_hitpattern(
        hitpattern = hitpattern,
        layout = array_layout,
        labels = array_labels,
        r_tpc = 160/2,
        cmap = 'coolwarm_r',
        log = False,
        ax = ax)
    
    ax.set_xlim(-100,100)
    ax.set_ylim(-100,100)
    ax.set_aspect('equal')

    if ax is None:
        plt.show()
    else:
        return ax, _map

def plot_mu_waveform_array(process, data_dict, n_wf, plot = True, save_fig = False):
    labels = {'mod0' : {'wf1': 'wf1 | Tile H', 'wf2': 'wf2 | Tile J', 
                        'wf3': 'wf3 | Tile K', 'wf4': 'wf4 | Tile L',
                        'wf5': 'wf5 | Tile M', 
                        'wf6': 'wf6 | Muon detector 1', 
                        'wf7': 'wf7 | Muon detector 2'},
              'mod1' : {'wf1': 'wf1 | Tile A', 'wf2': ' wf2 | Tile B', 
                        'wf3': 'wf3 | Tile C', 'wf4': 'wf4 | Tile D',
                        'wf5': 'wf5 | Tile E', 'wf6': 'wf6 | Tile F', 
                        'wf7': 'wf7 | Tile G' }
             }
    
    fig, axs = plt.subplots(3,1,sharex=True, figsize=(8,8))
    _x = np.arange(process.raw_data.n_samples)
    plt.subplots_adjust(hspace=0)
    
    for _ch in ['wf6','wf7']:
        axs[0].plot(_x, data_dict['mod0'][_ch][n_wf], 
                    label = labels['mod0'][_ch])
    #axs[0].legend()    
    
    for _ch in ['wf1','wf2','wf3', 'wf4', 'wf5',]:
        axs[2].plot(_x, data_dict['mod0'][_ch][n_wf], 
                    label = labels['mod0'][_ch])
    #axs[1].legend()
    
    for _ch in ['wf1','wf2','wf3', 'wf4', 'wf5', 'wf6', 'wf7']:
        axs[1].plot(_x, data_dict['mod1'][_ch][n_wf], 
                    label = labels['mod1'][_ch])
    #axs[2].legend()
    
    fig.suptitle(f'Evt # {n_wf}', y = 1.5, x = 0.5, horizontalalignment='center',
                 verticalalignment='center', transform=axs[0].transAxes)
    axs[1].set_xlabel('Sample #')
    #axs[0].set_xlim(400, 600)
    #axs[1].set_xlim(400, 600)
    fig.legend(ncol = 5, loc = 'lower center', 
               bbox_to_anchor  = (0,0.9,1,0))
    if save_fig != False: 
        if isinstance(save_fig, str):
            os.makedirs(f'figures/{save_fig}', exist_ok=True)
            plt.savefig(f'figures/{save_fig}/{save_fig}_{n_wf}.png', dpi = 80)
        else:
            plt.savefig(f'figures/{n_wf}_{int(time.time())}.png', dpi = 80)
    
    if plot:
        plt.show()
    plt.close()

## Waveform processing

def get_amplitude_area(wf, limits):
    _amplitude = pylars.processing.pulses.pulse_processing.get_amplitude(
        waveform = wf,
        baseline_value=15000,
        peak_start=limits[0],
        peak_end=limits[1],
        negative_polarity=True,
        baseline_subtracted=False)

    _area = pylars.processing.pulses.pulse_processing.get_area(
        waveform = wf,
        baseline_value=15000,
        pulse_start=limits[0],
        pulse_end=limits[1],
        negative_polarity=True,
        baseline_subtracted=False)

    return _amplitude, _area

def make_hitpatterns(data_dict, n_wf, limits):
    hitpattern_amplitude = []
    hitpattern_area = []

    for _ch in ['wf1','wf2','wf3', 'wf4', 'wf5','wf6','wf7']:
        _wf = data_dict['mod1'][_ch][n_wf]
        _amplitude, _area = get_amplitude_area(_wf, limits)
        
        hitpattern_amplitude.append(_amplitude)
        hitpattern_area.append(_area)
    
    for _ch in ['wf1','wf2','wf3', 'wf4', 'wf5']:
        _wf = data_dict['mod0'][_ch][n_wf]
        _amplitude, _area = get_amplitude_area(_wf, limits)
        
        hitpattern_amplitude.append(_amplitude)
        hitpattern_area.append(_area)
    return np.array(hitpattern_amplitude), np.array(hitpattern_area)

def check_wf_for_peak(wf, threshold = 14500, limits = None):
    if limits is None:
        _wf = wf
    else:
        _wf = wf[limits[0]:limits[1]]
    found_peak = (_wf < threshold).any()
    if found_peak:
        peak_ids = np.where(_wf < threshold)[0]
        return peak_ids
    else:
        return None
    
def check_all_wfs(data_dict, mod, ch, threshold = 14500, limits = None):
    channel_data = data_dict[f'mod{mod}'][ch]
    wfs_with_peaks = {}
    
    for i, wf in enumerate(channel_data):
        peak_ids = check_wf_for_peak(wf, threshold = threshold, 
                                     limits = limits)
        if peak_ids is not None:
            wfs_with_peaks[i] = peak_ids
            
    return wfs_with_peaks
            

def join_peak_list(found_bumps):
    peak_list = []
    for modch in found_bumps.keys():
        for n_wf in found_bumps[modch]:
            if n_wf not in peak_list:
                peak_list.append(n_wf)
    return peak_list

def process_file(n_run, 
                 run_name,
                 run_period = 'commissioning',
                 threshold = 14500, 
                 limits = None, 
                 plot_wfs = False, pattern = True,
                 save_fig = False):
    data_dict = get_data_dict(n_run, run_name, run_period)
    found_bumps = {}
    for mod in [0]:
        for ch in ['wf1','wf2','wf3','wf4','wf5']:
            found_bumps[f'{mod}_{ch}'] = check_all_wfs(
                data_dict, 
                mod,ch, 
                threshold=threshold,
                limits = limits)
    for mod in [1]:
        for ch in ['wf1','wf2','wf3','wf4','wf5','wf6','wf7']:
            found_bumps[f'{mod}_{ch}'] = check_all_wfs(
                data_dict, 
                mod,ch, 
                threshold=threshold,
                limits = limits)
    
    peaks_list = join_peak_list(found_bumps)
    peaks_list.sort()
    if plot_wfs or save_fig:
        for n_wf in peaks_list:
            if pattern == True:
                plot_mu_waveform_array_mitpattern(data_dict, 
                                                  n_wf=n_wf, 
                                                  plot = plot_wfs, 
                                                  limits = limits, 
                                                  hitp_var = 'amp',
                                                  save_fig = save_fig)
            else:
                plot_mu_waveform_array(data_dict, 
                                       n_wf = n_wf,
                                       plot = plot_wfs,
                                       save_fig=save_fig)
            
    return peaks_list, found_bumps


## Baselines 

### Calculate baselines

def get_baseline_channel(data_dict, mod, ch):
    channel_data = data_dict[f'mod{mod}'][ch]
    baselines = np.median(channel_data, axis = 1)
    assert len(baselines) == channel_data.shape[0]
    std = np.std(channel_data, axis = 1)
    return baselines, std

def get_avgbaseline_all_channels(data_dict):
    avg_baselines = {'mod0' : {}, 'mod1' : {}}
    avg_stds = {'mod0' : {}, 'mod1' : {}}

    for i, ch in enumerate(['wf1','wf2','wf3', 'wf4', 'wf5']):
        _baselines, _stds = get_baseline_channel(data_dict, 0, ch)  

        avg_baselines[f'mod0'][ch] = np.average(_baselines)
        avg_stds[f'mod0'][ch] = np.average(_stds)
    
    for j, ch in enumerate(['wf1','wf2','wf3', 'wf4', 'wf5', 'wf6','wf7']):
        _baselines, _stds = get_baseline_channel(data_dict, 1, ch)  

        avg_baselines[f'mod1'][ch] = np.average(_baselines)
        avg_stds[f'mod1'][ch] = np.average(_stds)
    
    return avg_baselines, avg_stds


###  Plot baselines

def plot_baseline_channel(data_dict, mod, ch, figax = None):
    if figax is None:
        fig, ax = plt.subplots(1,1, figsize = (8,3))
    else:
        fig, ax = figax
    baselines, std = get_baseline_channel(data_dict, mod, ch)

    ax.errorbar(np.arange(200),baselines, yerr=std, ls = '' , 
                marker = 'o', capsize=2)
    #ax.set_xlabel('Waveform #')
    #ax.set_ylabel('Baseline [ADC]')
    ax.text(0.05,0.95,s = f'Mod {mod} | '+labels[f'mod{mod}'][ch],
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.5))
    if figax is None:
        plt.show()
    else:
        return fig, ax

def plot_baseline_all_channels(data_dict, figax = None):
    if figax is None:
        fig, axs = plt.subplots(3,4, figsize = (20,10), 
                                sharey = True, sharex = True,
                                gridspec_kw = {'hspace':0, 'wspace':0},
                                constrained_layout = False)
        axs = axs.flatten()
    else:
        fig, axs = figax

    for i, ch in enumerate(['wf1','wf2','wf3', 'wf4', 'wf5']):
        fig, axs[i] = plot_baseline_channel(data_dict, 0, ch, 
                                            figax = (fig, axs[i]))
    
    for j, ch in enumerate(['wf1','wf2','wf3', 'wf4', 'wf5', 'wf6','wf7']):
        fig, axs[i+j+1] = plot_baseline_channel(data_dict, 1, ch, 
                                                figax = (fig, axs[i+j+1]))
    axs[0].set_ylim(14750, 15300)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)

    # Put x and y label in the center
    big_ax = fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big subplot
    big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    big_ax.set_xlabel('Waveform number', labelpad=10, )  # set the common x label
    big_ax.set_ylabel('Baseline [ADCcounts]', labelpad=25)  # set the common y label

    if figax is None:
        plt.savefig('baselines_run001.jpeg', dpi = 120, bbox_inches='tight', pad_inches=0.1)
        plt.show()
    else:
        return fig, axs
    

## Rates

def get_livetime(run_number, run_path, run_name):
    with open(run_path + 
              f'{run_name}_{run_number:0>3}/Summary_{run_name}_{run_number:0>3}.txt', 'r') as f:
        F = f.readlines()
    time = F[1].strip().split()[3]
    
    return datetime.timedelta(seconds = int(time))

def plot_rates(result_df, save_fig = False):
    fig, ax = plt.subplots(1,1,figsize = (6,4), facecolor='white', gridspec_kw = {'hspace':0, 'wspace':0})

    n_runs = len(result_df)
    ax.bar(np.arange(n_runs), 200/result_df['livetime'].dt.total_seconds(), alpha = 1, label = 'Double coin (2xPMT)')
    ax.bar(np.arange(n_runs), result_df['rate'], alpha = 1, label = 'Triple coin (2xPMT + SiPM)')
    ax.set_ylabel('Rate [Hz]')
    ax.set_xlabel('Run #')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.set_yscale('log')
    ax.legend(loc = 'lower left', bbox_to_anchor = (0,1,1,0.5))



    #ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1., 0, 0.25, 1], sharey=ax)
    ax_histy.hist(200/result_df['livetime'].dt.total_seconds(), bins = np.logspace(-5,-2,20), alpha = 1, histtype='step', color = 'C0', orientation='horizontal')
    ax_histy.hist(result_df['rate'], bins = np.logspace(-5,-2,20), alpha = 1, histtype='step', color = 'C1', orientation='horizontal')

    med_2coin = np.median(200/result_df['livetime'].dt.total_seconds())
    med_3coin = np.median(result_df['rate'])
    ax_histy.axhline(med_2coin, ls = '--', color = 'C0', 
                    label = f'2coin median: {med_2coin:.2e} Hz')
    ax_histy.axhline(np.median(result_df['rate']), ls = '--', color = 'C1',
                    label = f'3coin median: {med_3coin:.2e} Hz')
    #ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.set_xlabel('Rate [Hz]')
    ax_histy.legend(loc = 'lower right', bbox_to_anchor = (0,1,1,0.5))
    # plt.subplot(122)
    # plt.hist(result_df['rate'], bins = 10, alpha = 1, histtype='step', color = 'C1')
    # plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    # plt.xlabel('Rate [Hz]')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                        wspace=0, hspace=0)
    if save_fig is not False:
        plt.savefig('coin_rates_{save_fig}.jpeg', dpi = 120, 
                    bbox_inches='tight', pad_inches=0.1)
    else: 
        plt.show()
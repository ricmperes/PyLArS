


class mucoin_analysis():
    """Function and methods to perform muon coincidence analysis.
    """

    def get_data_dict(n_run):
        file_mod0 = get_file(n_run, 0)
        file_mod1 = get_file(n_run, 1)
        
        
        data_dict = {'mod0' : {}, 'mod1' : {}}
        
        process.load_raw_data(file_mod0, 47, 300, module = 0)
        for _ch in process.raw_data.channels:
            data_dict['mod0'][_ch] = process.raw_data.get_channel_data(_ch)

        process.load_raw_data(file_mod1, 47, 300, module = 0)
        for _ch in process.raw_data.channels:
            data_dict['mod1'][_ch] = process.raw_data.get_channel_data(_ch)
            
        return data_dict

    def join_peak_list(found_bumps):
        peak_list = []
        for modch in found_bumps.keys():
            for n_wf in found_bumps[modch]:
                if n_wf not in peak_list:
                    peak_list.append(n_wf)
        return peak_list

    def process_file(n_run, threshold = 14500, limits = None, 
                    plot_wfs = False, save_fig = False):
        data_dict = get_data_dict(n_run)
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
                plot_mu_waveform_array(data_dict, 
                                    n_wf = n_wf,
                                    plot = plot_wfs,
                                    save_fig=save_fig)
        return peaks_list, found_bumps
    
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
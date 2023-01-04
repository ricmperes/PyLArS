import argparse
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pylars
import pylars.utils.input
from pylars.utils.common import get_channel_list, get_gain
from tqdm.autonotebook import tqdm
from itertools import product

parser = argparse.ArgumentParser(
    description=('Script to compute properties of DCR runs based on '
                 'amplitude.'))

parser.add_argument('-r', '--run',
                    help='Run number.',
                    type=int,
                    required=True)
parser.add_argument('-t', '--temperature',
                    help='Temperature to look for.',
                    type=int,
                    required=True)
parser.add_argument('-n', '--n_processes',
                    help=('Number of processed to use when making patterns.'
                          'Defaults to None, i.e. all the available cpus.'),
                    required=False,
                    default=None)


args = parser.parse_args()


def prepare():
    print('Starting preparations of base_run, etc.')
    base_run = pylars.utils.input.run(
        run_number=args.run,
        main_data_path='/disk/gfs_atp/xenoscope/SiPMs/char_campaign/raw_data/')
    process = pylars.processing.rawprocessor.run_processor(
        run_to_process=base_run,
        processor_type='simple',
        sigma_level=5,
        baseline_samples=50)
    all_channels = get_channel_list(process)
    
    
    print('Channels found: ', all_channels)
    print('Preperations done.')

    DCR_datasets = [pylars.analysis.darkcount.DCR_dataset(run = base_run,
                                                    temperature = args.temperature,
                                                    module = mod,
                                                    channel = ch,
                                                    processor = process,
                                                    ) for mod, ch in all_channels]
    
    return base_run, process, all_channels, DCR_datasets

def load_data_to_ds(ds):
    print(f'Starting to load data for module{ds.module}, '
          f'channel {ds.channel}.')
    ds.load_processed_data()
    print(f'Loaded data for module{ds.module}, channel {ds.channel}.')
    return ds
    
def compute_properties_of_ds_amplitude(params):
    ds, _voltage = params
    length_cut_min = 4
    length_cut_max = 50
    plot_name_1pe_fit = (f'{ds.temp}K_{_voltage}V_mod{ds.module}_'
                         f'ch{ds.channel}_amplitude_based')
    try:
        ds.define_SiPM_config()
        df = ds.data[_voltage]
        pulse_count = ds.get_how_many_peaks_per_waveform(df)
        wfs_1_pulse = pulse_count[
            pulse_count['pulse_count'] == 1]['wf_number'].values
        mask_1_pulse_in_wf = [i in wfs_1_pulse for i in tqdm(
            df['wf_number'],
            desc= 'Concatenating number of pulses to mask: ',
            leave = False)]

        med_amplitude, std_amplitude = ds.get_med_amplitude(
            df, mask_1_pulse_in_wf)

        # Get SPE value from Gaussian fit
        (A, mu, sigma), cov = ds.get_1pe_value_fit(
            df = df[mask_1_pulse_in_wf], 
            length_cut = length_cut_min,
            plot=plot_name_1pe_fit)

        # Calculate DCR and CTP
        DCR, DCR_error, CTP, CTP_error = ds.get_DCR_amplitude(
            df=df,
            length_cut_min=length_cut_min,
            length_cut_max=length_cut_max,
            pe_amplitude=med_amplitude,
            pe_amplitude_std=std_amplitude,
            sensor_area=ds.SiPM_config['sensor_area'],
            t=ds.livetimes[_voltage])

        # Calculate gain
        gain = get_gain(F_amp = ds.run.ADC_config['F_amp'],
            spe_area = mu ,
            ADC_range = ds.run.ADC_config['ADC_range'],
            ADC_impedance = ds.run.ADC_config['ADC_impedance'],
            ADC_res = ds.run.ADC_config['ADC_res'],
            q_e = ds.run.ADC_config['q_e'])

    except:
        print(f"Process failed for module {ds.module}, channel "
              f"{ds.channel} at voltage {_voltage} V.")
        gain = mu = sigma = DCR = DCR_error = CTP = CTP_error = np.nan

    _results_dataset = pd.DataFrame({'module': [ds.module],
                                     'channel': [ds.channel],
                                     'T': [ds.temp],
                                     'V': [_voltage],
                                     'pe_area': [mu],
                                     'Gain': [gain],
                                     'res': [mu / sigma],
                                     'DCR': [DCR],
                                     'DCR_error': [DCR_error],
                                     'CTP': [CTP],
                                     'CTP_error': [CTP_error]})

    return _results_dataset

if __name__ == '__main__':
    base_run, process, all_channels, DCR_datasets = prepare()

    print('Start loading data:')
    with Pool() as pool:
        DCR_datasets_loaded = pool.map(load_data_to_ds, DCR_datasets)
    del DCR_datasets
    print('#### All data loaded, moving on with the computation. ####')
    #Assuming all the datasets have the same voltages...
    params = list(
        product(DCR_datasets_loaded, DCR_datasets_loaded[0].voltages))
    print(("Starting mp.pool! Lets goooo!\n"
           "(Beware tqdm, it's displaying for 1 out fo N runs. "
           "Just trust the process, ok??)"))
    
    with Pool() as pool:
        result = pool.map(compute_properties_of_ds_amplitude, params)
    
    print('Got results back. Concatenating...')
    df = pd.concat(result, ignore_index=True)

    df.to_hdf(f'DCR_amplitude_run{args.run}_{args.temperature}K.h5','df')
    print('Finished computing and saving results! '
          'All the best for your analysis!')

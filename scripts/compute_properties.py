import argparse
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pylars
import pylars.utils.input
from pylars.utils.common import get_channel_list, get_gain
from tqdm import tqdm
from itertools import product

parser = argparse.ArgumentParser(
    description=('Script to compute properties of DCR runs based on '
                 'amplitude.'))

parser.add_argument('-r', '--run',
                    help='Run number.',
                    type=int,
                    required=True)

args = parser.parse_args()


def prepare():
    main_data_path='/disk/gfs_atp/xenoscope/SiPMs/char_campaign/raw_data/'
    if args.run == 8:
        F_amp = 20
    else:
        F_amp = 200

    if args.run == 9:
        polarity = False
        main_data_path='/disk/gfs_atp/xenoscope/SiPMs/FebMar2022/6x6A/'
    else:
        polarity = True

    print('Starting preparations of base_run, etc.')
    base_run = pylars.utils.input.run(
        run_number=args.run,
        main_data_path=main_data_path,
        F_amp=F_amp,
        signal_negative_polarity=polarity)
    process = pylars.processing.rawprocessor.run_processor(
        run_to_process=base_run,
        processor_type='simple',
        sigma_level=5,
        baseline_samples=50)
    DCR_run = pylars.analysis.darkcount.DCR_run(run=base_run,
                                                processor=process,
                                                use_n_pulse=True)
    all_channels = get_channel_list(process)

    print('Channels found: ', all_channels)
    print('Preperations done.')

    return base_run, process, all_channels, DCR_run


def load_data_to_ds(ds):
    print(f'Starting to load data for module{ds.module}, '
          f'channel {ds.channel}.')
    ds.load_processed_data()
    print(f'Loaded data for module{ds.module}, channel {ds.channel}.')
    return ds


if __name__ == '__main__':
    base_run, process, all_channels, DCR_run = prepare()
    DCR_run.set_plots_flag(False)

    if args.run == 8:
        DCR_run.define_run_SiPM_config(24*24)
        DCR_run.set_standard_cuts_run(cut_n_pulses_max=8)
    elif args.run == 9:
        DCR_run.define_run_SiPM_config(6*6)
        DCR_run.set_standard_cuts_run(cut_n_pulses_max=2)
    else:
        DCR_run.define_run_SiPM_config()
        DCR_run.set_standard_cuts_run(cut_n_pulses_max=3)
    
    print('Starting loading and computation')
    DCR_run.compute_properties_of_run(amplitude_based=False)

    print('Got results back. Saving.')
    DCR_run.save_results('auto_13022022')

    print('Finished computing and saving results!\n'
          'All the best for your analysis!')

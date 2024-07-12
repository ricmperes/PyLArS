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
parser.add_argument('-t', '--temperature',
                    help='Temperature to look for.',
                    type=int,
                    required=True)
parser.add_argument('-v', '--voltage',
                    help='Voltage to look for.',
                    type=float,
                    required=True)
parser.add_argument('-m', '--module',
                    help='Module to look for.',
                    required=True,
                    type=int)
parser.add_argument('-c', '--channel',
                    help='Channel to look for.',
                    required=True,
                    type=str)

args = parser.parse_args()


def prepare():
    print('Starting preparations of base_run, etc.')
    if args.run == 8:
        F_amp = 20
    else:
        F_amp = 200
    base_run = pylars.utils.input.run(
        run_number=args.run,
        main_data_path='/disk/gfs_atp/xenoscope/SiPMs/char_campaign/raw_data/',
        F_amp=F_amp)
    process = pylars.processing.rawprocessor.run_processor(
        run_to_process=base_run,
        processor_type='simple',
        sigma_level=5,
        baseline_samples=50)

    DCR_dataset = pylars.analysis.darkcount.DCR_dataset(
        run=base_run,
        temperature=args.temperature,
        module=args.module,
        channel=args.channel,
        processor=process,
    )

    print('Preperations done.')
    return base_run, process, DCR_dataset


def get_count_df_and_mask(ds, _voltage):

    df = ds.data[_voltage]
    pulse_count = ds.get_how_many_peaks_per_waveform(df)
    wfs_1_pulse = pulse_count[
        pulse_count['pulse_count'] == 1]['wf_number'].values
    mask_1_pulse_in_wf = [i in wfs_1_pulse for i in tqdm(
        df['wf_number'],
        desc='Concatenating number of pulses to mask: ',
        leave=False)]

    return pulse_count, mask_1_pulse_in_wf


if __name__ == '__main__':
    base_run, process, DCR_dataset = prepare()
    DCR_dataset.load_processed_data()
    DCR_dataset.define_SiPM_config()
    pulse_count, mask_1_pulse_in_wf = get_count_df_and_mask(
        DCR_dataset, args.voltage
    )
    pulse_count.to_hdf((f'/disk/gfs_atp/xenoscope/SiPMs/char_campaign/'
                        f'analysis_data/aux_files/'
                        f'pulse_count_run{args.run}_T{args.temperature}'
                        f'_V{args.voltage}_mod{args.module}_'
                        f'ch{args.channel}.h5'), 'df')
    np.save((f'/disk/gfs_atp/xenoscope/SiPMs/char_campaign/'
             f'analysis_data/aux_files/'
             f'mask_1pulse_t_run{args.run}_T{args.temperature}'
             f'_V{args.voltage}_mod{args.module}_'
             f'ch{args.channel}'), np.array(mask_1_pulse_in_wf))

    print('Done!')

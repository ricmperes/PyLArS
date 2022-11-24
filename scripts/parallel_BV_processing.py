import argparse
from multiprocessing import Pool

import pylars
import pylars.utils.input
from pylars.utils.common import get_channel_list

parser = argparse.ArgumentParser(
    description=('Script to process BV LED to get BV.'))

parser.add_argument('-r', '--run',
                    help='Run number.',
                    type=int,
                    required=True)
parser.add_argument('-t', '--temperature',
                    help='Temperature to look for.',
                    type=int,
                    required=True)
parser.add_argument('-l', '--ledposition',
                    help=('Sample number where LED pulse was triggered'),
                    type = int,
                    required=True,
                    default=None)
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
    process = pylars.processing.rawprocessor.run_processor(run_to_process = base_run, 
                                                       processor_type='simple', 
                                                       sigma_level=5, 
                                                       baseline_samples=50)
    all_channels = get_channel_list(process)
    print('Channels found: ',all_channels)
    print('Preperations done.')
    return base_run, process, all_channels

def process_BV_dataset(param):
    mod, ch = param
    BV_dataset = pylars.analysis.BV_dataset(run = base_run, 
                                            temperature = args.temperature,
                                            module = mod, 
                                            channel =  ch,
                                            processor = process)
    BV_dataset.load_processed_data()
    BV = BV_dataset.compute_BV_LED_simple(LED_position = args.ledposition, 
                                          plot = True)
    return BV

if __name__ == '__main__':
    base_run, process, all_channels = prepare()
    
    print(("Starting mp.pool! Lets goooo!\n"
           "(Beware tqdm, it's displaying for 1 out fo N runs. "
           "Just trust the process, ok??)"))
    with Pool() as pool:
       result = pool.map(process_BV_dataset, all_channels)
    print(result)

    with open(f'BV_results_run{args.run}_{args.temperature}.csv', 'w') as F:
        F.write(f'#Run {args.run}; T = {args.temperature}\n')
        F.write('#Module, Channel, BV, BV_std, r2\n')
        for i, (mod, ch) in enumerate(all_channels):
            F.write((f'{mod}, {ch}, {result[i][0]}, {result[i][1]}, '
                     f'{result[i][2]}\n'))
    
    print('Done!')
    

import pylars
import os
import datetime
import argparse
from pylars.utils.common import get_channel_list
from itertools import product

parser = argparse.ArgumentParser(
    description=('Script to create ALL the batch jobs to process a run '
                 '(no conditions on temperature, voltage or kind).')
)
parser.add_argument('-r', '--run',
                    help='Run to consider.',
                    type=int,
                    required=True)
parser.add_argument('-t', '--temperature',
                    help='Temp to consider.',
                    type=int,
                    required=True)

args = parser.parse_args()


def make_batch_script(job_name, run, temp, vbias, module, channel):
    main_str = f"""#!/bin/bash
#SBATCH --partition=express
#SBATCH --job-name={job_name}
#SBATCH --output=/home/atp/rperes/logs/jobs/{job_name}.out
#SBATCH --error=/home/atp/rperes/logs/jobs/{job_name}.err
#SBATCH --mem=15G

source /home/atp/rperes/.bashrc
conda activate sipms
cd /home/atp/rperes/software/PyLArS/scripts
python count_peaks_in_waveform.py -r {run} -t {temp} -v {vbias} -m {module} -c {channel}
"""
    if not os.path.exists('jobs'):
        os.mkdir('jobs')

    with open(f'jobs/job_{job_name}.job', 'w') as F:
        F.write(main_str)
    print(f'Generated file with ID: {job_name}.')


def make_launch_file(ID_list):
    string_head = '''#!/bin/bash
'''
    with open('launch_process.sh', 'w') as F:
        F.write(string_head)
        for _ID in ID_list:
            F.write(f'sbatch jobs/job_{_ID}.job\n')
    print('Generated launch script file with %d IDs.' % len(ID_list))
    os.system('chmod +x launch_process.sh')

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

    DCR_datasets = pylars.analysis.darkcount.DCR_dataset(
        run = base_run,
        temperature = args.temperature,
        module = all_channels[0][0],
        channel = all_channels[0][1],
        processor = process,
        )
    voltages = DCR_datasets.voltages

    return all_channels, voltages

def main():

    ### INPUTS HERE ###
    run_number = args.run
    temp = args.temperature
    ### ### ###
    
    all_channels, voltages = prepare()
    all_combinations = list(product(all_channels, voltages))
    
    ID_list = []
    for params in all_combinations:
        module = params[0][0]
        channel = params[0][1]
        voltage = params[1]
       

        job_name = f'run{run_number}_{temp}_{voltage}_{module}_{channel}'
        ID_list.append(job_name)

        make_batch_script(job_name = job_name,
                          run = run_number,
                          temp = temp,
                          vbias = voltage,
                          module = module,
                          channel = channel)

    make_launch_file(ID_list)


if __name__ == '__main__':
    t0 = datetime.datetime.now()
    print('Starting to make job files at: ', t0)
    main()

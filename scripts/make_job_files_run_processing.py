import pylars
import os
import datetime
import argparse

parser = argparse.ArgumentParser(
    description=('Script to create ALL the batch jobs to process a run '
                 '(no conditions on temperature, voltage or kind).')
)
parser.add_argument('-r', '--run',
                    help='Run to consider.',
                    type=int,
                    required=True)

args = parser.parse_args()


def make_batch_script(job_name, run, kind, temp, vbias):
    main_str = f"""#!/bin/bash
#SBATCH --partition=express
#SBATCH --job-name={job_name}
#SBATCH --output=/home/atp/rperes/logs/jobs/{job_name}.out
#SBATCH --error=/home/atp/rperes/logs/jobs/{job_name}.err
#SBATCH --mem=14G

source /home/atp/rperes/.bashrc
conda activate sipms
cd /home/atp/rperes/software/PyLArS/scripts
python process_dataset.py -t {temp} -v {vbias} -tp {kind} -r {run}
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


def main():

    ### INPUTS HERE ###
    run_number = args.run
    ### ### ###
    if args.run == 8:
        F_amp = 20
    else:
        F_amp = 200
    base_run = pylars.utils.input.run(
        run_number=args.run,
        main_data_path='/disk/gfs_atp/xenoscope/SiPMs/char_campaign/raw_data/',
        F_amp=F_amp)

    datasets = base_run.get_run_df()
    ID_list = []
    for dset in datasets.itertuples():
        _kind = dset.kind
        _temp = dset.temp
        _vbias = dset.vbias
        _run = run_number

        job_name = f'run{run_number}_{_kind}_{_temp}_{_vbias}'
        ID_list.append(job_name)

        make_batch_script(job_name, _run, _kind, _temp, _vbias)

    make_launch_file(ID_list)


if __name__ == '__main__':
    t0 = datetime.datetime.now()
    print('Starting to make job files at: ', t0)
    main()

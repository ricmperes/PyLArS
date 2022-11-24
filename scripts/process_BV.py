
import pylars.utils.input
import pylars.utils.output
import pylars
import datetime
import argparse
from tqdm.autonotebook import tqdm
import numpy as np

parser = argparse.ArgumentParser(
    description=('Script to process BV datasets for a given set of'
                 'temperatures  and all the available voltages of each'
                 '(sigma=5, baseline counts = 50).')
)
parser.add_argument('-t', '--temperature',
                    nargs='+',
                    help='Temperatures to search for in available datasets.',
                    type=float,
                    required=True)

parser.add_argument('-r', '--run',
                    help='Run to consider.',
                    type=int,
                    required=True)

parser.add_argument('-pr', '--path_raw',
                    help='Path to the main raw files directory.',
                    default='/disk/gfs_atp/xenoscope/SiPMs/char_campaign/raw_data/',
                    type=str)
parser.add_argument('-pp', '--path_processed',
                    help='Path to the processed files directory.',
                    default='/disk/gfs_atp/xenoscope/SiPMs/char_campaign/processed_data/',
                    type=str)

args = parser.parse_args()


def run_process_dataset(run_number: int, kind: str,
                        vbias: float, temp: float,
                        main_data_path: str, path_processed: str) -> None:
    # Define run
    base_run = pylars.utils.input.run(
        run_number=run_number,
        main_data_path=main_data_path)

    # Define processor
    process = pylars.processing.rawprocessor.run_processor(
        base_run, 'simple', sigma_level=5, baseline_samples=50)

    # Turn off or on tqdm. In the future this should have a log anyway.
    process.set_tqdm_channel(bar=True, show=True)
    process.set_tqdm_run(bar=False, show=False)

    # Process the data. Not needed if using load_data=force
    #data = process.process_datasets(kind= kind, vbias=vbias, temp = 170)

    # Make the processed data object and save the data in a file
    processed_data = pylars.utils.output.processed_dataset(
        run=base_run,
        kind=kind,
        vbias=vbias,
        temp=temp,
        path_processed=path_processed,
        process_hash=process.hash)

    processed_data.load_data(force=True)


# MAIN
if __name__ == '__main__':
    t0 = datetime.datetime.now()
    print('Processing started at: ', t0)
    base_run = pylars.utils.input.run(
        run_number=args.run,
        main_data_path='/disk/gfs_atp/xenoscope/SiPMs/char_campaign/raw_data/')

    datasets_df = base_run.get_run_df()
    print('Temperatures to process: ', args.temperature)
    for _temp in tqdm(args.temperature, 'Temperatures to process:'):
        _voltages = np.unique(
            datasets_df[(datasets_df['kind'] == 'BV') &
                        (datasets_df['temp'] == _temp)]['vbias']
        )
        for _volt in tqdm(_voltages, 'Voltages to process:'):
            try:
                run_process_dataset(
                    run_number=args.run,
                    kind='BV',
                    vbias=_volt,
                    temp=_temp,
                    main_data_path=args.path_raw,
                    path_processed=args.path_processed)
            except BaseException:
                print(f'Could not process file for BV, {_volt} V, {_temp} K.')

    tf = datetime.datetime.now()
    print('Processing finished at: ', tf)

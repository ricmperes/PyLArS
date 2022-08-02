import datetime
import numpy as np
import pylars.utils.output
import pylars.utils.input
import pylars
import argparse

parser = argparse.ArgumentParser(
    description=("Script to process a all the datasets given a certain "
                 "condition on 'kind', 'temperature', or 'voltage' with "
                 "pylars. Loads the reference to all the datasets given "
                 "the condition provided and one by one checks if a run "
                 "is already saved and, if not, processes the run and "
                 "saves it. Uses Default processing config (sigma=5, "
                 "baseline counts = 50)."),
)
parser.add_argument('-r', '--run',
                    help='Run to consider.',
                    type=int,
                    required=True
                    )
parser.add_argument('-c', '--condition',
                    help='Temperatures to search for in available datasets.',
                    choices=['kind', 'temperature', 'voltage'],
                    type=str,
                    required=True
                    )
parser.add_argument('-v', '--values',
                    help='Value of the condition.',
                    required=True,
                    nargs='*'  # inputs are put in a list
                    )
parser.add_argument('-pr', '--path_raw',
                    help='Path to the main raw files directory.',
                    default='/disk/gfs_atp/xenoscope/SiPMs/char_campaign/raw_data/',
                    type=str
                    )
parser.add_argument('-pp', '--path_processed',
                    help='Path to the processed files directory.',
                    default='/disk/gfs_atp/xenoscope/SiPMs/char_campaign/processed_data/',
                    type=str
                    )

args = parser.parse_args()


def get_config_lists(run_number: int,
                     condition: str,
                     values: list,
                     main_data_path: str,
                     ) -> tuple:

    base_run = pylars.utils.input.run(
        run_number=run_number,
        main_data_path=main_data_path)

    all_datasets = base_run.get_run_df()

    if condition == 'kind':
        selection = all_datasets[all_datasets['kind'] == values[0]]
        kind_list = values
        vbias_list = np.unique(selection['vbias'].values)
        temp_list = np.unique(selection['temp'].values)

    elif condition == 'temperature':
        values = [float(_val) for _val in values]
        selection = all_datasets[all_datasets['temp'].isin(values)]
        kind_list = np.unique(selection['kind'].values)
        vbias_list = np.unique(selection['vbias'].values)
        temp_list = values

    elif condition == 'voltage':
        values = [float(_val) for _val in values]
        selection = all_datasets[all_datasets['vbias'].isin(values)]
        kind_list = np.unique(selection['kind'].values)
        vbias_list = values
        temp_list = np.unique(selection['temp'].values)

    print('The following kinds will be processed:', kind_list)
    print('The following voltages will be processed:', vbias_list)
    print('The following temperatures will be processed:', temp_list)
    return (kind_list, vbias_list, temp_list)


def process_dataset(run_number: int, kind: str,
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


def process_all_configs(run_number: int,
                        kind_list: list,
                        vbias_list: list,
                        temp_list: list,
                        main_data_path: str,
                        path_processed: str) -> None:

    for _kind in kind_list:
        for _temp in temp_list:
            for _vbias in vbias_list:
                process_dataset(run_number=run_number,
                                kind=_kind,
                                vbias=_vbias,
                                temp=_temp,
                                main_data_path=main_data_path,
                                path_processed=path_processed
                                )


# MAIN
if __name__ == '__main__':
    t0 = datetime.datetime.now()
    print('Processing started at: ', t0)
    (kind_list, vbias_list, temp_list) = get_config_lists(
        args.run,
        args.condition,
        args.values,
        args.path_raw)

    process_all_configs(
        run_number=args.run,
        kind_list=kind_list,
        vbias_list=vbias_list,
        temp_list=temp_list,
        main_data_path=args.path_raw,
        path_processed=args.path_processed
    )

    tf = datetime.datetime.now()
    print('Processing finished at: ', tf)

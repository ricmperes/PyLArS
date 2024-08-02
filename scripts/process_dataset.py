
import pylars.utils.input
import pylars.utils.output
import pylars
import datetime
import argparse

parser = argparse.ArgumentParser(
    description=('Script to process a single dataset with pylars. Checks '
                 'if a run is already saved before processing. Uses Default '
                 'processing config (sigma=5, baseline counts = 50).')
)
parser.add_argument('-t', '--temperature',
                    help='Temperatures to search for in available datasets.',
                    type=float,
                    required=True)
parser.add_argument('-v', '--voltage',
                    help='Operating voltage to search for in available datasets.',
                    type=float,
                    required=True)
parser.add_argument('-tp', '--type',
                    choices=['BV', 'DCR'],
                    help='Type (BV or DCR) to search for in available datasets.',
                    type=str,
                    required=True)
parser.add_argument('-r', '--run',
                    help='Run to consider.',
                    type=int,
                    required=True)
parser.add_argument('-a', '--amplification',
                    help='Amplification factor used.',
                    default=200,
                    type=float,
                    required=False)
parser.add_argument('-p', '--polarity',
                    help='Polarity of signal. 1 for negative, 0 for positive.',
                    type=int,
                    default=1)
parser.add_argument('-pr', '--path_raw',
                    help='Path to the main raw files directory.',
                    default='/disk/gfs_atp/xenoscope/SiPMs/char_campaign/raw_data/',
                    type=str)
parser.add_argument('-pp', '--path_processed',
                    help='Path to the processed files directory.',
                    default='/disk/gfs_atp/xenoscope/SiPMs/char_campaign/processed_data/',
                    type=str)

args = parser.parse_args()


def process_dataset(run_number: int, kind: str,
                    vbias: float, temp: float,
                    main_data_path: str, path_processed: str,
                    polarity: bool) -> None:
    # Define run
    base_run = pylars.utils.input.run(
        run_number=run_number,
        main_data_path=main_data_path,
        F_amp=args.amplification,
        signal_negative_polarity=polarity)

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
    polarity = bool(args.polarity)

    process_dataset(
        run_number=args.run,
        kind=args.type,
        vbias=args.voltage,
        temp=args.temperature,
        main_data_path=args.path_raw,
        path_processed=args.path_processed,
        polarity=polarity)

    tf = datetime.datetime.now()
    print('Processing finished at: ', tf)

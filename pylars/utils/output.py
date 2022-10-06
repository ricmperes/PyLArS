from .input import run
from pylars.processing.rawprocessor import run_processor
import pandas as pd
from glob import glob


class processed_dataset():
    """Main data type for processed data. Holds information on the
    type of dataset, the df of results and the methods for
    saving/loading cached processed files.
    """

    def __init__(self, run: run, kind: str, vbias: float,
                 temp: float, path_processed: str, process_hash: str = ''):
        self.run = run
        self.kind = kind
        self.vbias = vbias
        self.temp = temp
        self.process_hash = process_hash

        self.path_processed = path_processed

        self.hash = self.__hash__()

    def __hash__(self):
        return str(hash((self.process_hash, hash(
            (self.run.run_number, self.kind, self.vbias, self.temp)))))

    def input_data(self, results_df: pd.DataFrame) -> None:
        """Input the data pd.DataFrame to the processed_dataset object

        Args:
            results_df (pd.DataFrame): dataframe with the processed data results
        """
        self.data = results_df

    @staticmethod
    def format_kind_v_t(kind: str, v: float, t: float) -> str:
        string = f'{kind}_{t:.2f}_{v:.2f}'
        return string

    def save_data(self, type: str = 'hdf5') -> None:
        """Saves the data within the processed_dataset object
        to a file

        Args:
            type (str, optional): type of file to save the data as. Can be
            'hdf5' or 'csv'. Defaults to 'hdf5'.

        Raises:
            AssertionError: Raises the object has no data attached.
            NotImplementedError: Raises if the file format is
        different from 'hdf5' or 'csv'.
        """
        if not isinstance(self.data, pd.DataFrame):
            raise AssertionError(
                "Data not loaded to object, run input_data() first.")

        file_config = self.format_kind_v_t(self.kind, self.vbias, self.temp)

        if type == 'hdf5':
            file_name = f'{file_config}-{self.process_hash}.h5'
            file_path = f'{self.path_processed}run{self.run.run_number}/'
            self.data.to_hdf(
                file_path + file_name,
                key='data',
                complevel=5)
            print('Saved file to:', file_path + file_name)

        elif type == 'csv':
            file_name = f'{file_config}-{self.process_hash}.csv'
            file_path = f'{self.path_processed}run{self.run.run_number}/'
            self.data.to_csv(file_path + file_name)
            print('Saved file to:', file_path + file_name)

        else:
            raise NotImplementedError(f"The requested type ({type}) is not" +
                                      "implemented. Choose 'hdf5', 'csv' or make a PR.")

    def load_data(self, verbose: int = 0, force: bool = False) -> None:
        """Load cached processed data for a given processed data configuration.

        Args:
            force (bool, optional): If True and no cached file found,
        processes the raw_data (if found) with default processor
        options. Defaults to False.

        Raises:
            FileNotFoundError: If force=False and file not found.
        """
        file_path = f'{self.path_processed}run{self.run.run_number}/'
        file_config = self.format_kind_v_t(self.kind, self.vbias, self.temp)
        file_name = f'{file_path + file_config}-{self.process_hash}'

        if not force:
            try:
                try:

                    self.data = pd.read_hdf(file_name + '.h5')
                    if verbose > 0:
                        print('Loaded file: ', file_name + '.h5')

                except BaseException:
                    self.data = pd.read_csv(file_name + '.csv')
                    if verbose > 0:
                        print('Loaded file: ', file_name + '.csv')

            except BaseException:
                raise FileNotFoundError(
                    "Requested processed data not found. Process and save "
                    "with load_data(force=True) or process and save with "
                    "save_data.")

        else:

            if len(glob(file_name + '.h5')) > 0:
                self.data = pd.read_hdf(file_name + '.h5')
                if verbose > 0:
                    print('Loaded file: ', file_name + '.h5')

            else:
                processor = run_processor(run_to_process=self.run,
                                          processor_type='simple',
                                          sigma_level=5,
                                          baseline_samples=50)
                if verbose > 0:
                    print((
                        f'Using Default values for sigma '
                        f'({processor.sigma_level}) and baseline samples '
                        f'({processor.baseline_samples}) calculation.'))

                data = processor.process_datasets(
                    kind=self.kind, vbias=self.vbias, temp=self.temp)

                self.data = data

                self.save_data()
                if verbose > 0:
                    print('Processed and saved file: ', file_name + '.h5')

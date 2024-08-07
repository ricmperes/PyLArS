from glob import glob
from typing import Tuple, Union

import numpy as np
import pandas as pd
import uproot
from pylars.utils.common import load_ADC_config, get_summary_info
from pylars.utils.gsheets_db import xenoscope_db


class raw_data():
    '''
    General raw data class to define paths to raw and processed data,
    acquisition conditions, ...
    '''

    def __init__(self, raw_path: str, V: float, T: float, module: int):

        self.raw_path = raw_path
        self.tree = 't1'

        self.load_root()
        self.get_available_channels()

        self.get_n_samples()
        self.get_n_waveforms()

        self.set_general_conditions()
        self.set_specific_conditions(V, T, module)

    def set_general_conditions(self):
        '''
        Define the conditions of the data taking and of the setup
        to be propagated forward. TO DO: fetch and save in a DB
        '''
        self.ADC_range = 2.25
        self.ADC_impedance = 50
        self.F_amp = 20 * 10
        self.ADC_res = 2**14
        self.q_e = 1.602176634e-19
        self.dt = 10e-9
        self.charge_factor = self.ADC_range / self.ADC_impedance / \
            self.F_amp / self.ADC_res * self.dt / self.q_e

    def set_specific_conditions(self, V: float, T: float, module: int):
        """Sets the run specific conditions the data was taken

        Args:
            V (float): Bias voltage applied
            T (float): Temperature
        """
        self.bias_voltage = V
        self.temperature = T
        self.module = module

    def load_root(self):
        """Open the ROOT file and put in memory.
        """
        try:
            raw_file = uproot.open(self.raw_path)
            self.raw_file = raw_file
        except BaseException:
            print(f'No root file found for {self.raw_path}')

    def get_available_channels(self):
        '''
        Scans the loaded raw file for branches in tree the tree '''
        keys = self.raw_file[self.tree].keys()
        if keys[-1] == 'Time':
            keys.pop(-1)
        self.channels = keys

    def get_channel_data(self, ch: str) -> np.ndarray:
        '''
        Return the raw data array of a given channel.
        '''
        data = self.raw_file[self.tree][ch].array()  # type: ignore
        return np.array(data)

    def get_n_waveforms(self) -> int:
        """Get the number of waveforms in the root file without reading
        the whole array.

        Returns:
            int: The number of wfs in the file
        """
        first_channel = self.channels[0]

        
        n_waveforms = self.raw_file[self.tree][first_channel].num_entries # type: ignore
        self.n_waveforms = n_waveforms
        return n_waveforms

    def get_n_samples(self) -> int:
        """Get the number of samples in wach waveform in the root file
        without reading the whole array.

        Returns:
            int: The number of samples of the wfs in the file
        """
        first_channel = self.channels[0]
        
        n_samples = self.raw_file[self.tree][first_channel].interpretation.inner_shape[0] # type: ignore
        self.n_samples = n_samples
        return n_samples


class run():
    """A run is made of a collection of datasets taken at a given
    setup. Usually, opening and closing the setup defines a run.
    The datasets can be at different tmeperature and bias voltage
    conditions but the layout of the array stays the same."""

    def __init__(self, run_number: int, main_data_path: str, F_amp: float,
                 ADC_model: str = 'v1724',
                 signal_negative_polarity: bool = True):
        self.run_number = run_number
        self.main_data_path = main_data_path
        self.main_run_path = self.get_run_path()
        self.root_files = self.get_all_files_of_run()
        self.datasets = self.fetch_datasets()
        self.define_ADC_config(F_amp=F_amp, model=ADC_model)
        self.signal_negative_polarity = signal_negative_polarity

    def __repr__(self) -> str:
        repr = f'Run {self.run_number}'
        return repr

    def get_run_path(self) -> str:
        """Creates string with the run raw data directory.

        Returns:
            str: path to run raw data.
        """
        if self.run_number < 6:
            main_run_path = self.main_data_path + \
                f'run{self.run_number}/'
        else:
            main_run_path = self.main_data_path + \
                f'run{self.run_number}/data/'
        return main_run_path

    def read_layout(self):
        """Fetch the SiPM layout from a file.
        layout: dict(mod0 = dict(ch# = dict('tile': str
                                            'mppc:[###,###,...]
                                            )
                                 ch# = dict('tile': str
                                            'mppc:[###,###,...]
                                            )
                                ),
                     mod1 = dict(ch# = dict('tile': str
                                            'mppc:[###,###,...]
                                            )
                                 ch# = dict('tile': str
                                            'mppc:[###,###,...]
                                            )
                                ),
                    )

        fetch info in the form:
            which tile: layout[<module>][<channel>]['tile'] -> str
            which mppc(s): layout[<module>][<channel>]['mppc'] -> list of ints
        """
        raise NotImplementedError

    def define_ADC_config(self, F_amp: float, model: str = 'v1724') -> None:
        """Load the ADC related quantities for the run.

        Args:
        model (str): model of the digitizer
        F_amp (float): signal amplification from the sensor (pre-amp *
            external amplification on rack).
        """

        self.ADC_config = load_ADC_config(model, F_amp)

    def get_all_files_of_run(self) -> list:
        """Look for all the raw files stored for a given run.

        Returns:
            list: list of all ROOT files in the run.
        """

        all_root_files = glob(
            self.main_run_path + '**/*.root', recursive=True)
        return all_root_files

    def fetch_datasets(self) -> list:
        """Get all the datasets of a given run.

        This method needs to be adapted to the specific data storage system.

        Returns:
            list: list of all the datasets of a given run. Elements
                are type dataset.
        """
        all_root_files = self.root_files
        datasets = []
        if self.run_number == 9:
            self.root_files = []
            temps = [190, 195, 200, 205, 210]
            for t in temps:
                root_files = glob(
                    f'{self.main_data_path}{str(t)}/breakdown-v/**/*.root',
                    recursive=True)
                for path in root_files:
                    f = path.split('/')[-1].split('_')
                    if f[0] == 'test':
                        continue
                    v = float(f'{f[0]}.{f[1][:-1]}')
                    datasets.append(dataset(path, 'BV', 0, t, v))

            # LED OFF
            for t in temps:
                root_files = glob(
                    f'{self.main_data_path}{str(t)}/dcr/**/*.root',
                    recursive=True)
                for path in root_files:
                    f = path.split('/')[-1].split('_')
                    if f[0] == 'test':
                        continue
                    v = float(f'{f[0]}.{f[1][:-1]}')
                    datasets.append(dataset(path, 'DCR', 0, t, v))

        if self.run_number >= 6:

            for file in all_root_files:
                try:
                    split_file_path = file.split('/')
                    _module = int(split_file_path[-1][-8])
                    _temp = float(split_file_path[-1][-27:-24])
                    _vbias = float(
                        split_file_path[-1][-22:-17].replace('_', '.'))
                    if split_file_path[-1][0] == 'B':
                        _kind = 'BV'
                    elif split_file_path[-1][0] == 'D':
                        _kind = 'DCR'
                    elif split_file_path[-1][0] == 'f':
                        _kind = 'fplt'
                    else:
                        print('Ignoring file: ', file)
                        continue
                    datasets.append(
                        dataset(file, _kind, _module, _temp, _vbias))
                except BaseException:
                    print('Ignoring file: ', file)

        elif self.run_number == 1:
            for file in all_root_files:
                file_split = file.split('/')
                f_split = file_split[-1].split('_')
                if f_split[0] == 'test':
                    print('Ignoring test dataset: ', file)
                    continue
                if file_split[8] == 'breakdown-v':
                    _kind = 'BV'
                    _vbias = float(f_split[1] + '.' + f_split[2][:-1])
                elif file_split[8] == 'dcr':
                    _kind = 'DCR'
                    if len(f_split) == 5:
                        _vbias = float(f_split[1][:-1])
                    else:
                        _vbias = float(
                            f_split[1] + '.' + f_split[2][:-1])
                else:
                    print('Ignoring file due to unknown kind: ', file)
                    continue

                _temp = float(f_split[0][:-1])

                _module = int(f_split[-2])

                datasets.append(dataset(file, _kind, _module, _temp, _vbias))

        elif self.run_number in (2, 3):
            for file in all_root_files:
                file_split = file.split('/')
                f_split = file_split[-1].split('_')
                if f_split[0] == 'test':
                    print('Ignoring test dataset: ', file)
                    continue
                if file_split[8] == 'breakdown-v':
                    _kind = 'BV'
                    _vbias = float(f_split[1] + '.' + f_split[2][:-1])
                elif file_split[8] == 'dcr':
                    _kind = 'DCR'
                    _vbias = float(f_split[1][:-1])
                else:
                    print('Ignoring file due to unknown kind: ', file)
                    continue

                _temp = float(f_split[0][:-1])

                _module = int(f_split[-2])

                datasets.append(dataset(file, _kind, _module, _temp, _vbias))

        elif self.run_number == 4:
            for file in all_root_files:
                file_split = file.split('/')
                f_split = file_split[-1].split('_')
                if f_split[0] == 'test':
                    print('Ignoring test dataset: ', file)
                    continue
                if file_split[8] == 'breakdown-v':
                    _kind = 'BV'
                elif file_split[8] == 'dcr':
                    _kind = 'DCR'
                else:
                    print('Ignoring file due to unknown kind: ', file)
                    continue

                _vbias = float(f_split[1][:-1])
                _temp = float(f_split[0][:-1])

                _module = int(f_split[-2])

                datasets.append(dataset(file, _kind, _module, _temp, _vbias))

        elif self.run_number == 5:
            for file in all_root_files:
                file_split = file.split('/')
                f_split = file_split[-1].split('_')
                if f_split[0] == 'test':
                    print('Ignoring test dataset: ', file)
                    continue
                if file_split[8] == 'breakdown-v':
                    _kind = 'BV'
                elif file_split[8] == 'dcr':
                    _kind = 'DCR'
                else:
                    print('Ignoring file due to unknown kind: ', file)
                    continue

                _temp = float(f_split[0][:-1])
                _vbias = float(f_split[1] + '.' + f_split[2][:-1])
                _module = int(f_split[-2])

                datasets.append(dataset(file, _kind, _module, _temp, _vbias))
        else:
            raise NotImplementedError("Run not implemented yet.")
        return datasets

    def get_run_df(self) -> pd.DataFrame:
        """Get a frienly pandas dataframe with all the datasets available,
        their kind, V, T, module and path.

        Returns:
            pd.DataFrame: all the available datasets in the run.
        """
        dataset_list = self.datasets
        dicts_list = [ds.dict for ds in dataset_list]
        dataset_df = pd.DataFrame(dicts_list)
        dataset_df = dataset_df.sort_values(
            ['kind', 'temp', 'vbias', 'module'],
            ignore_index=True)
        return dataset_df


class dataset():
    """A dataset is an object with the individual setup of each
    data taking process, ie, each time the DAQ starts acquiring
    at a certain (T,V).
    """

    def __init__(self, path: str, kind: str,
                 module: int, temp: float, vbias: float):
        self.path = path
        self.kind = kind
        self.module = module
        self.temp = temp
        self.vbias = vbias
        # self.read_sizes()

        self.dict = dict(kind=self.kind,
                         module=self.module,
                         temp=self.temp,
                         vbias=self.vbias,
                         path=self.path,
                         )

    def __repr__(self) -> str:
        repr = f'{self.kind}_{self.temp}_{self.vbias}'
        return repr

    def read_sizes(self) -> Tuple[int, int]:
        """Gets the number of waveforms and number of samples per waveform
        of a given dataset.

        Brieafly creates a raw_data object of the dataset to access the ROOT
        file and read the number of entries as number of waveforms and size of
        entries as number of samples in each waveform.

        Returns:
            Tuple[int, int]: (number of waveforms, number of samples)
        """
        raw = raw_data(raw_path=self.path,
                       V=self.vbias,
                       T=self.temp,
                       module=self.module)
        raw.load_root()
        n_samples = raw.get_n_samples()
        n_waveforms = raw.get_n_waveforms()
        return n_waveforms, n_samples

    def print_config(self):
        config_print = f'''
        ---Dataset info---
        path: {self.path}
        kind: {self.kind}
        module: {self.module}
        temperature: {self.temp}
        bias voltage: {self.vbias}
        --- ---
        '''
        return config_print


class db_dataset():
    """A dataset is an object with the individual setup of each
    data taking process, ie, each time the DAQ starts acquiring
    at with a certain config. The db_dataset can point to different
    raw_files, one for each module used.
    """

    def __init__(self, run_number: str, db: Union[None, xenoscope_db]):

        if db is None:
            self.db = xenoscope_db()

        self.run_number = run_number
        self.run_dict = self.db.get_run_dict(self.run_number)
        self.path = self.run_dict['Path to remote raw data']
        self.run_type = self.run_dict['Run type']
        self.vbias = self.run_dict['SiPM bias voltage']
        # self.read_sizes()

        self.get_files()

    def get_files(self):
        files = glob(self.path + '**/*.root', recursive=True)
        self.files = files
        self.n_modules = len(files)
        # Assuming each file is a different module, i.e., the full dataset is
        # written to only 1 .root file (per module)

    def read_sizes(self) -> Tuple[int, int]:
        """Gets the number of waveforms and number of samples per waveform
        of a given dataset.

        Brieafly creates a raw_data object of the dataset to access the ROOT
        file and read the number of entries as number of waveforms and size of
        entries as number of samples in each waveform.

        Returns:
            Tuple[int, int]: (number of waveforms, number of samples)
        """
        raw = raw_data(raw_path=self.path,
                       V=0,  # mockup
                       T=0,  # mockup
                       module=0)  # mockup
        raw.load_root()
        n_samples = raw.get_n_samples()
        n_waveforms = raw.get_n_waveforms()
        return n_waveforms, n_samples

    def print_config(self):
        config_print = f'''
        ---Dataset info---
        path: {self.path}
        run_type: {self.run_type}
        number of modules found: {self.n_modules}
        bias voltage: {self.vbias}
        number of events: {self.run_dict['Total number of events']}
        --- ---
        '''
        return config_print


class xenoscope_run():
    """Main class of a Xenoscope run. It contains the necessary functions to
    find, load and process the data of muon and chntrg data taking.
    Some DB would be much better, please make a PR.
    """

    def __init__(self, run_number: int, main_data_path: str, F_amp: float,
                 ADC_model: str = 'v1724',
                 signal_negative_polarity: bool = True):
        self.run_number = run_number
        self.main_data_path = main_data_path

        self.root_files = self.get_all_files_of_run()
        self.files_df, self.files_fail = self.get_all_datasets()

        self.define_ADC_config(F_amp=F_amp, model=ADC_model)
        self.signal_negative_polarity = signal_negative_polarity
        self.load_labels()

    def __repr__(self) -> str:
        repr = f'Run {self.run_number}'
        return repr

    def get_all_files_of_run(self) -> list:
        """Look for all the raw files stored for a given run.

        Returns:
            list: list of all ROOT files in the run.
        """

        all_root_files = glob(
            self.main_data_path + '/**/*.root', recursive=True)
        return all_root_files

    def get_run_info(self, file):
        file_name = file.split('/')[-1]
        specific_name = file_name[:-16]
        run_type = specific_name.split('_')[0]
        summary_file = f'{self.main_data_path}/{specific_name}/Summary_{specific_name}.txt'

        end, duration, n_events = get_summary_info(summary_file)

        start = end - duration

        module = file_name[-8]

        info_dict = {'start': start, 'end': end,
                     'duration': duration,
                     'run_type': run_type,
                     'n_events': int(n_events),
                     'module': int(module), 'path': file}

        return info_dict

    def get_all_datasets(self) -> Union[pd.DataFrame, Tuple]:
        files_df = pd.DataFrame(columns=['start', 'end', 'duration',
                                         'run_type', 'n_events',
                                         'module', 'path'])
        files_fail = []
        for file in self.root_files:
            try:
                files_df = files_df.append(self.get_run_info(file),
                                           ignore_index=True)  # type: ignore
            except FileNotFoundError:
                files_fail.append(file)

        files_df.sort_values(by=['start', 'module'], inplace=True,
                             ignore_index=True)
        return files_df, files_fail

    def define_ADC_config(self, F_amp: float, model: str = 'v1724') -> None:
        """Load the ADC related quantities for the run.

        Args:
        model (str): model of the digitizer
        F_amp (float): signal amplification from the sensor (pre-amp *
            external amplification on rack).
        """

        self.ADC_config = load_ADC_config(model, F_amp)

    def load_labels(self) -> None:
        """Define the labeling of all the channels and corresponding tiles.
        """
        labels_complete = {'mod0': {'wf1': 'wf1 | Tile H',
                                    'wf2': 'wf2 | Tile J',
                                    'wf3': 'wf3 | Tile K',
                                    'wf4': 'wf4 | Tile L',
                                    'wf5': 'wf5 | Tile M',
                                    'wf6': 'wf6 | Muon detector 1',
                                    'wf7': 'wf7 | Muon detector 2'},
                           'mod1': {'wf1': 'wf1 | Tile A',
                                    'wf2': ' wf2 | Tile B',
                                    'wf3': 'wf3 | Tile C',
                                    'wf4': 'wf4 | Tile D',
                                    'wf5': 'wf5 | Tile E',
                                    'wf6': 'wf6 | Tile F',
                                    'wf7': 'wf7 | Tile G'}
                           }
        labels_tiles = {'mod0': {'wf1': 'H', 'wf2': 'J', 'wf3': 'K',
                                 'wf4': 'L', 'wf5': 'M',
                                 'wf6': 'Muon detector 1',
                                 'wf7': 'Muon detector 2'},
                        'mod1': {'wf1': 'A', 'wf2': 'B', 'wf3': 'C',
                                 'wf4': 'D', 'wf5': 'E', 'wf6': 'F',
                                 'wf7': 'G'}
                        }

        self.labels_complete = labels_complete
        self.labels_tiles = labels_tiles

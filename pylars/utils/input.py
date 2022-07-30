from glob import glob
import numpy as np
import pandas as pd
import uproot


class raw_data():
    '''
    General raw data class to define paths to raw and processed data,
    acquisition conditions, ...
    '''

    def __init__(self, raw_path: str, V: float, T: float, module: int):

        self.raw_path = raw_path
        self.tree = 't1'

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
        try:
            raw_file = uproot.open(self.raw_path)
            self.raw_file = raw_file
        except Exception:
            raise f'No root file found for {self.raw_path}'

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
        data = self.raw_file[self.tree][ch].array()
        return np.array(data)


class run():
    """A run is made of a collection of datasets taken at a given
    setup. Usually, opening and closing the setup defines a run.
    The datasets can be at different tmeperature and bias voltage
    conditions but the layout of the array stays the same."""

    def __init__(self, run_number, main_data_path):
        self.run_number = run_number
        self.layout = self.read_layout()
        self.main_data_path = main_data_path
        self.main_run_path = self.main_data_path + \
            f'run{self.run_number}/data/'

        self.root_files = self.get_all_files_of_run()
        self.datasets = self.fetch_datasets()

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
        pass

    def get_all_files_of_run(self) -> list:
        all_root_files = glob(self.main_run_path + '**/*.root', recursive=True)
        return all_root_files

    def fetch_datasets(self) -> list:
        all_root_files = self.root_files
        datasets = []
        for i, file in enumerate(all_root_files):
            split_file_path = file.split('/')
            _module = int(split_file_path[-1][-8])
            _temp = float(split_file_path[-1][-27:-24])
            _vbias = float(split_file_path[-1][-22:-17].replace('_', '.'))
            _kind = ('BV' if split_file_path[-1][0] == 'B' else 'DCR')

            datasets.append(dataset(file, _kind, _module, _temp, _vbias))

        return datasets

    def get_run_df(self) -> pd.DataFrame:
        dataset_list = self.datasets
        dicts_list = [ds.dict for ds in dataset_list]
        dataset_df = pd.DataFrame(dicts_list)
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

        self.dict = dict(kind=self.kind,
                         module=self.module,
                         temp=self.temp,
                         vbias=self.vbias,
                         path=self.path)

    def __repr__(self) -> str:
        repr = f'{self.kind}_{self.temp}_{self.vbias}'
        return repr

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

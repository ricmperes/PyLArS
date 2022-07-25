import awkward
import numpy as np
import uproot


class raw_data():
    '''
    General raw data class to define paths to raw and processed data,
    acquisition conditions, ...
    '''

    def __init__(self, raw_path: str, V: float, T: float):

        self.raw_path = raw_path
        self.tree = 't1'

        self.set_general_conditions()
        self.set_specific_conditions(V, T)

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

    def set_specific_conditions(self, V: float, T: float):
        """Sets the run specific conditions the data was taken

        Args:
            V (float): Bias voltage applied
            T (float): Temperature
        """
        self.bias_voltage = V
        self.temperature = T

    def load_root(self):
        try:
            raw_file = uproot.open(self.raw_path)
            self.raw_file = raw_file
        except Exception:
            raise f'No root file found for {self.raw_path}'

    def get_available_channels(self):
        '''
        Scans the loaded raw file for branches in tree the tree '''
        self.channels = self.raw_file[self.tree].keys()

    def get_channel_data(self, ch):
        '''
        Return the raw data array of a given channel.
        '''

        return self.raw_file[self.tree][ch].array()

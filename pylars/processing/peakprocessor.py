import csv
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

from pylars.utils.input import raw_data, run
from pylars.utils.common import get_deterministic_hash, load_ADC_config
from .waveforms import waveform_processing
from .peaks import peak_processing

class peak_processor():
    """Define the functions for a simple peak processor.

    Defines a processor object to process waveforms from summing all the 
    channels in the photosensor array.
    """

    version = '0.0.1'
    processing_method = 'peaks_simple'

    # for run6
    index_reorder_data_to_layout = [ 9, 4, 5, 6, 11, 10, 8, 0, 2, 3, 7, 1]

    def __init__(self, sigma_level: float, baseline_samples: int, 
                 gains_tag: str , gains_path: str) -> None:

        self.sigma_level = sigma_level
        self.baseline_samples = baseline_samples
        self.hash = get_deterministic_hash(f"{self.processing_method}" +
                                           f"{self.version}" +
                                           f"{self.sigma_level}" +
                                           f"{self.baseline_samples:.2f}")
        self.processed_data = dict()
        self.show_loadbar_channel = True
        self.show_tqdm_channel = True

        self.gains_tag = gains_tag
        self.gains_path = gains_path
        """Path to where the gains are saved. 
        
        In the future this should be defined in a config file.
        """
        self.gains = self.load_gains()
        """Dictionary of gains [e/pe] for each channel.
        """

    def __hash__(self) -> str:
        return self.hash

    def load_gains(self) -> np.ndarray:
        """Load gains of photosensors based on the defined path and tag.

        Returns:
            np.ndarray: array with channel gains [e/pe]) in ascending order (mo
                dule->channel) 
        """

        # csv is extremely fast but pandas is easier to load and sort 
        # right away

        # with open(self.gains_path + self.gains_tag, mode='r') as file:
        #     reader = csv.reader(file)
        #     gains = {rows[0]:float(rows[1]) for rows in reader}

        # return gains
        
        gain_df = pd.read_csv(self.gains_path + self.gains_tag,
                              header=None, 
                              names=['module', 'channel','gain'])
        gain_df = gain_df.sort_values(['module','channel'], ignore_index=True)

        return np.array(gain_df['gain'])

    def set_tqdm_channel(self, bar: bool, show: bool):
        """Change the tqdm config

        Args:
            bar (bool): show or not the tqdm bar.
            show (bool): use tqdm if true, disable if false
        """
        self.show_loadbar_channel = bar
        self.show_tqdm_channel = show

    def define_ADC_config(self, F_amp: float, model: str = 'v1724') -> None:
        """Load the ADC related quantities for the dataset.

        Args:
        model (str): model of the digitizer
        F_amp (float): signal amplification from the sensor (pre-amp *
            external amplification on the rack).
        """

        self.ADC_config = load_ADC_config(model, F_amp)

    def process_waveform_set(self, waveforms:np.ndarray, 
                             return_waveforms = False):
        """Process an array of waveforms from all channels.

        The waveforms are assumed to be synchronized and each row of the 
        array is a channel. Once a summed waveform is formed, uses the same
        functions as the pulse processing to find peaks and compute its
        properties.

        Args:
            waveforms (np.ndarray): waveforms of all channels stacked.
        """

        baselines = np.apply_along_axis(
            func1d = waveform_processing.get_baseline_rough,
            axis = 1,
            arr = waveforms,
            baseline_samples = self.baseline_samples)

        waveforms_pe = peak_processing.apply_waveforms_transform(
            waveforms, baselines, self.gains, self.ADC_config)
        sum_waveform = peak_processing.get_sum_waveform(waveforms_pe)

        areas, lengths, positions, amplitudes = waveform_processing.process_waveform(
                    sum_waveform, self.baseline_samples, self.sigma_level)

        if return_waveforms is True:
            return waveforms_pe, sum_waveform

        return areas, lengths, positions, amplitudes



    def load_raw_data(self, path_to_raw: str, V: float, T: float, module: int):
        """Raw data loader to pass to the processing scripts.

        Args:
            path_to_raw (str): _description_
            V (float): _description_
            T (float): _description_

        Returns:
            raw_data: _description_
        """
        raw = raw_data(path_to_raw, V, T, module)
        raw.load_root()
        raw.get_available_channels()

        self.raw_data = raw

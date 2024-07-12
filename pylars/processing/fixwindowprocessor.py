import numpy as np
import pandas as pd
from pylars.utils.common import get_deterministic_hash
from pylars.utils.input import raw_data, run
from tqdm import tqdm
from typing import Tuple, Union

from .fixwindow import fixed_window_processing


class window_processor():
    """Define the processor for LED window analysis.
    """

    version = '0.0.1'
    
    def __init__(self, 
                 led_window: Tuple[int, int], 
                 baseline_samples: int) -> None:
        self.baseline_samples = baseline_samples
        self.led_window = led_window
        self.hash = get_deterministic_hash(f"{self.version}" +
                                           f"{self.led_window}" +
                                           f"{self.baseline_samples:.2f}")
        self.processed_data = dict()
        self.show_loadbar_channel = True
        self.show_tqdm_channel = True

    def __hash__(self) -> str:
        return self.hash
    
    def set_tqdm_config(self, bar: bool, show: bool):
        """Change the tqdm config

        Args:
            bar (bool): show or not the tqdm bar.
            show (bool): use tqdm if true, disable if false
        """
        self.show_loadbar_channel = bar
        self.show_tqdm_channel = show

    # General functions for I/O
    def load_raw_data(self, path_to_raw: str, module: int):
        """Raw data loader to pass to the processing scripts.

        Args:
            path_to_raw (str): _description_

        Returns:
            raw_data: the raw data object
        """
        raw = raw_data(raw_path = path_to_raw, V = -1, 
                       T = -1, module = module)
        raw.load_root()
        raw.get_available_channels()

        self.raw_data = raw

    # Processing functions
    def process_channel(self, ch: str) -> dict:
        """Process a channel by iterating over all its waveforms and 
        calculating the following:
            - baseline
            - integral in the LED window
            - max amplitude in the LED window

        Args:
            ch (str): channel name as in the ROOT file.
                In files from DAQ_zero/XenoDAQ these will be 'wf#' with #
                the number of the channel [0,7]

        Returns:
            dict: Dictionary of keys module, channel, wf_number, led_amplitude, 
            led_area, led_ADCcounts in led window of the processed waveforms.
        """
        if ch not in self.raw_data.channels:
            raise AssertionError(
                f'The requested channel is not available. '
                f'Loaded channels:{self.raw_data.channels}')

        module = self.raw_data.module
        channel_data = self.raw_data.get_channel_data(ch)

        amplitudes, areas, ADCcounts = fixed_window_processing.process_all_waveforms(
            channel_data, self.baseline_samples, self.led_window)
        
        module_number = [module] * len(areas)
        ch_name = [ch] * len(areas)
        wf_number = np.arange(len(areas))
            
        results = {'module': module_number,
                   'channel': ch_name,
                   'wf_number':wf_number,
                   'led_amplitude': amplitudes,
                   'led_area': areas,
                   'led_ADCcounts': ADCcounts,
                   }

        return results
    
    def process_all_channels(self) -> pd.DataFrame:
        """Calls the process_channel method of each of
        the available channels in the dataset.

        Returns:
            pd.DataFrame: Results for all the channels of a dataset.
        """
        channels_to_process = self.raw_data.channels
        results_list = [pd.DataFrame(self.process_channel(
            _channel)) for _channel in channels_to_process]
        results_df = pd.concat(results_list, ignore_index=True)

        return results_df
    
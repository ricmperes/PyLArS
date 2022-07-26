import numpy as np
import pandas as pd
from tqdm import tqdm

from pylars.utils.input import raw_data
from .waveforms import waveform_processing


class simple_processor():
    """Define the atributes and functions for a simple processor.
    """

    version = '0.0.1'
    processing_method = 'simple'

    def __init__(self, sigma_level: float, baseline_samples: int):
        self.sigma_level = sigma_level
        self.baseline_samples = baseline_samples
        self.hash = hash(
            f"{self.processing_method}{self.sigma_level}{self.baseline_samples}")
        self.processed_data = dict()
        self.show_loadbar = True
        self.disable_tqdm = False

    def set_tqdm(self, bar: bool, disable: bool):
        """Change the tqdm config

        Args:
            bar (bool): shwo or not the tqdm bar.
            disable (bool): disable tqdm use intirely
        """
        self.show_loadbar = bar
        self.disable_tqdm = disable

    # General functions for I/O
    def load_raw_data(self, path_to_raw: str, V: float, T: float):
        """Raw data loader to pass to the processing scripts.

        Args:
            path_to_raw (str): _description_
            V (float): _description_
            T (float): _description_

        Returns:
            raw_data: _description_
        """
        raw = raw_data(path_to_raw, V, T)
        raw.load_root()
        raw.get_available_channels()

        self.raw_data = raw

    # Processing functions
    def process_channel(self, ch: str) -> dict:
        assert ch in self.raw_data.channels, f'The requested channel is not available. Loaded channels:{self.raw_data.channels}'

        channel_data = self.raw_data.get_channel_data(ch)
        # converted before to np to speed up ahead
        #channel_data = np.array(channel_data)
        results = {'channel': [],
                   'wf_number': [],
                   'peak_number': [],
                   'area': [],
                   'length': [],
                   'position': []
                   }
        if self.show_loadbar:
            total = len(channel_data)
        else:
            total = None

        for i, _waveform in tqdm(enumerate(
                channel_data), disable=self.disable_tqdm, total=total, desc=f'Processing channel {ch}'):
            try:
                areas, lengths, positions = waveform_processing.process_waveform(
                    _waveform, self.baseline_samples, self.sigma_level)

                assert len(areas) == len(positions) == len(lengths)
                
                #check if any peaks were found
                if len(areas) == 0:
                    continue

                ch_name = [ch] * len(areas)
                wf_number = np.ones(len(areas), dtype=int) * i
                peak_number = np.arange(len(areas))

                results['channel'] += list(ch_name)
                results['wf_number'] += list(wf_number)
                results['peak_number'] += list(peak_number)
                results['area'] += list(areas)
                results['length'] += list(lengths)
                results['position'] += list(positions)

            except Exception:
                print('Ups! There was a problem on iteration number: ', i)

        return results

    def process_all_channels(self) -> pd.DataFrame:
        results_list = []
        for _channel in self.raw_data.channels:
            results_ch = self.process_channel(_channel)
            results_list.append(pd.DataFrame(results_ch))
        results_df = pd.concat(results_list, ignore_index=True)

        return results_df


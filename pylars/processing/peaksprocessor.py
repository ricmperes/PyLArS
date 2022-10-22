import csv
import numpy as np
import pandas as pd
from tqdm import tqdm

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

    def __hash__(self) -> int:
        return self.hash

    def load_gains(self) -> np.array:
        """Load gains of photosensors based on the defined path and tag.

        Returns:
            np.array: array with channel gains [e/pe]) in ascending order (mo
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

        return gain_df['gain'].values

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

    def process_waveform_set(self, waveforms:np.array):
        """Process an array of waveforms from all channels.

        The waveforms are asusmed to be synchronized and each row of the 
        array is a channel. Once a summed waveform is formed, uses the same
        functions as the pulse processing to find peaks and compute its
        properties.

        Args:
            waveforms (np.array): waveforms of all channels stacked.
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

    # Processing functions
    def process_channel(self, ch: str) -> dict:
        """Process a channel by iterating over all its waveforms, running
    the pulse finding algorithm and the calculating the following pulse
    properties: areas, lengths, positions and amplitudes.

        Args:
            ch (str): channel name as in the ROOT file.
        In files from DAQ_zero/XenoDAQ these will be 'wf#' with # the
        number of the channel [0,7]

        Raises:
            AssertionError: if the requested channel
        is not available on the raw file.
            AssertionError: if the there was a problem
        in the processing of a waveform

        Returns:
            dict: Dictionary of keys module, channel, wf_number
        area, length, position where the values are lists (order
        matters) of the processed waveforms.
        """
        if ch not in self.raw_data.channels:
            raise AssertionError(
                f'The requested channel is not available. '
                f'Loaded channels:{self.raw_data.channels}')

        module = self.raw_data.module
        channel_data = self.raw_data.get_channel_data(ch)
        # converted before to np to speed up ahead
        #channel_data = np.array(channel_data)
        results = {'module': [],
                   'channel': [],
                   'wf_number': [],
                   'pulse_number': [],
                   'area': [],
                   'length': [],
                   'position': [],
                   'amplitude': [],
                   }
        if self.show_loadbar_channel:
            total = len(channel_data)
        else:
            total = None

        for i, _waveform in tqdm(enumerate(channel_data),
                                 disable=(not self.show_tqdm_channel),
                                 total=total,
                                 desc=(f'Processing module {module} '
                                       f'channel {ch}')
                                 ):
            try:
                areas, lengths, positions, amplitudes = waveform_processing.process_waveform(
                    _waveform, self.baseline_samples, self.sigma_level)

                assert len(areas) == len(positions) == len(
                    lengths) == len(amplitudes)

                # check if any pulses were found
                if len(areas) == 0:
                    continue

                module_number = [module] * len(areas)
                ch_name = [ch] * len(areas)
                wf_number = np.ones(len(areas), dtype=int) * i
                pulse_number = np.arange(len(areas))

                results['module'] += module_number
                results['channel'] += ch_name
                results['wf_number'] += list(wf_number)
                results['pulse_number'] += list(pulse_number)
                results['area'] += list(areas)
                results['length'] += list(lengths)
                results['position'] += list(positions)
                results['amplitude'] += list(amplitudes)

            except Exception:
                raise AssertionError(
                    'Ups! There was a problem on iteration number: ', i)

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

    def purge_processor(self):
        del self.raw_data
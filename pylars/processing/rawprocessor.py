import base64
import hashlib
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from pylars.utils.input import raw_data, run

from .waveforms import waveform_processing


class simple_processor():
    """Define the atributes and functions for a simple processor.
    """

    version = '0.0.2'
    processing_method = 'simple'

    def __init__(self, sigma_level: float, baseline_samples: int):
        self.sigma_level = sigma_level
        self.baseline_samples = baseline_samples
        self.hash = self.get_deterministic_hash(f"{self.processing_method}" +
                                                f"{self.version}" +
                                                f"{self.sigma_level}" +
                                                f"{self.baseline_samples:.2f}")
        self.processed_data = dict()
        self.show_loadbar_channel = True
        self.show_tqdm_channel = True

    def __hash__(self) -> int:
        return self.hash

    @staticmethod
    def get_deterministic_hash(id: str) -> str:
        """Return a base32 lowercase string of length determined from hashing
        the configs. Based on https://github.com/AxFoundation/strax/blob/
        156254287c2037876a7040460b3551d590bf5589/strax/utils.py#L303

        Args:
            id (str): thing to hash

        Returns:
            str: hashed version of the thing
        """
        jsonned = json.dumps(id)
        digest = hashlib.sha1(jsonned.encode('ascii')).digest()
        readable_hash = base64.b32encode(digest)[:7].decode('ascii').lower()
        return readable_hash

    def set_tqdm_channel(self, bar: bool, show: bool):
        """Change the tqdm config

        Args:
            bar (bool): shwo or not the tqdm bar.
            show (bool): use tqdm if true, disable if false
        """
        self.show_loadbar_channel = bar
        self.show_tqdm_channel = show

    # General functions for I/O
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
        """_summary_

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
                   'peak_number': [],
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

                assert len(areas) == len(positions) == len(lengths) == len(amplitudes)

                # check if any peaks were found
                if len(areas) == 0:
                    continue

                module_number = [module] * len(areas)
                ch_name = [ch] * len(areas)
                wf_number = np.ones(len(areas), dtype=int) * i
                peak_number = np.arange(len(areas))

                results['module'] += module_number
                results['channel'] += ch_name
                results['wf_number'] += list(wf_number)
                results['peak_number'] += list(peak_number)
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
        the availabel channels in the dataset.

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


class run_processor(simple_processor):
    """The 'run_processor' extends the use of the 'simple_processor'
    to full run, ie, a set of datasets taken at different operating
    conditions but with the same setup.

    Args:
        simple_processor (simple_processor): super class with
    dataset-level processing methods.
    """

    def __init__(self, run_to_process: run, processor_type: str,
                 sigma_level: float, baseline_samples: int):
        if not isinstance(run_to_process, run):
            raise TypeError("Needs run type object.")
        if processor_type != 'simple':
            raise NotImplementedError(
                "Only 'simple' is available. Please make a PR to add more.")

        self.run = run_to_process

        super().__init__(sigma_level, baseline_samples)

        self.datasets_df = self.run.get_run_df()
        self.show_loadbar_run = True
        self.show_tqdm_run = True
        self.show_loadbar_channel = True
        self.show_tqdm_channel = True

    def set_tqdm_run(self, bar: bool, show: bool):
        """Define the use of tqdm for run level processing.

        Args:
            bar (bool): show tqdm bar
            show (bool): use tqdm or not
        """
        self.show_loadbar_run = bar
        self.show_tqdm_run = show

    def print_tqdm_options(self):
        print(f'show bar channel:{self.show_loadbar_channel}\n' +
              f'show tqdm channel:{self.show_tqdm_channel}\n' +
              f'show bar run:{self.show_loadbar_run}\n' +
              f'show tqdm run:{self.show_tqdm_run}')

    def process_datasets(self, kind: str, vbias: float,
                         temp: float) -> pd.DataFrame:
        """Runs the loaded processor through a full dataset, ie,
        Processes all the channels of all the boards for a set of
        given operating conditions (kind, vbias, temp).

        Args:
            kind (str): type of data ('BV' for breakdown voltage/LED
        ON, 'DCR' for dark count rate data/LED OFF.)
            vbias (float): bias voltage applied
            temp (float): temperature of the setup

        Returns:
            pd.DataFrame: processed data with computed area, length
        and position, retaining info on module, channel and waveform
        of the peak.
        """

        selection = ((self.datasets_df['kind'] == kind) &
                     (self.datasets_df['vbias'] == vbias) &
                     (self.datasets_df['temp'] == temp))

        datasets_to_process = self.datasets_df[selection]

        if len(datasets_to_process) == 0:
            # prints screw up tqdm and are not that useful
            # #print(
            #    f'No datasets found on run with kind = {kind}, '
            #    f'voltage = {vbias} and temperature = {temp}.')
            return None

        print(
            f'Found {len(datasets_to_process)} datasets. '
            f'Ramping up processor!')

        if self.show_loadbar_run:
            total = len(datasets_to_process)
        else:
            total = None

        results = []

        for dataset in tqdm(datasets_to_process.itertuples(), 
                            'Loading and processing datasets: ', 
                            total=total, 
                            disable=(not self.show_tqdm_run)):

            self.load_raw_data(path_to_raw=dataset.path,
                               V=dataset.vbias,
                               T=dataset.temp,
                               module=dataset.module)
                               
            # this returns a pd.DataFrame
            _results_of_dataset = self.process_all_channels()  
            results.append(_results_of_dataset)
            self.purge_processor()

        results = pd.concat(results, ignore_index=True)

        return results

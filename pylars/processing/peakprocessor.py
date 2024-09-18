
import numpy as np
import pandas as pd
from tqdm import tqdm
from pylars.utils.common import get_deterministic_hash, load_ADC_config
from pylars.utils.input import raw_data

from .peaks import peak_processing
from .waveforms import waveform_processing


class peak_processor():
    """Define the functions for a simple peak processor.

    Defines a processor object to process waveforms from summing all the
    channels in the photosensor array.
    """

    version = '0.0.1'
    processing_method = 'peaks_simple'

    # for run6
    index_reorder_channels = [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]
    list_of_tiles = [
        'A',
        'B',
        'C',
        'D',
        'E',
        'F',
        'G',
        'H',
        'J',
        'K',
        'L',
        'M']

    def __init__(self, sigma_level: float, baseline_samples: int,
                 gains_tag: str, gains_path: str, layout_path: str) -> None:

        self.sigma_level = sigma_level
        self.baseline_samples = baseline_samples
        self.hash = get_deterministic_hash(f"{self.processing_method}" +
                                           f"{self.version}" +
                                           f"{self.sigma_level}" +
                                           f"{self.baseline_samples:.2f}")
        self.show_loadbar_channel = True
        self.show_tqdm_channel = True

        self.gains_tag = gains_tag
        self.gains_path = gains_path
        """Path to where the gains and layout are saved.

        In the future this should be defined in a config file.
        """
        self.gains = self.load_gains()
        self.load_layout(layout_path)
        """Dictionary of gains [e/pe] for each channel.
        """

    def __hash__(self) -> str:
        return self.hash

    def load_layout(self, path: str) -> None:
        """Load the layout of the photosensor array.

        Args:
            path (str): path to the layout file.
        """
        array_layout = np.loadtxt(path)
        array_labels = ['A', 'B', 'C', 'D', 'E', 'F',
                        'G', 'H', 'J', 'K', 'L', 'M']
        layout_centers_x = (array_layout[:,0] + array_layout[:,1])/2
        layout_centers_y = (array_layout[:,2] + array_layout[:,3])/2
        array_centers = np.vstack([layout_centers_x, layout_centers_y])

        self.array_layout = array_layout
        self.array_centers = array_centers
        self.array_labels = array_labels

        
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

        gain_df = pd.read_csv(f'{self.gains_path}/gains_{self.gains_tag}.csv')
        gain_df = gain_df.sort_values(by=['tile'],
                                      ignore_index=True)
        if gain_df.iloc[0]['gain'] < 1e4:
            gain_df['gain'] = gain_df['gain'] * 1e6

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

    def load_raw_data(self, path_to_raw_both_modules: list,
                      modules: list, V: float, T: float,
                      ignore_channels: list = []):
        """Raw data loader to pass to the processing scripts.

        Args:
            path_to_raw_both_modules (list): list with path to both raw files,
                one for each module. If len(pat_list) = 1, then it assumes
                one only module.
            V (float): _description_
            T (float): _description_
            ignore_channels (list): list of channels to ignore. Each element
                should be a list corresponding to each module, in order.

        Returns:
            raw_data: _description_
        """

        assert len(path_to_raw_both_modules) == len(modules) == len(
            ignore_channels), "Number of paths and modules needs to be the same."

        self.raw_data_list = []
        for module, path_to_raw in zip(modules, path_to_raw_both_modules):

            raw = raw_data(path_to_raw, V, T, module)
            raw.load_root()
            raw.get_available_channels()
            raw.channels = [ch for ch in raw.channels if
                            ch not in ignore_channels[module]]

            self.raw_data_list.append(raw)

    def get_stacked_waveforms(self):
        """Get the waveforms from all channels stacked.
        FROM HERE THE WAVEFORMS ARE STACKED IN THE ORDER OF THE TILES,
        FROM A TO M!


        Returns:
            np.ndarray: waveforms of all channels stacked.
        """
        _stacked_channel_data_list = []
        for raw in self.raw_data_list:
            _stacked_channel_data_list.append(np.stack(
                [raw.get_channel_data(ch) for ch in raw.channels],
                axis=0))
        stacked_waveforms = np.concatenate(_stacked_channel_data_list, axis=0)
        stacked_waveforms = stacked_waveforms[self.index_reorder_channels]

        return stacked_waveforms

    def make_sum_waveforms_all_channels(self):

        waveforms = self.get_stacked_waveforms()

        baselines = np.apply_along_axis(
            func1d=waveform_processing.get_baseline_rough,
            axis=2,
            arr=waveforms,
            baseline_samples=50)

        stds = np.apply_along_axis(
            func1d=waveform_processing.get_std_rough,
            axis=2,
            arr=waveforms,
            baseline_samples=50)

        waveforms_pe = peak_processing.apply_waveforms_transform(
            waveforms, baselines, self.gains, self.ADC_config)

        sum_waveform = peak_processing.get_sum_waveform(waveforms_pe)

        return waveforms_pe, sum_waveform

    def process_waveform_set(self, waveforms_pe_single: np.ndarray,
                             sum_waveform_single: np.ndarray):
        """Process an array of waveforms from all channels.

        The waveforms are assumed to be synchronized and each row of the
        array is a channel. Once a summed waveform is formed, uses the same
        functions as the pulse processing to find peaks and compute its
        properties.

        Args:
            waveforms (np.ndarray): waveforms of all channels stacked.
        """

        areas, lengths, positions, amplitudes = waveform_processing.process_waveform(
            waveform=sum_waveform_single,
            baseline_samples=self.baseline_samples,
            sigma_level=self.sigma_level,
            negative_polarity=False,
            baseline_subtracted=True)

        areas_individual_channels = [peak_processing.get_area_of_single_waveform_each_channel(
            waveforms_pe_single, positions[i], positions[i] + lengths[i]) for i in range(len(positions))]

        return areas, lengths, positions, amplitudes, areas_individual_channels

    def process_all_waveforms(self):

        waveforms_pe, sum_waveform = self.make_sum_waveforms_all_channels()

        n_wfs = sum_waveform.shape[0]

        results = {'wf_number': [],
                   'peak_number': [],
                   'n_peaks': [],
                   'area': [],
                   'length': [],
                   'position': [],
                   'amplitude': [],
                   'rec_x': [],
                   'rec_y': [],
                   }

        areas_individual_channels_results = {
            'wf_number': [],
            'peak_number': [],
            'A': [],
            'B': [],
            'C': [],
            'D': [],
            'E': [],
            'F': [],
            'G': [],
            'H': [],
            'J': [],
            'K': [],
            'L': [],
            'M': [],
        }

        for wf_i in tqdm(range(n_wfs), total=n_wfs,
                         desc='Processing waveforms: '):

            _sum_waveform_single = sum_waveform[wf_i, :]
            _waverform_pe_single = waveforms_pe[:, wf_i, :]

            areas, lengths, positions, amplitudes, areas_individual_channels = self.process_waveform_set(
                waveforms_pe_single=_waverform_pe_single,
                sum_waveform_single=_sum_waveform_single)

            wf_number = np.ones(len(areas), dtype=int) * wf_i
            peak_number = np.arange(len(areas))
            n_pulses = np.ones(len(areas), dtype=int) * len(areas)

            results['wf_number'] += list(wf_number)
            results['peak_number'] += list(peak_number)
            results['n_peaks'] += list(n_pulses)
            results['area'] += list(areas)
            results['length'] += list(lengths)
            results['position'] += list(positions)
            results['amplitude'] += list(amplitudes)

            cog_pos = peak_processing.reconstruct_xy_position(
                area_per_sensor=np.array(areas_individual_channels),
                sensor_layout=self.array_centers)

            results['rec_x'] += list(cog_pos[0,:])
            results['rec_y'] += list(cog_pos[1,:])


            areas_individual_channels_results['wf_number'] += list(wf_number)
            areas_individual_channels_results['peak_number'] += list(
                peak_number)

            for peak_i in range(len(areas)):
                for tile_i, tile_name in enumerate(self.list_of_tiles):
                    areas_individual_channels_results[tile_name] += [
                        areas_individual_channels[peak_i][tile_i]]

        results = pd.DataFrame(results)
        areas_individual_channels_results = pd.DataFrame(
            areas_individual_channels_results)
        self.results_df = results
        self.areas_individual_channels_results_df = areas_individual_channels_results
        return results, areas_individual_channels_results

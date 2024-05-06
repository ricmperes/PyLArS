"""This file contains classes and methods to conduct an analysis of an LED OFF
dataset/run. It is divided into:
    - `DCR_analysis`: the collection of methods required for the analysis (fits,
        DCR, CTP, gain calculation, etc.)
    - `DCR_dataset`: the object of a LED OFF dataset, i.e., given temp, given
        module and given channel, several bias voltages.
    - `DCR_run`: the object to collect and handle all the `DCR_dataset`
        objects of a run.
"""

import copy
import time
from typing import Tuple, Union, Optional

import numba
import numpy as np
import pandas as pd
import pylars
import pylars.plotting.plotanalysis
import pylars.utils.common
import pylars.utils.input
import pylars.utils.output
import scipy.interpolate as itp
from pylars.analysis.breakdown import compute_BV_df
from pylars.utils.common import Gaussian, get_gain
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from tqdm.autonotebook import tqdm


class DCR_analysis():
    """A class with all the methods needed for DCR analysis.

    Why a separate class? DCR_dataset was too messy with @classmethods.
    These we be put in this parent class and DCR_dataset and DCR_run will
    have only object-specific methods.
    """

    __version__ = 'v0.0.1'

    def __init__(self):
        pass

    @classmethod
    def get_1pe_value_fit(cls,
                          df: pd.DataFrame,
                          length_cut_min: int = 5,
                          length_cut_max: int = 80,
                          plot: Union[bool, str] = False,
                          use_scipy_find_peaks: bool = False) -> tuple:
        """Try to fit the SPE peak in the area histogram and return the
        Gaussian paramenters.

        Args:
            df (pd.DataFrame): _description_
            length_cut (int, optional): cut to impose on the length of
                        the peaks for noise suppression. Defaults to 5.
            plot (boolorstr, optional): _description_. Defaults to False.

        Returns:
            tuple: (A, mu, sigma), cov
        """

        (area_hist_x, DCR_values, DCR_der_x_points,
         DCR_der_y_points, min_area_x) = cls.get_1pe_rough(
            df, length_cut_min, length_cut_max,
            use_scipy_find_peaks=use_scipy_find_peaks)

        if plot != False:
            pylars.plotting.plotanalysis.plot_DCR_curve(
                plot, area_hist_x, DCR_values, DCR_der_x_points,
                DCR_der_y_points, min_area_x)

        area_hist = np.histogram(df[(df['length'] > length_cut_min) &
                                    (df['length'] < length_cut_max)]['area'],
                                 bins=np.linspace(0.5 * min_area_x, 1.5 * min_area_x, 300))
        area_hist_x = area_hist[1]
        area_hist_x = (
            area_hist_x + (area_hist_x[1] - area_hist_x[0]) / 2)[:-1]
        area_hist_y = area_hist[0]

        (A, mu, sigma), cov = curve_fit(Gaussian, area_hist_x, area_hist_y,
                                        p0=(2000, min_area_x, 0.05 * min_area_x))

        if plot != False:
            pylars.plotting.plotanalysis.plot_SPE_fit(
                df, length_cut_min, length_cut_max, plot, area_hist_x, min_area_x, A, mu, sigma)

        return (A, mu, sigma), cov

    @classmethod
    def get_1pe_rough(cls, df: pd.DataFrame,
                      length_cut_min: int,
                      length_cut_max: int,
                      bins: int or list = 200,
                      use_scipy_find_peaks: bool = False) -> tuple:
        """From an event df (1 channel), find the rough position of the
        SPE from the DCR vs threshold curve and its derivative

        Args:
            df (pd.DataFrame): dataframe with the processed data.
            length_cut_min (int): minimum value of lenght to consider
            length_cut_max (int): maximum value of lenght to consider
            bins (intorlist, optional): number of bins to make the are
                histogram or list of bin edges to consider on the histogram.
                Defaults to 200.

        Returns:
            tuple: computed arrays: area_hist_x, DCR_values, DCR_der_x_points,
                DCR_der_y_points, min_area_x
        """

        (area_hist_x, DCR_values) = cls.get_DCR_above_threshold_values(
            df, length_cut_min, length_cut_max, bins, output='values')  # type: ignore
        grad = np.gradient(DCR_values)
        grad_spline = itp.UnivariateSpline(area_hist_x, grad)
        # , s = len(area_hist_x)*3)
        DCR_der_x_points = np.linspace(area_hist_x[0], area_hist_x[-1], bins)
        DCR_der_y_points = grad_spline(DCR_der_x_points)

        if use_scipy_find_peaks:
            pks, props = find_peaks(-1 * DCR_der_y_points,
                                    prominence=20)
            peaks_area_values = DCR_der_x_points[pks]
            if len(peaks_area_values) == 0:
                print('Could not find any peaks')
                min_area_x = np.nan
            elif len(peaks_area_values) > 0:
                if (peaks_area_values[0] > 2000 *
                        200) or (len(peaks_area_values) == 1):
                    min_area_x = peaks_area_values[0]
                else:
                    min_area_x = peaks_area_values[1]
        else:
            min_idx = np.where(DCR_der_y_points == min(DCR_der_y_points))
            min_area_x = DCR_der_x_points[min_idx][0]

        return (area_hist_x, DCR_values, DCR_der_x_points,
                DCR_der_y_points, min_area_x)  # type:ignore

    @classmethod
    def get_DCR_above_threshold_values(cls, df: pd.DataFrame,
                                       length_cut_min: int = 5,
                                       length_cut_max: int = 80,
                                       bins: int or list = 200,
                                       output: str = 'values',
                                       **kwargs) -> Union[
            tuple,
            itp.UnivariateSpline,
            itp.interp1d,
            None]:
        """Computes the event rate in a sweep of area thresholds and
        returns the pair [area thersholds, DCR values]

        Args:
            df (pd.DataFrame): a pd.DataFrame with the series "area" and
                "length".
            length_cut_min (int, optional): cut to impose on the minimum
                length of the peaks for noise suppression. Defaults to 5.
            length_cut_max (int, optional): cut to impose on the maximum
                length of the peaks for noise suppression. Defaults to 5.
            bins (intorlist, optional): number of bins to make the are
                histogram or list of bin edges to consider on the histogram.
                Defaults to 200.
            output (str): type of output. Can be 'values', 'spline' or
                'interp1d'

        Returns:
            tuple: pair of np.ndarrays (area thersholds, DCR values)
        """

        if output not in ['values', 'spline', 'interp1d']:
            raise NotImplementedError("Specifiy a valid output. Options are "
                                      "'values', 'spline' or 'interp1d'.")

        area_hist = np.histogram(df[(df['length'] > length_cut_min) &
                                    (df['length'] < length_cut_max)]['area'],
                                 bins=bins)
        area_hist_x = area_hist[1]
        area_hist_x = (area_hist_x +
                       (area_hist_x[1] - area_hist_x[0]) / 2)[:-1]
        area_hist_y = area_hist[0]

        DCR_values = np.flip(np.cumsum(np.flip(area_hist_y)))

        if output == 'values':
            return (area_hist_x, DCR_values)
        elif output == 'spline':
            DCR_func = itp.UnivariateSpline(area_hist_x, DCR_values, **kwargs)
            return DCR_func
        elif output == 'interp1d':
            DCR_func = itp.interp1d(area_hist_x, DCR_values)
            return DCR_func

    @staticmethod
    def get_DCR(df: pd.DataFrame,
                length_cut_min: int,
                length_cut_max: int,
                spe_area: float,
                sensor_area: float,
                t: float) -> tuple:
        """Compute the dark count rate (DCR) and crosstalk probability
        (CTP) of a dataset.

        Args:
            df (pd.DataFrame): DataFrame of the dataset (single ch)
            length_cut_min (int): low length cut to apply
            length_cut_max (int): high length cut to apply
            spe_area (float): SPE area
            sensor_area (float): effective are of the sensor
            t (float): livetime of the dataset

        Returns:
            tuple: DCR value, error of DCR value, CTP value, error of
                CTP value
        """
        pe_0_5 = spe_area * 0.5
        pe_1_5 = spe_area * 1.5

        DC_0_5 = len(df[(df['length'] > length_cut_min) &
                        (df['length'] < length_cut_max) &
                        (df['area'] > pe_0_5)])
        DC_1_5 = len(df[(df['length'] > length_cut_min) &
                        (df['length'] < length_cut_max) &
                        (df['area'] > pe_1_5)])

        DC_0_5_error = np.sqrt(DC_0_5)
        DC_1_5_error = np.sqrt(DC_1_5)

        DCR = DC_0_5 / sensor_area / t
        DCR_error = DC_0_5_error / sensor_area / t

        CTP = DC_1_5 / DC_0_5
        CTP_error = np.sqrt((DC_1_5_error / DC_0_5)**2 +
                            (DC_1_5 * DC_0_5_error / (DC_0_5)**2)**2)

        return DCR, DCR_error, CTP, CTP_error

    @staticmethod
    def get_DCR_amplitude(df: pd.DataFrame,
                          length_cut_min: int,
                          length_cut_max: int,
                          pe_amplitude: float,
                          pe_amplitude_std: float,
                          sensor_area: float,
                          t: float) -> tuple:
        """Compute the dark count rate (DCR) and crosstalk probability
        (CTP) of a dataset, amplitude based.

        Args:
            df (pd.DataFrame): DataFrame of the dataset (single ch)
            length_cut_min (int): low length cut to apply
            length_cut_max (int): high length cut to apply
            pe_amplitude (float): SPE amplitude
            sensor_area (float): effective are of the sensor
            t (float): livetime of the dataset

        Returns:
            tuple: DCR value, error of DCR value, CTP value, error of
                CTP value
        """
        pe_0_5 = pe_amplitude * 0.5
        pe_1_5 = pe_amplitude * 1.5
        pe_m5sigma = pe_amplitude - 5 * pe_amplitude_std
        pe_p5sigma = pe_amplitude + 5 * pe_amplitude_std

        DC_0_5 = len(df[(df['length'] > length_cut_min) &
                        (df['length'] < length_cut_max) &
                        (df['amplitude'] < pe_p5sigma)])
        DC_1_5 = len(df[(df['length'] > length_cut_min) &
                        (df['length'] < length_cut_max) &
                        (df['area'] < pe_m5sigma)])

        DC_0_5_error = np.sqrt(DC_0_5)
        DC_1_5_error = np.sqrt(DC_1_5)

        DCR = DC_0_5 / sensor_area / t
        DCR_error = DC_0_5_error / sensor_area / t

        CTP = DC_1_5 / DC_0_5
        CTP_error = np.sqrt((DC_0_5_error / DC_1_5)**2 +
                            (DC_0_5 * DC_1_5_error / (DC_1_5)**2)**2)

        return DCR, DCR_error, CTP, CTP_error

    @staticmethod
    def print_DCR_CTP(DCR: float, DCR_error: float,
                      CTP: float, CTP_error: float) -> None:
        """Print the dark count rate (DCR) and crosstalk probability
        (CTP) values in a nice formatted and welcoming message.

        Args:
            DCR (float): DCR value
            DCR_error (float): Error on the DCR value
            CTP (float): CTP value
            CTP_error (float): Error on the CTP value
        """

        print(f'Your lovely DCR is: ({DCR:.2f} +- {DCR_error:.2f}) Hz/mm^2')
        print(f'Your lovely CTP is: ({CTP*100:.2f} +- {CTP_error*100:.2f})%')


class DCR_dataset(DCR_analysis):
    """Object class to hold dark count related instances and
    methods. Collects all the data and properties of a single MMPC,
    meaning the pair (module, channel) for all the available voltages at
    a certain temperature.
    """

    __version__ = '0.0.2'

    def __init__(self, run: pylars.utils.input.run, temperature: float,
                 module: int, channel: str,
                 processor: pylars.processing.rawprocessor.run_processor,
                 force_processing = False):

        self.run = run
        self.temp = temperature
        self.module = module
        self.channel = channel
        self.process = processor
        self.voltages = self.get_voltages_available()
        self.plots_flag = False
        self.livetimes = self.get_livetimes()
        self.force_processing = force_processing

        self.set_standard_cuts()

        # There are better ways to set these options but this is stil
        # better then a lot of flags
        self.use_scipy_find_peaks = False
        self.amplitude_based = False

    def set_plots_flag(self, flag: bool) -> None:
        """Set if computing properties makes plots (True) or not (False).
        Assumes a ./figures/ directory exists.

        Args:
            flag (bool): True for plotting stuff, False for not making nice
                pictures (plots) to hang on the wall. Yes, I hang plots on
                my bedroom wall and it looks nice.
        """
        self.plots_flag = flag

    def define_SiPM_config(self,
                           sensor_area: float = 12 * 12,
                           ) -> None:
        r"""Define the SiPM data related quantities for the dataset.

        Args:
            sensor_area (float, optional): Area of the photosensor (mm\*\*2).
                Defaults to 12\*12.
        """
        SiPM_config = {'sensor_area': sensor_area,
                       }

        self.SiPM_config = SiPM_config

    def get_voltages_available(self) -> np.ndarray:
        """Checks the loaded run for which voltages are available for the
        defined temperature.

        Returns:
            np.ndarray: array of the available voltages
        """

        voltages = []
        for _dataset in self.run.datasets:
            if (_dataset.temp == self.temp) and (_dataset.kind == 'DCR'):
                voltages.append(_dataset.vbias)
        voltages = np.unique(voltages)

        return voltages

    def get_livetimes(self) -> dict:
        """Fetch the livetimes of a dataset.

        Based on the propertied of the DCR_dataset, creates its dataset object
        and computes the livetime for each vbias by multiplying the number
        of samples per waveform with the number of waveforms in the dataset
        and the duration of a sample (given by `self.run.ADC_config['dt']`).
        Provides the result in dict format where the keys are the vbias of
        the dataset.

        Returns:
            dict: _description_
        """
        livetimes = {}
        for v in self.voltages:

            # Need all this to find the correct path...
            self.datasets_df = self.run.get_run_df()
            selection = ((self.datasets_df['kind'] == 'DCR') &
                         (self.datasets_df['vbias'] == v) &
                         (self.datasets_df['temp'] == self.temp) &
                         # to end with just one of the modules.
                         # It's assumed both have the same number of
                         # entries and size of entries
                         (self.datasets_df['module'] == 0))

            ds_selected = self.datasets_df[selection]

            assert len(
                self.datasets_df[selection]) == 1, "Found more than 1 ds with the same config. Help."

            ds_temp = pylars.utils.input.dataset(path=str(ds_selected['path'].values[0]),
                                                 kind='DCR',
                                                 module=0,
                                                 temp=self.temp,
                                                 vbias=v)
            try:
                n_waveforms, n_samples = ds_temp.read_sizes()
                livetimes[v] = n_waveforms * \
                    n_samples * self.run.ADC_config['dt']
            except BaseException:
                livetimes[v] = np.nan

        return livetimes

    def set_standard_cuts(self,
                          cut_area_min: float = 5,
                          cut_area_max: float = 1e6,
                          cut_length_min: int = 4,
                          cut_length_max: int = 70,
                          cut_n_pulses_min: float = 0,
                          cut_n_pulses_max: float = 2) -> None:
        """Sets the cuts to use in analysis as object variables.

        Args:
            cut_area_min (float, optional): Area minimum value. Defaults to 5.
            cut_area_max (float, optional): Area maximum value. Defaults
                to 1e6.
            cut_length_min (float, optional): Lenght minimum value.
                Defaults to 4.
            cut_length_max (float, optional): Lenght minimum value.
                 Defaults to 70.
            cut_n_pulses_min (float, optional): Minimum number of pulses in
                the waveform. Defaults to 0.
            cut_n_pulses_max (float, optional): Maximum number of pulses in
                the waveform. Defaults to 2.
        """

        self.cut_area_min = cut_area_min
        self.cut_area_max = cut_area_max
        self.cut_length_min = cut_length_min
        self.cut_length_max = cut_length_max
        self.cut_n_pulses_min = cut_n_pulses_min
        self.cut_n_pulses_max = cut_n_pulses_max

    def load_processed_data(self, force_processing: bool = False) -> None:
        """For all the voltages of a DCR_dataset (smae temperature) looks
        for already processed files to load. If force_processing=True and
        no saved file is found, processes the dataset with standard
        options (sigma = 5, N_baseline = 50).

        Args:
            force_processing (bool, optional): Flag to force processing
                of raw data in case the processed dataset is not found.
                Defaults to False.

        Returns:
            dict: dictionary with the data df for each voltage as a key.
        """

        if force_processing != self.force_processing:
            self.force_processing = force_processing

        self.data = {}

        for _voltage in tqdm(self.voltages,
                             desc=f'Loading processed data for DCR ' +
                             f'data at {self.temp}K: ',
                             total=len(self.voltages),
                             leave=False):
            processed_data = pylars.utils.output.processed_dataset(
                run=self.run,
                kind='DCR',
                vbias=_voltage,
                temp=self.temp,
                path_processed=('/disk/gfs_atp/xenoscope/SiPMs/char_campaign/'
                                'processed_data/'),
                process_hash=self.process.hash)
            processed_data.load_data(force=self.force_processing)

            _df = processed_data.data
            mask = ((_df['module'] == self.module) &
                    (_df['channel'] == self.channel))

            self.data[_voltage] = copy.deepcopy(_df[mask])

    def compute_properties_of_dataset(
            self,
            use_n_pulse_wf: bool = False,
            compute_BV_flag: bool = True,
            amplitude_based: bool = False) -> pd.DataFrame:
        """Calculate the gain, DCR, CTP and BV for the dataset in a single
        line!

        Args:
            use_1_pulse_wf (bool):
            compute_BV_flag (bool):

        Returns:
            pd.DataFrame: dataframe with all the computed properties with
                the columns ['T', 'V', 'SPE_area','SPE_area_error', 'Gain',
                'Gain_error', 'SPE_res', 'SPE_res_error', 'DCR',
                'DCR_error', 'CTP', 'CTP_error']
        """

        assert isinstance(self.data, dict), ('Woops, no data found! Load'
                                             ' data into the dataset first')

        if amplitude_based:
            raise NotImplementedError

        voltage_list = self.voltages

        _results_dataset = pd.DataFrame(
            columns=['T', 'V', 'SPE_area', 'SPE_area_error', 'Gain',
                     'Gain_error', 'SPE_res', 'SPE_res_error', 'DCR',
                     'DCR_error', 'CTP', 'CTP_error'])

        for _volt in voltage_list:
            # select voltage
            df = self.data[_volt]
            if use_n_pulse_wf == True:
                df = df[(df['n_pulses'] > self.cut_n_pulses_min) &
                        (df['n_pulses'] <= self.cut_n_pulses_max)]
            if self.plots_flag == True:
                plot_name_1pe_fit = (f'{self.temp}K_{_volt}V_mod{self.module}_'
                                     f'ch{self.channel}')
            else:
                plot_name_1pe_fit = False

            try:
                # Get SPE value from Gaussian fit
                (A, mu, sigma), cov = self.get_1pe_value_fit(
                    df, plot=plot_name_1pe_fit,
                    length_cut_min=self.cut_length_min,
                    length_cut_max=self.cut_length_max,
                    use_scipy_find_peaks=self.use_scipy_find_peaks)

                A_err, mu_err, sigma_err = np.sqrt(np.diag(cov))

                # Calculate DCR and CTP
                DCR, DCR_error, CTP, CTP_error = self.get_DCR(
                    df=df,
                    length_cut_min=self.cut_length_min,
                    length_cut_max=self.cut_length_max,
                    spe_area=mu,
                    sensor_area=self.SiPM_config['sensor_area'],
                    t=self.livetimes[_volt])

                # Calculate gain
                gain = get_gain(F_amp=self.run.ADC_config['F_amp'],
                                spe_area=mu,
                                ADC_range=self.run.ADC_config['ADC_range'],
                                ADC_impedance=self.run.ADC_config['ADC_impedance'],
                                ADC_res=self.run.ADC_config['ADC_res'],
                                q_e=self.run.ADC_config['q_e'])
                gain_error = get_gain(F_amp=self.run.ADC_config['F_amp'],
                                      spe_area=mu_err,
                                      ADC_range=self.run.ADC_config['ADC_range'],
                                      ADC_impedance=self.run.ADC_config['ADC_impedance'],
                                      ADC_res=self.run.ADC_config['ADC_res'],
                                      q_e=self.run.ADC_config['q_e'])

                SPE_res = np.abs(sigma / mu) * 100  # in %
                SPE_res_err = np.sqrt(
                    (sigma_err / mu)**2 + (sigma * mu_err / mu**2)**2) * 100  # in %

                # Merge into rolling dataframe (I know it's slow... make a PR,
                # pls)
                _results_dataset = pd.concat(
                    (_results_dataset,
                     pd.DataFrame({'T': [self.temp],
                                   'V': [_volt],
                                   'SPE_area': [mu],
                                   'SPE_area_error': [mu_err],
                                   'Gain': [gain],
                                   'Gain_error': [gain_error],
                                   'SPE_res': [SPE_res],
                                   'SPE_res_error': [SPE_res_err],
                                   'DCR': [DCR],
                                   'DCR_error': [DCR_error],
                                   'CTP': [CTP * 100],
                                   'CTP_error': [CTP_error * 100]})
                     ), ignore_index=True
                )
            except:
                print(f'Could not compute properties of module {self.module}, '
                      f'channel {self.channel}, {self.temp} K, {_volt} V. '
                      f'Skipping dataset.')
                _results_dataset = pd.concat(
                    (_results_dataset,
                     pd.DataFrame({'T': [self.temp],
                                   'V': [_volt],
                                   'SPE_area': [np.nan],
                                   'SPE_area_error': [np.nan],
                                   'Gain': [np.nan],
                                   'Gain_error': [np.nan],
                                   'SPE_res': [np.nan],
                                   'SPE_res_error': [np.nan],
                                   'DCR': [np.nan],
                                   'DCR_error': [np.nan],
                                   'CTP': [np.nan],
                                   'CTP_error': [np.nan]})
                     ), ignore_index=True
                )

        # Compute BV from gain(V) curve and update df
        if compute_BV_flag == True:
            if self.plots_flag == True:
                plot_BV = f'BV_mod{self.module}_ch{self.channel}'
            else:
                plot_BV = False
            _results_dataset, _a, _b, _b_err = compute_BV_df(
                _results_dataset, plot_BV)

        return _results_dataset

    @staticmethod
    def get_how_many_peaks_per_waveform(df: pd.DataFrame,
                                        verbose: bool = False) -> pd.DataFrame:
        """Finds how many peaks each waveform has.
        Only looks in wavforms with at least 1 peak

        Since commit 6f8e0c72b8e3e0bf7c5307941b49bdf57879d554 this method
        is not needed as n_pulses is stored during processing.

        Args:
            df (pd.DataFrame): dataframe with the processed pulse data.
            verbose (bool): print info during processing.

        Returns:
            pd.DataFrame: dataframe with the pairs: wf_number - pulse_count.
        """
        t0 = time.time()
        if verbose:
            print('Starting counting peaks at:', t0)
        wf_number_arr = np.array(df['wf_number'].values)
        waveforms_w_pulses, N_pulses = _get_how_many_peaks_per_waveform(
            wf_number_arr)

        pulse_count_df = pd.concat(
            [pd.Series(waveforms_w_pulses, name='wf_number'),
             pd.Series(N_pulses, name='pulse_count')], axis=1
        )
        t1 = time.time()
        if verbose:
            print('Finished conting peaks at: ', t1)
            print('Took time in calc: ', t1 - t0)
        return pulse_count_df

    def get_med_amplitude(self, df: pd.DataFrame,
                          cut_mask: np.ndarray) -> tuple:
        """Calculate the median and standard deviation of the distribution
        of amplitudes, applying a given mask.

        Args:
            df (pd.DataFrame): dataframe with the processed pulse data
            extra_cut_mask (np.ndarray): cut mask

        Returns:
            tuple: _description_
        """
        med_amplitude = np.median(df[cut_mask]['amplitude'])
        std_amplitude = np.std(df[cut_mask]['amplitude'])
        return med_amplitude, std_amplitude

    def compute_properties_of_dataset_amplitude_based(self,
                                                      use_n_pulse_wf: bool = True) -> pd.DataFrame:
        """NOT IMPLEMENTED - Needs review and merge with other compute method

        Calculate the gain, DCR, CTP and BV for the dataset in a single
        line! This is an alternative method to the main one, using amplitude
        cuts instead of area cuts.

        Returns:
            pd.DataFrame: dataframe with all the computed properties with
                the columns ['V','T','path','module','channel','Gain','DCR',
                'CTP','DCR_error','CTP_error','BV','OV']
        """
        raise NotImplementedError  # needs review and merging with normal


class DCR_run():
    """Collection of all the DCR_datasets results for a run, ie, for all
    the channels and modules, for every available temperatures and voltages.

    The results of every dataset (channel, V, T) is saved on the instance
    DCR_run.results_df .
    """

    __version__ = '0.0.1'

    def __init__(self, run: pylars.utils.input.run,
                 processor: pylars.processing.rawprocessor.run_processor,
                 use_n_pulse: bool = True,
                 force_processing: bool = False,
                 analysis_path: Optional[str] = None):

        self.run = run
        self.process = processor
        self.use_n_pulse = use_n_pulse
        self.datasets = self.process.datasets_df
        self.temperatures = self.get_run_temperatures()
        self.plots_flag = False
        if analysis_path is None:
            self.analysis_path = (f'{self.run.main_data_path[:-9]}analysis_data'
                              f'/run{self.run.run_number}/')
        else: self.analysis_path = analysis_path
        self.force_processing = force_processing

    def set_plots_flag(self, flag: bool) -> None:
        """Set if computing properties makes plots (True) or not (False).
        Assumes a ./figures/ directory exists.

        Args:
            flag (bool): True for plotting stuff, False for not making nice
                pictures (plots) to hang on the wall. Yes, I hang plots on my
                bedroom wall and it looks nice.
        """
        self.plots_flag = flag

    def get_run_temperatures(self) -> np.ndarray:
        """Get all the temperatures available in the DCR run.

        Returns:
            np.ndarray: array with all the available temperatures.
        """
        temp_list = np.unique(self.datasets['temp'])

        return temp_list

    def initialize_results_df(self) -> None:
        """Initialize a clean results_df instance in the object.
        """

        results_df = pd.DataFrame(
            columns=['T', 'V', 'SPE_area', 'SPE_area_error', 'Gain',
                     'Gain_error', 'SPE_res', 'SPE_res_error', 'DCR',
                     'DCR_error', 'CTP', 'CTP_error'])

        self.results_df = results_df

    def define_run_SiPM_config(self, sensor_area: float = 12 * 12,
                               ) -> None:
        r"""Define the SiPM data related quantities for the dataset.

        Args:
            livetime (float): livetime of the measurement for DCR porpuses.
            sensor_area (float, optional): Area of the photosensor (mm\*\*2).
                Defaults to 12\*12.
        """
        SiPM_config = {'sensor_area': sensor_area,
                       }
        self.SiPM_config = SiPM_config
    
    def set_standard_cuts_run(self,
                          cut_area_min: float = 5,
                          cut_area_max: float = 1e6,
                          cut_length_min: int = 4,
                          cut_length_max: int = 70,
                          cut_n_pulses_min: float = 0,
                          cut_n_pulses_max: float = 2) -> None:
        """Sets the cuts to use in analysis as (run) object variables.

        Args:
            cut_area_min (float, optional): Area minimum value. Defaults to 5.
            cut_area_max (float, optional): Area maximum value. Defaults
                to 1e6.
            cut_length_min (float, optional): Lenght minimum value.
                Defaults to 4.
            cut_length_max (float, optional): Lenght minimum value.
                 Defaults to 70.
            cut_n_pulses_min (float, optional): Minimum number of pulses in
                the waveform. Defaults to 0.
            cut_n_pulses_max (float, optional): Maximum number of pulses in
                the waveform. Defaults to 2.
        """

        self.cut_area_min = cut_area_min
        self.cut_area_max = cut_area_max
        self.cut_length_min = cut_length_min
        self.cut_length_max = cut_length_max
        self.cut_n_pulses_min = cut_n_pulses_min
        self.cut_n_pulses_max = cut_n_pulses_max

    def load_dataset(self, temp: float,
                     module: int,
                     channel: str) -> DCR_dataset:
        """Create a DCR_dataset object for a (T, mod, ch) configuration and
        load the corresponding data into it.

        ! This assumes processed data is available for all the raw files of
        the DCR run datasets !

        Args:
            temp (float): temperature to consider
            module (int): module to load
            channel (str): channel in the module to select

        Returns:
            DCR_dataset: dataset obeject
        """
        particular_DCR_dataset = DCR_dataset(
            run=self.run,
            temperature=temp,
            module=module,
            channel=channel,
            processor=self.process,
            force_processing=self.force_processing)

        particular_DCR_dataset.load_processed_data(
            force_processing=self.force_processing)

        return particular_DCR_dataset

    def compute_properties_of_ds(self, temp: float,
                                 module: int,
                                 channel: str,
                                 amplitude_based: bool = False
                                 ) -> pd.DataFrame:
        """Loads and computes the properties of a single dataset (temp,
        module, channel) by creating a DCR_dataset object and calling its
        methods.

        Args:
            temp (float): temperature
            module (int): module
            channel (str): channel
            amplitude_based (bool): if the computation method is based on
                amplitude instead of area

        Returns:
            pd.DataFrame: dataframe
        """
        assert isinstance(self.run.ADC_config, dict), 'No ADC_config found!'

        ds = self.load_dataset(temp, module, channel)
        ds.set_plots_flag(self.plots_flag)
        ds.SiPM_config = self.SiPM_config
        ds.cut_area_min = self.cut_area_min
        ds.cut_area_max = self.cut_area_max
        ds.cut_length_min = self.cut_length_min
        ds.cut_length_max = self.cut_length_max
        ds.cut_n_pulses_min = self.cut_n_pulses_min
        ds.cut_n_pulses_max = self.cut_n_pulses_max

        if amplitude_based:
            ds_results = ds.compute_properties_of_dataset_amplitude_based(
                use_n_pulse_wf=self.use_n_pulse,
            )
        else:
            ds_results = ds.compute_properties_of_dataset(
                compute_BV_flag=False,
                use_n_pulse_wf=self.use_n_pulse)

        # Add module and channel columns
        module_Series = pd.Series([module] * len(ds_results), name='module')
        channel_Series = pd.Series([channel] * len(ds_results), name='channel')
        ds_results = pd.concat([ds_results, module_Series, channel_Series],
                               axis=1)
        return ds_results

    def compute_properties_of_run(self, amplitude_based: bool = False) -> None:
        """Loads and computes the properties of ALL the datasets.

        Args:
            amplitude_based (bool): Turn True to compute SPE based on
                amplitude.

        Returns:
            pd.DataFrame: The results.
        """
        self.initialize_results_df()

        all_channels = pylars.utils.common.get_channel_list(self.process)
        for temperature in self.temperatures:
            for (module, channel) in tqdm(all_channels,
                                          (f'Computing properties for '
                                           f'T={temperature}: ')):
                _ds_results = self.compute_properties_of_ds(
                    temp=temperature,
                    module=module,
                    channel=channel,
                    amplitude_based=amplitude_based)

                self.results_df = pd.concat(
                    [self.results_df, _ds_results],  # type: ignore
                    ignore_index=True)

    def save_results(self, custom_name: str) -> None:
        """Save dataframe of results to a hdf5 file. Saved files go to
        self.analysis_path.

        Args:
            name (str): name to give the file (without extension).
        """
        assert isinstance(
            self.results_df, pd.DataFrame), ("Trying to save results that do "
                                             "not exist in the object, c'mon"
                                             ", you know better.")
        assert len(self.results_df) > 0, ("Results df is empty, please compute"
                                          "something before trying to save, "
                                          "otherwire it's just a waste of "
                                          "disk space")

        name = f'DCR_results_{custom_name}'
        self.results_df.to_hdf(self.analysis_path + name + '.h5', 'df')
        print('Saved results to ')

    def load_results(self, name: str) -> None:
        """Load dataframe of results from a hdf5 file. Looks for files in
        the standard analysis cache directory.

        Args:
            name (str): name of the file to load (without extension)
        """

        _df = pd.read_hdf(self.analysis_path + name + '.h5')
        self.results_df = _df


@numba.njit
def _get_how_many_peaks_per_waveform(wf_number_list: np.ndarray) -> Tuple:
    """Numbafied process of counting how peaks there are in each waveform.

    Not used any longer since "n_pulses" is stored.

    Args:
        wf_number_list (np.ndarray): the wf number of each identified peak.

    Returns:
        tuple: (wf number with pulses, the number of pulses in the wf)
    """
    waveforms_w_pulses = np.unique(wf_number_list)
    n_waveforms = len(waveforms_w_pulses)

    N_pulses = np.zeros(n_waveforms)

    for i, _wf in enumerate(waveforms_w_pulses):
        _n_pulses = np.count_nonzero(wf_number_list == _wf)
        N_pulses[i] = _n_pulses
    return (waveforms_w_pulses, N_pulses)

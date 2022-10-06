import copy
from datetime import datetime

import numpy as np
import pandas as pd
import pylars
import pylars.plotting.plotanalysis
import pylars.utils.input
import pylars.utils.output
import scipy.interpolate as itp
from pylars.analysis.breakdown import compute_BV
from scipy.optimize import curve_fit
from tqdm import tqdm

from .common import Gaussean, func_linear


class DCR_dataset():
    """Object class to hold dark count related instances and
    methods. Collects all the data and properties of a single MMPC,
    meaning the pair (module, channel) for all the available voltages at
    a certain temperature.
    """

    __version__ = '0.0.1'

    def __init__(self, run: pylars.utils.input.run, temperature: float,
                 module: int, channel: str,
                 processor: pylars.processing.rawprocessor.run_processor):

        self.run = run
        self.temp = temperature
        self.module = module
        self.channel = channel
        self.process = processor
        self.voltages = self.get_voltages_available()
        self.plots_flag = False

    def set_plots_flag(self, flag: bool) -> None:
        """Set if computing properties makes plots (True) or not (False).
        Assumes a ./figures/ directory exists.

        Args:
            flag (bool): True for plotting stuff, False for not making nice
        pictures (plots) to hang on the wall. Yes, I hang plots on my bedroom
        wall and it looks nice.
        """
        self.plots_flag = flag

    def define_ADC_config(self, F_amp: float, ADC_range: int = 2.25,
                          ADC_impedance: int = 50, ADC_res: float = 2**14,
                          q_e: float = 1.602176634e-19) -> None:
        """Define the ADC related quantities for the dataset.

        Args:
            F_amp (float): signal amplification from the sensor (pre-amp *
        external amplification on rack)
            ADC_range (int, optional): Range of the ADC [V]. Defaults to 2.25.
            ADC_impedance (int, optional): Impedance of the ADC and cables
        [ohm]. Defaults to 50.
            ADC_res (float, optional): Resolution/bits of the ADC (2**N_bits).
        Defaults to 2**14.
            q_e (float, optional): Element charge of the electron [C].
        Defaults to 1.602176634e-19.
        """
        ADC_config = {'ADC_range': ADC_range,
                      'ADC_impedance': ADC_impedance,
                      'F_amp': F_amp,
                      'ADC_res': ADC_res,
                      'q_e': q_e}

        self.ADC_config = ADC_config

    def define_SiPM_config(self, livetime: float,
                           sensor_area: float = 12 * 12,
                           ) -> None:
        """Define the SiPM data related quantities for the dataset.

        Args:
            livetime (float): livetime of the measurement for DCR porpuses.
            sensor_area (float, optional): Area of the photosensor (mm**2).
        Defaults to 12*12.
        """
        SiPM_config = {'livetime': livetime,
                       'sensor_area': sensor_area,
                       }

        self.SiPM_config = SiPM_config

    def get_voltages_available(self) -> np.array:
        """Checks the loaded run for which voltages are available for the
        defined temperature.

        Returns:
            np.array: array of the available voltages
        """

        voltages = []
        for _dataset in self.run.datasets:
            if (_dataset.temp == self.temp) and (_dataset.kind == 'DCR'):
                voltages.append(_dataset.vbias)
        voltages = np.unique(voltages)

        return voltages

    def load_processed_data(self, force_processing: bool = False) -> dict:
        """For all the voltages of a DCR_dataset (smae temperature) looks
        for already processed files to load. If force_processing=True and
        no saved file is found, processes the dataset with standard
        options (sigma = 5, N_baseline = 50).

        Args:
            force_processing (bool, optional): Flag to force processing
        of raw data in case the processed dataset is not found. Defaults
        to False.

        Returns:
            dict: _description_
        """
        self.data = {}
        for _voltage in self.voltages:
            processed_data = pylars.utils.output.processed_dataset(
                run=self.run,
                kind='DCR',
                vbias=_voltage,
                temp=self.temp,
                path_processed=('/disk/gfs_atp/xenoscope/SiPMs/char_campaign/'
                                'processed_data/'),
                process_hash=self.process.hash)
            processed_data.load_data(force=force_processing)

            _df = processed_data.data
            mask = ((_df['module'] == self.module) &
                    (_df['channel'] == self.channel))

            self.data[_voltage] = copy.deepcopy(_df[mask])

    @classmethod
    def get_1pe_value_fit(cls,
                          df: pd.DataFrame,
                          length_cut: int = 5,
                          plot: bool or str = False) -> tuple:
        """Try to fit the SPE peak in the area histogram and return the
        Gaussian paramenters.

        Args:
            df (pd.DataFrame): _description_
            length_cut (int, optional): cut to impose on the length of
        the peaks for noise suppression. Defaults to 5.
            plot (boolorstr, optional): _description_. Defaults to False.

        Returns:
            tuple: _description_
        """

        (area_hist_x, DCR_values, DCR_der_x_points,
         DCR_der_y_points, min_area_x) = cls.get_1pe_rough(df, length_cut)

        if plot != False:
            pylars.plotting.plotanalysis.plot_DCR_curve(
                plot, area_hist_x, DCR_values, DCR_der_x_points,
                DCR_der_y_points, min_area_x)

        area_hist = np.histogram(df[df['length'] > length_cut]['area'],
                                 bins=np.linspace(0.5 * min_area_x, 1.5 * min_area_x, 300))
        area_hist_x = area_hist[1]
        area_hist_x = (
            area_hist_x + (area_hist_x[1] - area_hist_x[0]) / 2)[:-1]
        area_hist_y = area_hist[0]

        (A, mu, sigma), cov = curve_fit(Gaussean, area_hist_x, area_hist_y,
                                        p0=(2000, min_area_x, 0.05 * min_area_x))

        if plot != False:
            pylars.plotting.plotanalysis.plot_SPE_fit(
                df, length_cut, plot, area_hist_x, min_area_x, A, mu, sigma)

        return (A, mu, sigma), cov

    @classmethod
    def get_1pe_rough(cls, df: pd.DataFrame,
                      length_cut: int,
                      bins: int or list = 200) -> tuple:
        """From an event df (1 channel), find the rough position of the
        SPE from the DCR vs threshold curve and its derivative

        Args:
            df (_type_): _description_
            length_cut (_type_): _description_
            bins (intorlist, optional): number of bins to make the are
        histogram or list of bin edges to consider on the histogram.
        Defaults to 200.

        Returns:
            tuple: _description_
        """

        (area_hist_x, DCR_values) = cls.get_DCR_above_threshold_values(
            df, length_cut, bins)
        grad = np.gradient(DCR_values)
        grad_spline = itp.UnivariateSpline(area_hist_x, grad)
        # , s = len(area_hist_x)*3)
        _x = np.linspace(area_hist_x[0], area_hist_x[-1], 500)
        _y = grad_spline(_x)
        min_idx = np.where(_y == min(_y))
        min_area_x = _x[min_idx][0]
        return area_hist_x, DCR_values, _x, _y, min_area_x

    @classmethod
    def get_DCR_above_threshold_values(cls, df: pd.DataFrame,
                                       length_cut: int = 5,
                                       bins: int or list = 200) -> tuple:
        """Computes the event rate in a sweep of area thresholds and
        returns the pair [area thersholds, DCR values]

        Args:
            df (pd.DataFrame): a pd.DataFrame with the series "area" and
        "length".
            length_cut (int, optional): cut to impose on the length of
        the peaks for noise suppression. Defaults to 5.
            bins (intorlist, optional): number of bins to make the are
        histogram or list of bin edges to consider on the histogram.
        Defaults to 200.

        Returns:
            tuple: pair of np.arrays (area thersholds, DCR values)
        """
        area_hist = np.histogram(df[df['length'] > length_cut]['area'],
                                 bins=bins)
        area_hist_x = area_hist[1]
        area_hist_x = (area_hist_x +
                       (area_hist_x[1] - area_hist_x[0]) / 2)[:-1]
        area_hist_y = area_hist[0]

        DCR_values = np.flip(np.cumsum(np.flip(area_hist_y)))

        return (area_hist_x, DCR_values)

    @classmethod
    def get_DCR_above_threshold_spline(cls, df: pd.DataFrame,
                                       length_cut: int = 5,
                                       bins: int or list = 200,
                                       **kwargs) -> itp.UnivariateSpline:
        """Computes the event rate in a sweep of area thresholds and
        returns a spline object of the curve.

        Args:
            df (pd.DataFrame): a pd.DataFrame with the series "area" and
        "length".
            length_cut (int, optional): cut to impose on the length of
        the peaks for noise suppression. Defaults to 5.
            bins (intorlist, optional): number of bins to make the are
        histogram or list of bin edges to consider on the histogram.
        Defaults to 200.

        Returns:
            itp.UnivariateSpline: spline object of the DCR vs area
        threshold curve.
        """
        area_hist = np.histogram(df[df['length'] > length_cut]['area'],
                                 bins=bins)
        area_hist_x = area_hist[1]
        area_hist_x = (area_hist_x +
                       (area_hist_x[1] - area_hist_x[0]) / 2)[:-1]
        area_hist_y = area_hist[0]
        DCR_values = np.flip(np.cumsum(np.flip(area_hist_y)))
        DCR_func = itp.UnivariateSpline(area_hist_x, DCR_values, **kwargs)

        return DCR_func

    @classmethod
    def get_DCR_above_threshold_interp1d(cls, df: pd.DataFrame,
                                         length_cut: int = 5,
                                         bins: int or list = 200
                                         ) -> itp.interp1d:
        """Computes the event rate in a sweep of area thresholds and
        returns a 1D interpolation object of the curve.

        Args:
            df (pd.DataFrame): a pd.DataFrame with the series "area" and
        "length".
            length_cut (int, optional): cut to impose on the length of
        the peaks for noise suppression. Defaults to 5.
            bins (intorlist, optional): number of bins to make the are
        histogram or list of bin edges to consider on the histogram.
        Defaults to 200.

        Returns:
            itp.interp1d: 1D interpolation object of the DCR vs area
        threshold curve.
        """
        area_hist = np.histogram(df[df['length'] > length_cut]['area'],
                                 bins=bins)
        area_hist_x = area_hist[1]
        area_hist_x = (area_hist_x +
                       (area_hist_x[1] - area_hist_x[0]) / 2)[:-1]
        area_hist_y = area_hist[0]
        DCR_values = np.flip(np.cumsum(np.flip(area_hist_y)))
        DCR_func = itp.interp1d(area_hist_x, DCR_values)

        return DCR_func

    @staticmethod
    def get_DCR(df: pd.DataFrame,
                length_cut_min: int,
                length_cut_max: int,
                pe_area: float,
                sensor_area: float,
                t: float) -> tuple:
        """Compute the dark count rate (DCR) and crosstalk probability
        (CTP) of a dataset.

        Args:
            df (pd.DataFrame): DataFrame of the dataset (single ch)
            length_cut_min (int): low length cut to apply
            length_cut_max (int): high length cut to apply
            pe_area (float): SPE area
            sensor_area (float): effective are of the sensor
            t (float): livetime of the dataset

        Returns:
            tuple: DCR value, error of DCR value, CTP value, error of
        CTP value
        """
        pe_0_5 = pe_area * 0.5
        pe_1_5 = pe_area * 1.5

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

    def get_gain(self, spe_area: float) -> float:
        """Compute the gain given the value of the SPE area

        Args:
            spe_area (float): Area of the SPE peak in integrated ADC
        counts [ADC count * ns].

        Raises:
            ValueError: self.ADC_config not defined for the current dataset.
        Run dataset.define_ADC_config(...)

        Returns:
            float: the calculated gain.
        """

        if not isinstance(self.ADC_config, dict):
            raise ValueError('Define ADC_config first.')

        ADC_range = self.ADC_config['ADC_range']
        ADC_impedance = self.ADC_config['ADC_impedance']
        F_amp = self.ADC_config['F_amp']
        ADC_res = self.ADC_config['ADC_res']
        q_e = self.ADC_config['q_e']

        gain = (ADC_range * spe_area * 1e-9 / ADC_impedance / F_amp /
                ADC_res / q_e)

        return gain

    def compute_properties_of_dataset(self,
                                      length_cut_min: int = 4,
                                      length_cut_max: int = 20) -> pd.DataFrame:
        """Calculate the gain, DCR, CTP and BV for the dataset in a single
        line!

        Args:
            temp (float): temperature to consider in the dataset.
            module (int): module to load.
            channel (str): channel to load.
            length_cut_min (int): lower bound accepted for the length of
        peaks. Defaults to 4.
            length_cut_max (int): upper bound accepted for the length of
        peaks. Defaults to 20.

        Returns:
            pd.DataFrame: dataframe with all the computed properties with
        the columns ['V','T','path','module','channel','Gain','DCR','CTP',
        'DCR_error','CTP_error','BV','OV']
        """

        assert isinstance(self.data, dict), ('Woops, no data found! Load'
                                             ' data into the dataset first')

        voltage_list = self.voltages

        _results_dataset = pd.DataFrame(columns=['T', 'V', 'pe_area', 'Gain',
                                                 'res', 'DCR', 'DCR_error',
                                                 'CTP', 'CTP_error'])

        for _volt in voltage_list:
            # select voltage
            df = self.data[_volt]
            if self.plots_flag == True:
                plot_name_1pe_fit = (f'{self.temp}K_{_volt}V_mod{self.module}_'
                                     f'ch{self.channel}')
            else:
                plot_name_1pe_fit = False

            # Get SPE value from Gaussian fit
            (A, mu, sigma), cov = self.get_1pe_value_fit(
                df, plot=plot_name_1pe_fit)

            # Calculate DCR and CTP
            DCR, DCR_error, CTP, CTP_error = self.get_DCR(
                df=df,
                length_cut_min=length_cut_min,
                length_cut_max=length_cut_max,
                pe_area=mu,
                sensor_area=self.SiPM_config['sensor_area'],
                t=self.SiPM_config['livetime'])

            # Calculate gain
            gain = self.get_gain(mu)

            # Merge into rolling dataframe (I know it's slow... make a PR, pls)
            _results_dataset = pd.concat((_results_dataset,
                                          pd.DataFrame({'T': [self.temp],
                                                        'V': [_volt],
                                                        'pe_area': [mu],
                                                        'Gain': [gain],
                                                        'res': [mu / sigma],
                                                        'DCR': [DCR],
                                                        'DCR_error': [DCR_error],
                                                        'CTP': [CTP],
                                                        'CTP_error': [CTP_error]})
                                          ), ignore_index=True)

        # Compute BV from gain(V) curve and update df
        if self.plots_flag == True:
            plot_BV = f'BV_mod{self.module}_ch{self.channel}'
        else:
            plot_BV = False
        _results_dataset = compute_BV(_results_dataset, plot_BV)

        return _results_dataset


class DCR_run():
    """Collection of all the DCR_datasets results for a run, ie, for all
    the channels and modules, for every available temperatures and voltages.

    The results of every dataset (channel, V, T) is saved on the instance
    DCR_run.results_df .
    """

    __version__ = '0.0.1'

    def __init__(self, run: pylars.utils.input.run,
                 processor: pylars.processing.rawprocessor.run_processor):

        self.run = run
        self.process = processor
        self.datasets = self.process.datasets_df
        self.temperatures = self.get_run_temperatures()
        self.plots_flag = False
        self.analysis_path = (f'{self.run.main_data_path[:-9]}analysis_data'
                              f'/run{self.run.run_number}/')

    def set_plots_flag(self, flag: bool) -> None:
        """Set if computing properties makes plots (True) or not (False).
        Assumes a ./figures/ directory exists.

        Args:
            flag (bool): True for plotting stuff, False for not making nice
        pictures (plots) to hang on the wall. Yes, I hang plots on my bedroom
        wall and it looks nice.
        """
        self.plots_flag = flag

    def get_run_temperatures(self) -> np.array:
        """Get all the temperatures available in the DCR run.

        Returns:
            np.array: array with all the available temperatures.
        """
        temp_list = np.unique(self.datasets['temp'])

        return temp_list

    def initialize_results_df(self) -> None:
        """Initialize a clean results_df instance in the object.
        """

        results_df = pd.DataFrame(columns=['V', 'T', 'module', 'channel',
                                           'Gain', 'DCR', 'CTP', 'DCR_error',
                                           'CTP_error', 'BV', 'OV']
                                  )
        self.results_df = results_df

    def define_run_ADC_config(self, F_amp: float, ADC_range: int = 2.25,
                              ADC_impedance: int = 50, ADC_res: float = 2**14,
                              q_e: float = 1.602176634e-19) -> None:
        """Define the ADC related quantities for the dataset.

        Args:
            F_amp (float): signal amplification from the sensor (pre-amp *
        external amplification on rack)
            ADC_range (int, optional): Range of the ADC [V]. Defaults to 2.25.
            ADC_impedance (int, optional): Impedance of the ADC and cables
        [ohm]. Defaults to 50.
            ADC_res (float, optional): Resolution/bits of the ADC (2**N_bits).
        Defaults to 2**14.
            q_e (float, optional): Element charge of the electron [C].
        Defaults to 1.602176634e-19.
        """
        ADC_config = {'ADC_range': ADC_range,
                      'ADC_impedance': ADC_impedance,
                      'F_amp': F_amp,
                      'ADC_res': ADC_res,
                      'q_e': q_e}

        self.ADC_config = ADC_config

    def define_run_SiPM_config(self, livetime: float,
                               sensor_area: float = 12 * 12,
                               ) -> None:
        """Define the SiPM data related quantities for the dataset.

        Args:
            livetime (float): livetime of the measurement for DCR porpuses.
            sensor_area (float, optional): Area of the photosensor (mm**2).
        Defaults to 12*12.
        """
        SiPM_config = {'livetime': livetime,
                       'sensor_area': sensor_area,
                       }
        self.SiPM_config = SiPM_config

    def load_dataset(self, temp: float,
                     module: int,
                     channel: str) -> DCR_dataset:
        """Create a DCR_dataset object for a (T, mod, ch) configuration and
        load the corresponding data into it.
        ! This assumes processed data is availabel for all the raw files of
        the DCR run datasets !

        Args:
            temp (float): temperature to consider
            module (int): module to load
            channel (str): channel in the module to select

        Returns:
            DCR_dataset: _description_
        """
        particular_DCR_dataset = DCR_dataset(run=self.run,
                                             temperature=temp,
                                             module=module,
                                             channel=channel,
                                             processor=self.process,
                                             )

        particular_DCR_dataset.load_processed_data()

        return particular_DCR_dataset

    def compute_properties_of_ds(self, temp: float,
                                 module: int,
                                 channel: str) -> pd.DataFrame:
        """Loads and computes the properties of a single dataset (temp,
        module, channel) by creating a DCR_dataset object and calling its
        methods.

        Args:
            temp (float): temperature
            module (int): module
            channel (str): channel

        Returns:
            pd.DataFrame: dataframe
        """

        assert isinstance(self.SiPM_config, dict), 'No SiPM_config found!'
        assert isinstance(self.ADC_config, dict), 'No ADC_config found!'

        ds = self.load_dataset(temp, module, channel)
        ds.set_plots_flag(self.plots_flag)
        ds.ADC_config = self.ADC_config
        ds.SiPM_config = self.SiPM_config

        ds_results = ds.compute_properties_of_dataset()

        # Add module and channel columns
        module_Series = pd.Series([module] * len(ds_results), name='module')
        channel_Series = pd.Series([channel] * len(ds_results), name='channel')
        ds_results = pd.concat([ds_results, module_Series, channel_Series],
                               axis=1)
        return ds_results

    def read_channel_map(self, path_to_map: str) -> None:
        """Define the active modules and channels for the run.
        UNDER CONSTRUCTION. I know it doesn't work that nicely but
        I need to have somthing that works for now, so...
        Define it with self.channel_map = dict(mod:[ch#,...],...)

        Args:
            path_to_map (str): path to csv file with the channel map
        """
        if self.run.run_number == 7:
            channel_map = {0: ['wf0', 'wf3', 'wf4', 'wf6'],
                           1: ['wf0', 'wf3', 'wf4', 'wf6'], }
        else:
            channel_map = pd.read_csv(path_to_map)

        self.channel_map = channel_map

    def compute_properties_of_run(self) -> pd.DataFrame:
        """Loads and computes the properties of ALL the datasets.

        Returns:
            pd.DataFrame: The results.
        """
        self.initialize_results_df()

        for temperature in self.temperatures:
            for module in self.channel_map.keys():
                for channel in tqdm(self.channel_map[module],
                                    (f'Computing properties for '
                                     f'T={temperature}; module '
                                     f'{module}: ')):
                    _ds_results = self.compute_properties_of_ds(
                        temp=temperature,
                        module=module,
                        channel=channel)

                    self.results_df = pd.concat([self.results_df, _ds_results],
                                                ignore_index=True)

        datetime.now()

    def save_results(self, custom_name:str = str(int(
            datetime.timestamp(datetime.now())))) -> None:
        f"""Save dataframe of results to a hdf5 file. Saved files go to 
        {self.analysis_path} .

        Args:
            name (str): name to give the file (without extension). Defaults to
        timestamp of 
        """
        assert isinstance(self.results_df, pd.DataFrame), ("Trying to save "
            "results that do not exist in the object, c'mon, you know better.")
        assert len(self.results_df) > 0, ("Results df is empty, please compute"
            "something before trying to save, otherwire it's just a waste of "
            "disk space")

        name = f'DCR_results_{custom_name}'
        self.results_df.to_hdf(self.analysis_path + name + '.h5', 'df')
        print('Saved results to ')
    
    def load_results(self, name:str) -> None:
        f"""Load dataframe of results from a hdf5 file. Looks for files in 
        {self.analysis_path} .

        Args:
            name (str): name of the file to load (without extension)
        """
        assert isinstance(self.results_df, pd.DataFrame), ("Trying to save "
            "results that do not exist in the object, c'mon, you know better.")
        assert len(self.results_df) > 0, ("Results df is empty, please compute"
            "something before trying to save, otherwire it's just a waste of "
            "disk space")

        _df = pd.read_hdf(self.analysis_path + name + '.h5')
        self.results_df = _df
        
        
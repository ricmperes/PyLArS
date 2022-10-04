import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylars
import pylars.plotting.plotanalysis
import pylars.utils.input
import pylars.utils.output
import scipy.interpolate as itp
from scipy.optimize import curve_fit

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
                          df:pd.DataFrame, 
                          length_cut:int = 5, 
                          plot:bool or str = False) -> tuple:
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
                DCR_der_y_points,min_area_x)
        
        area_hist = np.histogram(df[df['length']> length_cut]['area'], 
            bins = np.linspace(0.9*min_area_x, 1.1*min_area_x, 300))
        area_hist_x = area_hist[1]
        area_hist_x = (area_hist_x + (area_hist_x[1]-area_hist_x[0])/2)[:-1]
        area_hist_y = area_hist[0]
        
        (A, mu, sigma), cov = curve_fit(Gaussean, area_hist_x, area_hist_y,
                                        p0 = (2000, min_area_x, 0.05*min_area_x))
    
        if plot != False:
            pylars.plotting.plotanalysis.plot_SPE_fit(
                df, length_cut, plot, area_hist_x, min_area_x, A, mu, sigma)

        return (A, mu, sigma), cov

    @classmethod
    def get_1pe_rough(cls, df:pd.DataFrame, 
                      length_cut:int, 
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
        grad_spline = itp.UnivariateSpline(area_hist_x, grad, s = len(area_hist_x)*3)
        _x = np.linspace(area_hist_x[0],area_hist_x[-1], 500)
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
    def get_DCR(df:pd.DataFrame, 
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


class fingerplot_dataset(DCR_dataset):
    """Object class to hold finger spectrum related instances and
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

    def get_voltages_available(self) -> np.array:
        """Checks the loaded run for which voltages are available for the
        defined temperature.

        Returns:
            np.array: array of the available voltages
        """

        voltages = []
        for _dataset in self.run.datasets:
            if (_dataset.temp == self.temp) and (_dataset.kind == 'fplt'):
                voltages.append(_dataset.vbias)
        voltages = np.unique(voltages)

        return voltages

    def load_processed_data(self, force_processing: bool = False) -> dict:
        self.data = {}
        for _voltage in self.voltages:
            processed_data = pylars.utils.output.processed_dataset(
                run=self.run,
                kind='fplt',
                vbias=_voltage,
                temp=self.temp,
                path_processed=('/disk/gfs_atp/xenoscope/SiPMs/char_campaign/'
                                'processed_data/'),
                process_hash=self.process.hash)
            processed_data.load_data(force=force_processing)

            _df = processed_data.data
            mask = ((_df['module'] == self.module) &
                    (_df['channel'] == self.channel))

            self.data[_voltage] = _df[mask].copy()

    def make_finger_plot(df, length_cut_min,
                         length_cut_max, plot=False,
                         ax=None):
        _cuts = ((df['length'] > length_cut_min) &
                 (df['length'] < length_cut_max)
                 )

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=400)
        area_hist = plt.hist(
            df[_cuts]['area'],
            bins=1000,
            histtype='step',
            color='k')

        area_hist_x = area_hist[1]
        area_hist_x = (
            area_hist_x + (area_hist_x[1] - area_hist_x[0]) / 2)[:-1]
        area_hist_y = area_hist[0]

        (A, mu, sigma), cov = curve_fit(Gaussean, area_hist_x, area_hist_y,
                                        p0=(2000, 30000, 0.05 * 30000))

        if plot != False:
            _x = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 400)
            ax.hist(df[_cuts]['area'], bins=1000, histtype='step', color='k')
            ax.plot(_x, Gaussean(_x, A, mu, sigma), c='r')

            ax.yscale('log')
            ax.ylabel('# Events')
            ax.title('Module 0 | Channel 0')
            ax.xlabel('Peak Area [integrated ADC counts]')
            if isinstance(plot, str):
                fig.savefig(f'figures/fingerplot_{plot}.png')
            else:
                plt.show()
        return mu, sigma


def Gaussean(x, A, mu, sigma):
    y = A * np.exp(-((x - mu) / sigma)**2 / 2) / np.sqrt(2 * np.pi * sigma**2)
    return y

r"""`breakdown.py` contains the methods and tools for analysis of an LED ON
dataset and functions to compute the breakdown voltage out of V vs Gain plots
of LED OFF runs.
"""
import copy
from typing import Tuple, Union

import numpy as np
import pandas as pd
import pylars
import pylars.plotting.plotanalysis
import pylars.utils.input
import pylars.utils.output
from pylars.utils.common import get_peak_rough_positions
from scipy import stats
from tqdm.autonotebook import tqdm


class BV_dataset():
    """Object class to hold breakdown voltage related instances and
    methods. Collects all the data and properties of a single MMPC,
    meaning the pair (module, channel) for all the available voltages at
    a certain temperature.
    """

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
                pictures (plots) to hang on the wall. Yes, I hang plots on
                my bedroom wall and it looks nice.
        """
        self.plots_flag = flag

    def get_voltages_available(self) -> np.ndarray:
        """Checks the loaded run for which voltages are available for the
        defined temperature.

        Returns:
            np.ndarray: array of the available voltages
        """

        voltages = []
        for _dataset in self.run.datasets:
            if (_dataset.temp == self.temp) and (_dataset.kind == 'BV'):
                voltages.append(_dataset.vbias)
        voltages = np.unique(voltages)

        return voltages

    def load_processed_data(self, force_processing: bool = False) -> None:
        """For all the voltages of a BV_dataset (same temperature) looks
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
        self.data = {}
        for _voltage in tqdm(self.voltages,
                             desc=f'Loading processed data for BV ' +
                             f'data at {self.temp}K: ',
                             total=len(self.voltages)):
            processed_data = pylars.utils.output.processed_dataset(
                run=self.run,
                kind='BV',
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

    def compute_BV_LED_simple(self, LED_position: int,
                              plot: bool = False) -> tuple:
        """Comute BV for a given T dataset with the first no-noise peak
        of the area spectrum given width and position cut.

        Args:
            LED_position (int): number of sample where to center the
                position cut.
            plot (bool, optional): Show the plot, save the plot or only
                compute output. Defaults to False.

        Returns:
            tuple: Calculated BV, error on the fit and r2 of the linear
                regression. Returns on the form `(BV, BV_std, r2)`.
        """

        bias_voltage = []
        good_peaks = []
        v_list = self.voltages

        for i, v in enumerate(v_list):

            # For very low bias voltage the SiPM shows a pulse when the LED
            # shines but it could be not yet in Geiger-Mode.
            if v < 48:
                continue

            _df = self.data[v]
            _cuts = (
                (_df['length'] > 3) &
                (_df['position'] > LED_position - 10) &
                (_df['position'] < LED_position + 20)
            )
            peaks, peak_prop = get_peak_rough_positions(
                _df['area'],
                _cuts,
                bins=np.linspace(0, np.percentile(_df['area'], 99), 1500),
                plot=False)

            if len(peaks > 5):
                # likely there is a nice LED fingerplot dominating everything
                first_good_peak = np.median(_df[_cuts]['area'])

            elif len(peaks) == 0:
                print('Could not compute LED area for V=', v)
                continue
            elif len(peaks) > 0:
                if (peaks[0] > 3500) or (len(peaks) == 1):
                    first_good_peak = peaks[0]
                else:
                    first_good_peak = peaks[1]
            # elif len(peaks) == 0 and peaks[0] > 10000:
            #    spe = peaks[0]
            else:
                print('Could not compute LED area for V=', v)
                continue

            bias_voltage.append(v)
            good_peaks.append(first_good_peak)

        linres = stats.linregress(good_peaks, bias_voltage)

        if plot != False:
            pylars.plotting.plotanalysis.plot_BV_fit(
                plot=f'{self.module}_{self.channel}',
                temperature=self.temp,
                voltages=bias_voltage,
                gains=good_peaks,
                a=linres.slope,  # type: ignore
                b=linres.intercept,  # type: ignore
                _breakdown_v=linres.intercept,  # type: ignore
                _breakdown_v_error=linres.intercept_stderr)  # type: ignore

        return (linres.intercept, linres.intercept_stderr,  # type: ignore
                linres.rvalue**2)  # type: ignore


def compute_BV_df(df_results: pd.DataFrame,
                  plot: Union[bool, str] = False
                  ) -> Tuple[pd.DataFrame, float, float, float]:
    """Computes the breakdown voltage with a linear fit of gain over
    V points.

    Takes a dataframe with the collumns 'T' (temperature), 'V' (voltage)
    and 'gain' (self-explanatory, c'mon...)

    Args:
        df_results (pd.DataFrame): dataframe with the processed datasets
            calculated gains.
        plot (boolorstr, optional): flag for plotting choice. Defaults to
            False.

    Returns:
        pd.DataFrame: copy of input dataset with the extra collumns 'BV'
            (breakdown voltage) and 'OV' (over-voltage).
    """

    df_results = copy.deepcopy(df_results)  # just to be sure
    temp_list = np.unique(df_results['T'])
    df_results = pd.concat([df_results,
                            pd.Series(np.zeros(len(df_results)), name='BV')
                            ], axis=1
                           )

    for _temp in temp_list:
        volt_list_in_temp = np.unique(
            df_results[df_results['T'] == _temp]['V'])
        gain_list_in_temp = np.unique(
            df_results[df_results['T'] == _temp]['Gain'])

        linres = stats.linregress(gain_list_in_temp, volt_list_in_temp)
        a = linres.slope,  # type: ignore
        b = linres.intercept,  # type: ignore
        b_err = linres.intercept_stderr  # type: ignore
        a = float(a[0])  # why is this needed? Who knows...
        b = float(b[0])
        b_err = float(b_err)

        _breakdown_v = b  # type: ignore
        _breakdown_v_error = b_err  # type: ignore
        # print(_breakdown_v)
        df_results.loc[df_results['T'] == _temp, 'BV'] = _breakdown_v

        if plot != False:
            pylars.plotting.plotanalysis.plot_BV_fit(plot, _temp,
                                                     volt_list_in_temp,
                                                     gain_list_in_temp,
                                                     a, b,
                                                     _breakdown_v,
                                                     _breakdown_v_error)

    df_results['OV'] = df_results['V'] - df_results['BV']

    return df_results, a, _breakdown_v, _breakdown_v_error  # type: ignore


def compute_BV_DCRds_results(
        results_df: pd.DataFrame, plot: bool) -> pd.DataFrame:
    """Compute BVs for a finished DCR analysis results dataframe.

    Args:
        results_df (pd.DataFrame): DCR analysis result df (without BV)
        plot (bool): save plot of V vs Gain fit.

    Returns:
        pd.DataFrame: a df with all the BV and their std.
    """
    temps = np.unique(results_df['T'])
    BVs = pd.DataFrame(columns=['T', 'module', 'channel', 'BV', 'BV_error'])
    for _temp in temps:
        _select_temp = (results_df['T'] == _temp)
        for mod in np.unique(results_df[_select_temp]['module']):
            _select_mod = (results_df['module'] == mod)
            for ch in np.unique(
                    results_df[_select_temp & _select_mod]['channel']):
                _select = (_select_temp & _select_mod &
                           (results_df['channel'] == ch))
                df = results_df[_select]
                if plot == True:
                    plot_BV = f'BV_mod{mod}_ch{ch}'
                else:
                    plot_BV = False

                try:
                    df, a, bv, bv_err = compute_BV_df(df,  # type: ignore
                                                      plot=plot_BV)
                    BVs = pd.concat([BVs,
                                     pd.DataFrame({'T': [_temp],
                                                   'module': [mod],
                                                   'channel': [ch],
                                                   'BV': [bv],
                                                   'BV_error': [bv_err]})],
                                    ignore_index=True)
                except BaseException:
                    BVs = pd.concat([BVs,
                                    pd.DataFrame({'T': [_temp],
                                                  'module': [mod],
                                                  'channel': [ch],
                                                  'BV': [np.nan],
                                                  'BV_error': [np.nan]})],
                                    ignore_index=True)
    return BVs

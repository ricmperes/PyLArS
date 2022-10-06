import copy

import numpy as np
import pandas as pd
import pylars
from scipy.optimize import curve_fit

from .common import Gaussean, func_linear


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

    def get_voltages_available(self) -> np.array:
        """Checks the loaded run for which voltages are available for the
        defined temperature.

        Returns:
            np.array: array of the available voltages
        """

        voltages = []
        for _dataset in self.run.datasets:
            if (_dataset.temp == self.temp) and (_dataset.kind == 'BV'):
                voltages.append(_dataset.vbias)
        voltages = np.unique(voltages)

        return voltages

    def load_processed_data(self, force_processing: bool = False) -> dict:
        self.data = {}
        for _voltage in self.voltages:
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

            self.data[_voltage] = _df[mask].copy()

        return self.data  # not needed but doesn't harm and can be useful


def compute_BV(df_results: pd.DataFrame,
               plot: bool or str = False) -> pd.DataFrame:
    """Computes the breakdown voltage with a linear fit of gain over V points.
    Takes a dataframe with the collumns 'T' (temperature), 'V' (voltage) and
    'gain' (self-explanatory, c'mon...)

    Args:
        df_results (pd.DataFrame): dataframe with the processed datasets
    calculated gains.
        plot (boolorstr, optional): flag for plotting choice. Defaults to False.

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
        (a, b), cov = curve_fit(func_linear, volt_list_in_temp, gain_list_in_temp)
        _breakdown_v = -b / a
        # print(_breakdown_v)
        df_results.loc[df_results['T'] == _temp, 'BV'] = _breakdown_v

        if plot != False:
            pylars.plotting.plotanalysis.plot_BV_fit(plot, _temp,
                                                     volt_list_in_temp,
                                                     gain_list_in_temp, a, b,
                                                     _breakdown_v)

    df_results['OV'] = df_results['V'] - df_results['BV']

    return df_results

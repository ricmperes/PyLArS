import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import pylars


class LED_window():
    '''Class to hold analysis of LED ON data with integrating window.
    Use independently of BV datasets.
    '''

    def __init__(self,
                 led_window: Tuple[int, int],
                 led_data_path: str) -> None:
        self.led_window = led_window
        self.led_data_path = led_data_path
        self.baseline_samples = 50
        self.files_df = self.find_files()

    def find_files(self) -> pd.DataFrame:  # type: ignore
        '''Find the raw data file and the LED data file.
        TODO
        '''
        pass

        # files = glob.glob(self.led_data_path + '**/*.root', recursive=True)
        # _dfs = []
        # for f in tqdm(files, desc='Finding LED data files: '):
        #     try:
        #         _module = int(f.split('/')[-1].split('_')[4])
        #         _width, _voltage = self.get_LED_width_voltage(f)
        #         _dfs.append({'LEDwidth': _width,
        #                      'LEDvoltage': _voltage,
        #                      'module': _module,
        #                      'path': f})
        #     except:
        #         print(f'Failed to get info for file {f}')
        #         continue
        # files_df = pd.DataFrame(_dfs)
        # files_df.sort_values(by=['LEDwidth', 'LEDvoltage', 'module'],
        #              ignore_index = True, inplace=True)
        # del _dfs, files

        # return files_df

    @staticmethod
    def get_LED_width_voltage(path):
        """Read LED width and voltage from the file path."""

        root_name_list = path.split('/')[-1].split('_')
        pulser_width = int(root_name_list[0][:-2])
        voltage = float(root_name_list[1] + '.' + root_name_list[2][0])
        return pulser_width, voltage

    def process_dataset(self, data_path: str, module: int = 0) -> pd.DataFrame:
        '''Process a dataset with fixwindowprocessor. Requires a raw data file
        and a LED window specified.
        '''

        processor = pylars.processing.fixwindowprocessor.window_processor(
            baseline_samples=self.baseline_samples,
            led_window=(self.led_window[0], self.led_window[1]))

        processor.load_raw_data(path_to_raw=data_path, module=module)
        df_processed = processor.process_all_channels()
        return df_processed

    def process_all_datasets(self) -> None:
        '''Process all datasets in the LED data path.
        '''

        if not hasattr(self, 'files_df'):
            raise AssertionError('No files found. Run find_files first.')

        _dfs = []
        # for i, row in tqdm(self.files_df.iterrows(),  # type: ignore
        #                    total=len(self.files_df),  # type: ignore
        #                    desc='Processing LED data: '):^
        for i, row in self.files_df.iterrows():
            _df = self.process_dataset(row['path'], module=row['module'])
            _df['Vbias'] = row['Vbias']
            _df['LEDwidth'] = row['LEDwidth']
            _df['LEDvoltage'] = row['LEDvoltage']
            _df['module'] = row['module']
            _dfs.append(_df)
        df_processed = pd.concat(_dfs)
        del _dfs
        df_processed.sort_values(by=['Vbias', 'LEDvoltage',
                                     'LEDwidth', 'module',
                                     'channel', 'wf_number'])

        self.df_processed = df_processed

    def check_settings_available(self):
        """Set led_voltages, led_widths and sipms_voltages as object
        attributes.
        """

        self.led_voltages = self.files_df['LEDvoltage'].unique()
        self.led_widths = self.files_df['LEDwidth'].unique()
        self.sipms_voltages = self.files_df['Vbias'].unique()

    def get_1_pe_fit_led(self,
                         df_processed: pd.DataFrame,
                         module: int,
                         channel: str) -> Tuple[float, float,
                                                float, np.ndarray]:
        """Fit the SPE peak of an LED area hsitogram for a given module and
        channel. Uses scipy.curve_fit with a Gaussian function.

        Args:
            df_processed (pd.DataFrame): dataframe with processed data
            module (int): module to consider
            channel (str): channel to consider

        Returns:
            Tuple[float, float, float, np.ndarray]: A, mu, sigma,
                and cov resulting of the fit
        """

        df_processed_mask = (
            (df_processed['module'] == module) &
            (df_processed['channel'] == channel))

        hist = np.histogram(df_processed[df_processed_mask]['led_area'],
                            bins=np.linspace(-2000, 20000, 300))
        middle_bins = (hist[1][:-1] + hist[1][1:]) / 2

        try:
            peaks, properties = find_peaks(hist[0],
                                           prominence=100,
                                           distance=5)
            spe_rough = middle_bins[peaks[1]]
        except BaseException:
            spe_rough = 2500
        #if (spe_rough -2000) > 1000: spe_rough = 2000

        spe_mask = np.abs(middle_bins - spe_rough) < spe_rough * 0.5
        (A, mu, sigma), cov = curve_fit(pylars.utils.common.Gaussian,
                                        middle_bins[spe_mask],
                                        hist[0][spe_mask],
                                        p0=[2000, spe_rough,
                                            spe_rough * 0.05])
        return A, mu, sigma, cov

    def get_occupancy(self, results_one_channel: pd.DataFrame,
                      mu: float, mu_err: float,
                      method: str = 'mean') -> Tuple[float, float]:
        """Calculate the occupancy of an LED dataset for a given channel.

        Args:
            results_one_channel (pd.DataFrame): dataframe with the results
                of a single channel
            mu (float): the SPE fit mean in ADCcounts
            mu_err (float): error in mu in ADCcounts
            method (str, optional): Method to calculate the occupancy. Either
                'mean' or 'median'. Defaults to 'mean'.

        Returns:
            Tuple[float, float]: Occupancy and its error. Uses mean by default!
        """

        if method == 'mean':
            med = np.mean(results_one_channel['led_area'])
        elif method == 'median':
            med = np.median(results_one_channel['led_area'])
        else:
            raise ValueError(
                'Method for occupancy calculation neither mean nor median')

        med_err = np.std(
            results_one_channel['led_area']) / np.sqrt(
                len(results_one_channel['led_area']))

        occ = med / mu
        occ_err = ((med_err / mu)**2 + ((med / mu**2) * mu_err)**2)**0.5
        return occ, occ_err

    def calculate_gain_occ(self, processed_df_single_led: pd.DataFrame,
                           module: int, channel: str,
                           occ_method: str = 'mean') -> Tuple[float, float,
                                                              float, float]:
        """Calculates the gain, occupancy and their errors for a given module
        and channel from a dataframe with the processed LED data.

        Args:
            processed_df_single_led (pd.DataFrame): dataframe with the
                processed LED data
            module (int): module to consider
            channel (str): channel to consider
            occ_method (str, optional): Method to calculate the occupancy.
                'mean' or 'median'. Defaults to 'mean'.

        Returns:
            Tuple[float, float, float, float]: Return the gain, gain_err,
                occ, occ_err of the requested channel.
        """

        A, mu, sigma, cov = self.get_1_pe_fit_led(processed_df_single_led,
                                                  module,
                                                  channel)
        A_err, mu_err, sigma_err = np.sqrt(np.diag(cov))

        gain = pylars.utils.common.get_gain(F_amp=20, spe_area=mu) / 1e6
        gain_err = pylars.utils.common.get_gain(
            F_amp=20, spe_area=mu_err) / 1e6

        results_mask = ((processed_df_single_led['module'] == module) &
                        (processed_df_single_led['channel'] == channel)
                        )
        occ, occ_err = self.get_occupancy(processed_df_single_led[results_mask],
                                          mu, mu_err, method=occ_method)

        return gain, gain_err, occ, occ_err

    def calculate_all_gains_occ(self):
        '''Calculate gains and occupancies for all channels available in the
        processed data. The results are stored in self.results_df and a
        registry of failed processing in self.failed_calculation_df.
        '''

        # Check if processed data exists
        if not hasattr(self, 'df_processed'):
            raise AssertionError(
                'No processed data found. Run process_all_datasets first.')

        results_df = pd.DataFrame(columns=['Vbias', 'LEDvoltage',
                                           'LEDwidth', 'module',
                                           'channel', 'gain',
                                           'gain_err', 'occ',
                                           'occ_err'])
        failed_calculation_df = pd.DataFrame(columns=['Vbias', 'LEDvoltage',
                                                      'LEDwidth', 'module',
                                                      'channel'])

        # for i, row in tqdm(self.files_df.iterrows(),
        #                    total=len(self.files_df),
        #                    desc='Calculating gains and occupancies: '):
        for i, row in self.files_df.iterrows():
            _vbias = row['Vbias']
            _led_voltage = row['LEDvoltage']
            _led_width = row['LEDwidth']
            _module = row['module']

            _df_select_processed = self.df_processed[
                (self.df_processed['Vbias'] == _vbias) &
                (self.df_processed['LEDvoltage'] == _led_voltage) &
                (self.df_processed['LEDwidth'] == _led_width)]

            _channels = _df_select_processed[
                _df_select_processed['module'] == _module]['channel'].unique()

            for _channel in _channels:
                try:
                    _gain, _gain_err, _occ, _occ_err = self.calculate_gain_occ(
                        _df_select_processed, _module, _channel)

                    results_df = pd.concat(
                        (results_df,
                            pd.DataFrame({'Vbias': [_vbias],
                                          'LEDvoltage': [_led_voltage],
                                          'LEDwidth': [_led_width],
                                          'module': [_module],
                                          'channel': [_channel],
                                          'gain': [_gain],
                                          'gain_err': [_gain_err],
                                          'occ': [_occ],
                                          'occ_err': [_occ_err]}),
                         ), ignore_index=True)
                except BaseException:
                    failed_calculation_df = pd.concat(
                        (failed_calculation_df,
                         pd.DataFrame({'Vbias': [_vbias],
                                       'LEDvoltage': [_led_voltage],
                                       'LEDwidth': [_led_width],
                                       'module': [_module],
                                       'channel': [_channel]})),
                        ignore_index=True)

        self.failed_calculation_df = failed_calculation_df
        self.results_df = results_df

    def save_gain_results(self, name: str = ''):
        """Save the results of the gain and occupancy calculations to a csv
        file.

        Args:
            name (str, optional): name to give to the file. Defaults to '',
            saving a file with the current timestamp.

        Raises:
            AssertionError: raised if no results dataframe is found in the
                object.
        """

        # Check if results exist
        if not hasattr(self, 'results_df'):
            raise AssertionError(
                'No results found. Run calculate_all_gains_occ first.')

        # Save results
        if name == '':
            now = datetime.datetime.now().isoformat()
            self.results_df.to_csv(f'gain_results_{str(now)}.csv', index=False)
        else:
            self.results_df.to_csv(f'{name}.csv', index=False)

    def print_gains_occ_for_wiki(self):
        if not hasattr(self, 'df_gains'):
            raise AssertionError('No computed gains found. Run '
                                 'calculate_all_gains_occ first.')
        for i, row in self.df_gains.iterrows():  # type: ignore
            print(f"| {row['tile']} | {row['gain']:.3f} $\\pm$ "  # type: ignore
                  # type: ignore
                  f"{row['gain_err']:.3f} | {row['occ']:.3f} $\\pm$ "
                  f"{row['occ_err']:.3f} |")

    @staticmethod
    def export_gains(gain_evolution: pd.DataFrame,
                     method: str = 'mean',
                     tag: str = 'vx'):
        """Export the gains as a single number to a csv file. Method can be
            'mean', 'median' or 'last'. Default is 'mean'.

        Args:
            gain_evolution (pd.DataFrame): dataframe with the gain evolution
            method (str, optional): method to calculate the number to use as
                gain. Can be 'mean', 'median' or 'last'. Defaults to 'mean'.
            tag (str, optional): Tag to give the gains in the form v#
                (v0,v1,v2,...). Defaults to 'vx'.

        Raises:
            ValueError: Raised if the method is not recognized.
        """
        if method == 'mean':
            gains = gain_evolution.groupby(
                ['module', 'tile', 'channel']).mean()
        elif method == 'median':
            gains = gain_evolution.groupby(
                ['module', 'tile', 'channel']).median()
        elif method == 'last':
            gains = gain_evolution.groupby(
                ['module', 'tile', 'channel']).last()
        else:
            raise ValueError('Method not recognized')

        gains.to_csv(f'gains_{tag}.csv')

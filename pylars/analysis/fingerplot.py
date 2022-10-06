import numpy as np
import pylars
from matplotlib import pyplot as plt
from pylars.analysis.common import Gaussean
from scipy.optimize import curve_fit

from .darkcount import DCR_dataset


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

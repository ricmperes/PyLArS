import numpy as np
from .peaks import peak_processing


class waveform_processing():
    """All the things waveforms.
    """

    @classmethod
    def get_std_rough(cls, waveform_array, baseline_samples):
        """Returns the std value of the first self.baseline_samples
        as a good estimate for the std of the baseline."

        Args:
            waveform_array (_type_): _description_

        Returns:
            _type_: _description_
        """
        std_rough = np.std(waveform_array[:baseline_samples])
        return std_rough

    @classmethod
    def get_baseline_rough(cls, waveform_array, baseline_samples):
        """Returns the median value of the first self.baseline_samples
        as a good estimate for the baseline.

        Args:
            waveform_array (_type_): _description_

        Returns:
            _type_: _description_
        """
        baseline_rough = np.median(waveform_array[:baseline_samples])
        return baseline_rough

    @classmethod
    def process_waveform(cls, waveform, baseline_samples: int,
                         sigma_level: float = 5):

        baseline_rough = cls.get_baseline_rough(waveform, baseline_samples)
        std_rough = cls.get_std_rough(waveform, baseline_samples)

        peaks = peak_processing.find_peaks_simple(
            baseline_rough, std_rough, sigma_level)

        areas = peak_processing.get_all_areas(waveform, peaks)
        lengths = peak_processing.get_all_lengths(peaks)
        positions = peak_processing.get_all_positions(peaks)

        return areas, lengths, positions

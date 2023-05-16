import numpy as np

from .pulses import pulse_processing


class waveform_processing():
    """All the things waveforms.
    """

    @classmethod
    def get_std_rough(cls, waveform_array: np.ndarray,
                      baseline_samples: int) -> float:
        """Returns the std value of the first self.baseline_samples
        as a good estimate for the std of the baseline.
        """
        std_rough = np.std(waveform_array[:baseline_samples])
        return std_rough

    @classmethod
    def get_baseline_rough(cls, waveform_array: np.ndarray,
                           baseline_samples: int) -> float:
        """Returns the median value of the first self.baseline_samples
        as a good estimate for the baseline.
        """
        baseline_rough = np.median(waveform_array[:baseline_samples])
        return baseline_rough

    @classmethod
    def process_waveform(cls, waveform: np.ndarray, baseline_samples: int,
                         sigma_level: float = 5, negative_polarity:bool = True):
        """Main process function on the waveform level. Finds pulses above
        (actually bellow) threshold and calls pulse level processing like
        area, length and position.

        Args:
            waveform (np.ndarray): ADC counts of the waveform;
            baseline_samples (int): how many samples to take for the
        baseline calculation (at the begining of the waveform);
            sigma_level (float, optional): define threshodl by "sigma_level"
        times the std of the computed baseline. Defaults to 5.

        Returns:
            tuple: areas, lengths, positions
        """

        baseline_rough = cls.get_baseline_rough(waveform, baseline_samples)
        std_rough = cls.get_std_rough(waveform, baseline_samples)

        pulses = pulse_processing.find_pulses_simple(
            waveform, baseline_rough, std_rough, sigma_level,
            negative_polarity=negative_polarity, baseline_subtracted=False)

        # handle case where no pulses were found
        if len(pulses[0]) == 0:
            return [], [], [], []

        areas = pulse_processing.get_all_areas(
            waveform, pulses, baseline_rough,
            negative_polarity = negative_polarity)
        lengths = pulse_processing.get_all_lengths(pulses)
        positions = pulse_processing.get_all_positions(pulses)
        amplitudes = pulse_processing.get_all_amplitudes(
            waveform, pulses, baseline_rough,
            negative_polarity = negative_polarity)

        return areas, lengths, positions, amplitudes

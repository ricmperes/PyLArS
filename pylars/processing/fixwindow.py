import numpy as np
from typing import Tuple


class fixed_window_processing():
    """Functions for LED window analysis. Since no peak finding is required,
    all the waveforms can be computed in a single step for each channel.
    """

    @classmethod
    def process_all_waveforms(cls, waveforms: np.ndarray,
                              baseline_samples: int,
                              led_window: Tuple[int, int],
                              dt: int = 10,
                              negative_polarity: bool = True):
        """Main process function on the waveform level for LED window
        analysis. All wfs are processed in a single step.

        Args:
            waveforms (np.ndarray): ADC counts of the waveform;
            baseline_samples (int): how many samples to take for the
        baseline calculation (at the begining of the waveform);
            led_window (tuple, optional): LED window in samples to consider.

        Returns:
            tuple: amplitudes, areas, ADCcounts of max/min
        """

        baselines = np.median(waveforms[:, :baseline_samples], axis=1)

        if negative_polarity:
            polarity = -1
            ADCcounts = np.min(waveforms[:, led_window[0]:led_window[1]],
                               axis=1)
            amplitudes = polarity * np.min(waveforms[:, led_window[0]:led_window[1]]
                                           - baselines[:, np.newaxis],
                                           axis=1)
        else:
            polarity = 1
            ADCcounts = np.max(waveforms[:, led_window[0]:led_window[1]],
                               axis=1)
            amplitudes = np.max(waveforms[:, led_window[0]:led_window[1]]
                                - baselines[:, np.newaxis],
                                axis=1)

        sum = np.sum(waveforms[:, led_window[0]:led_window[1]]
                     - baselines[:, np.newaxis],
                     axis=1)

        areas = (polarity * dt * sum)

        return amplitudes, areas, ADCcounts

import numpy as np
import numba as nb
from typing import List


class pulse_processing():
    """All the things pulses.
    """

    @classmethod
    def get_area(cls, waveform: np.ndarray, baseline_value: float,
                 pulse_start: int, pulse_end: int, dt: int = 10,
                 negative_polarity: bool = True,
                 baseline_subtracted: bool = False) -> float:
        """Get area of a single identified pulse in a waveform. Points to
        the numbafied function `_get_area`.
        """
        area_under = _get_area(
            waveform,
            baseline_value,
            pulse_start,
            pulse_end,
            dt,
            negative_polarity,
            baseline_subtracted)
        return area_under

    @classmethod
    def get_all_areas(cls, waveform: np.ndarray, pulses: list,
                      baseline_value: float, dt: int = 10,
                      negative_polarity: bool = True,
                      baseline_subtracted: bool = False) -> np.ndarray:
        """Compute the areas of all the pulses in a waveform.
        TO DO: use np.apply_along_axis or similar and see if
        there is speed improvement.
        """
        areas = np.zeros(len(pulses))
        for i, _pulse in enumerate(pulses):
            areas[i] = cls.get_area(
                waveform, baseline_value, _pulse[0], _pulse[-1],
                dt,
                negative_polarity,
                baseline_subtracted)
        return areas

    @classmethod
    def get_all_lengths(cls, pulses: list) -> list:
        """Compute the lengths of all the pulses in a waveform.
        (It's faster without @numba.njit)

        Args:
            pulses (list): list of arrays where the elements are the index
                of samples within each pulse.
        Returns:
            list: list with the lenght for each pulse.
        """
        lengths = [len(_pulse) for _pulse in pulses]
        return lengths

    @classmethod
    def get_all_positions(cls, pulses: list) -> list:
        """Calcultes the initial position of the identified pulse
        in number of samples.

        (Faster without numba...?)

        Args:
            pulses (list): array of identified pulses.

        Returns:
            list: list of positions of pulses.
        """
        positions = [_pulse[0] for _pulse in pulses]
        return positions

    @classmethod
    def get_amplitude(cls, waveform: np.ndarray, baseline_value: float,
                      peak_start: int, peak_end: int,
                      negative_polarity: bool = True,
                      baseline_subtracted: bool = False) -> float:
        """Get area of a single identified pulse in a waveform. Points to
        the numbafied function `_get_amplitude`.
        """
        amplitude = _get_amplitude(
            waveform,
            baseline_value,
            peak_start,
            peak_end,
            negative_polarity,
            baseline_subtracted)
        return amplitude

    @classmethod
    def get_all_amplitudes(cls, waveform: np.ndarray, pulses: list,
                           baseline_value: float,
                           negative_polarity: bool = True,
                           baseline_subtracted: bool = False) -> List[float]:
        """Calcultes the max amplitude of the identified pulse
        in number of samples.

        (Faster without numba...?)

        Args:
            waveform (np.ndarray): 1D array of all the ADC counts where each
                element is a sample number in the waveform. The length of the
                array is the ammount of samples in the waveform.
            pulses (np.ndarray): array of identified pulse.
            baseline_value (float): value of ADC counts to use as baseline.

        Returns:
            list: list of amplitudes of pulses.
        """
        amplitudes = np.zeros(len(pulses))
        for i, _peak in enumerate(pulses):
            amplitudes[i] = cls.get_amplitude(
                waveform, baseline_value, _peak[0], _peak[-1],
                negative_polarity,
                baseline_subtracted)
        return amplitudes.tolist()

    @classmethod
    def split_consecutive(cls, array: np.ndarray, stepsize: int = 1) -> list:
        """Splits an array into several where values are consecutive
        to each other. Points to numbafied function _split_consecutive().
        """
        split_array = _split_consecutive(array, stepsize)
        return split_array

    @classmethod
    def find_pulses_simple(cls, waveform_array: np.ndarray,
                           baseline_value: float, std_value: float,
                           sigma_lvl: float = 5, negative_polarity: bool = True,
                           baseline_subtracted: bool = False) -> list:
        """Pulse processing to find pulses above sigma times baseline rms.

        Args:
           waveform_array (np.ndarray): 1D array of all the ADC counts where
                each element is a sample number in the waveform. The length of the
                array is the ammount of samples in the waveform.
            baseline_value (float): value of ADC counts to use as baseline.
            std_value (float): standard deviation of the calculated baseline
                value.
            sigma_lvl (float, optional): number of times above baseline in
                stds to consider as new pulse. Defaults to 5.
            negative_polarity (bool): Polarity of the signal, True for
                negative (as standard SiPM waveforms), False for positive.
                Defaults to True.
            baseline_subtracted (bool): info if the baseline is already
                subtracted in the waveform. Defaults to False.

        Returns:
            list: list with the found pulses.
        """

        if baseline_subtracted:
            baseline_value = 0

        if negative_polarity:
            above_baseline = np.where(
                waveform_array < (
                    baseline_value -
                    std_value *
                    sigma_lvl))[0]

        elif not negative_polarity:
            above_baseline = np.where(
                waveform_array > (
                    baseline_value +
                    std_value *
                    sigma_lvl))[0]

        pulses = cls.split_consecutive(above_baseline)  # type: ignore

        return pulses


@nb.njit
def _get_area(waveform: np.ndarray, baseline_value: float,
              pulse_start: int, pulse_end: int, dt: int = 10,
              negative_polarity: bool = True,
              baseline_subtracted: bool = False) -> float:
    """Get area of a single identified pulse in a waveform. Numbafied.

    Args:
        waveform (np.ndarray): 1D array of all the ADC counts where each
            element is a sample number in the waveform. The length of the array is the
            ammount of samples in the waveform.
        baseline_value (float): value of ADC counts to use as baseline.
        peak_start (int): index of start of the pulse
        peak_end (int): index of end of the pulse
        dt (int, optional): duration of each sample in the waveform in
            nanoseconds. Defaults to 10.
        negative_polarity (bool): Polarity of the signal, True for
                negative (as standard SiPM waveforms), False for positive.
                Defaults to True.
            baseline_subtracted (bool): info if the baseline is already
                subtracted in the waveform. Defaults to False.

    Returns:
        float: return calculated integrated ADC counts.
    """

    if baseline_subtracted:
        baseline_value = 0

    if negative_polarity:
        polarity = -1
    else:
        polarity = 1

    pulse_wf = waveform[pulse_start:pulse_end]
    area_under = np.sum(baseline_value + polarity * pulse_wf) * dt
    return area_under


@nb.njit
def _get_amplitude(waveform: np.ndarray, baseline_value: float,
                   peak_start: int, peak_end: int,
                   negative_polarity: bool = True,
                   baseline_subtracted: bool = False) -> float:
    """Get area of a single identified pulse in a waveform. Numbafied.

    Args:
        waveform (np.ndarray): 1D array of all the ADC counts where each
            element is a sample number in the waveform. The length of the array is the
            ammount of samples in the waveform.
        baseline_value (float): value of ADC counts to use as baseline.
        peak_start (int): index of start of the pulse
        peak_end (int): index of end of the pulse
        negative_polarity (bool): Polarity of the signal, True for
                negative (as standard SiPM waveforms), False for positive.
                Defaults to True.
            baseline_subtracted (bool): info if the baseline is already
                subtracted in the waveform. Defaults to False.

    Returns:
        float: return amplitude of pulse in ADC counts (minimum value of
            ADC counts registered in pulse)
    """
    peak_wf = waveform[peak_start:peak_end]

    if baseline_subtracted:
        baseline_value = 0

    if len(peak_wf) > 0:
        if negative_polarity:
            amplitude = min(peak_wf)
        else:
            amplitude = max(peak_wf)
    else:
        amplitude = baseline_value
    return amplitude


@nb.njit
def _split_consecutive(array: np.ndarray, stepsize: int = 1) -> list:
    """Splits an array into several where values are consecutive
    to each other, in a numbafied verison.

    Args:
        array (np.ndarray): array of indexes recognised as pulses.
        stepsize (int, optional): minimum consecutive indexes to split into
            different pulses. Defaults to 1.

    Returns:
        np.ndarray: array with splitted pulses.
    """
    split_index = np.where(np.diff(array) != stepsize)[0] + 1
    split_array = np.split(array, split_index)

    return split_array

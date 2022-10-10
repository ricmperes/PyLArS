import numpy as np
from awkward.highlevel import Array as akarray
import numba as nb


class peak_processing():
    """All the things peaks.
    """

    @classmethod
    def get_area(cls, waveform: np.ndarray, baseline_value: float,
                 peak_start: int, peak_end: int, dt: int = 10) -> float:
        """Get area of a single identified peak in a waveform. Points to
        the numbafied function _get_area(...).
        """
        area_under = _get_area(
            waveform,
            baseline_value,
            peak_start,
            peak_end,
            dt)
        return area_under

    @classmethod
    def get_all_areas(cls, waveform: np.ndarray, peaks: list,
                      baseline_value: float) -> np.ndarray:
        """Compute the areas of all the peaks in a waveform.
        TO DO: use np.apply_along_axis or similar and see if
        there is speed improvement.
        """
        areas = np.zeros(len(peaks))
        for i, _peak in enumerate(peaks):
            areas[i] = cls.get_area(
                waveform, baseline_value, _peak[0], _peak[-1])
        return areas

    @classmethod
    def get_all_lengths(cls, peaks: list) -> list:
        """Compute the lengths of all the peaks in a waveform.
        (It's faster without @numba.njit)

        Args:
            peaks (list): list of arrays where the elements are the index
        of samples within each peak.

        Returns:
            list: list with the lenght for each peak.
        """
        lengths = [len(_peak) for _peak in peaks]
        return lengths

    @classmethod
    def get_all_positions(cls, peaks: list) -> list:
        """Calcultes the initial position of the identified peak
        in number of samples.
        (Faster without numba...?)

        Args:
            peaks (list): array of identified peaks.

        Returns:
            list: list of positions of peaks.
        """
        positions = [_peak[0] for _peak in peaks]
        return positions

    @classmethod
    def get_amplitude(cls, waveform: np.ndarray, baseline_value: float,
                      peak_start: int, peak_end: int, dt: int = 10) -> float:
        """Get area of a single identified peak in a waveform. Points to
        the numbafied function _get_area(...).
        """
        amplitude = _get_amplitude(
            waveform,
            baseline_value,
            peak_start,
            peak_end,
            dt)
        return amplitude

    @classmethod
    def get_all_amplitudes(cls, waveform: np.ndarray, peaks: list,
                           baseline_value: float) -> list:
        """Calculates the max amplitude of the identified peak
        in number of samples.
        (Faster without numba...?)

        Args:
            waveform (np.ndarray): 1D array of all the ADC counts where each
    element is a sample number in the waveform. The length of the array is the
    ammount of samples in the waveform.
            peaks (np.ndarray): array of identified peaks.
            baseline_value (float): value of ADC counts to use as baseline.

        Returns:
            list: list of amplitudes of peaks.
        """
        amplitudes = np.zeros(len(peaks))
        for i, _peak in enumerate(peaks):
            amplitudes[i] = cls.get_amplitude(
                waveform, baseline_value, _peak[0], _peak[-1])
        return amplitudes

    @classmethod
    def split_consecutive(cls, array: np.array, stepsize: int = 1):
        """Splits an array into several where values are consecutive
        to each other. Points to numbafied function _split_consecutive().

        Args:
            array (np.array): 
        """
        split_array = _split_consecutive(array, stepsize)
        return split_array

    @classmethod
    def find_peaks_simple(cls, waveform_array: np.ndarray,
                          baseline_value: float, std_value: float,
                          sigma_lvl: float = 5) -> list:
        """Pulse processing to find peaks above sigma times baseline rms.

        Args:
            waveform_array (np.ndarray): 1D array of all the ADC counts where
        each element is a sample number in the waveform. The length of the
        array is the ammount of samples in the waveform.
            baseline_value (float): value of ADC counts to use as baseline.
            std_value (float): standard deviation of the calculated baseline
        value.
            sigma_lvl (float, optional): number of times above baseline in
        stds to consider as new peak. Defaults to 5.

        Returns:
            list: list with the found peaks.
        """

        bellow_baseline = np.where(
            waveform_array < (
                baseline_value -
                std_value *
                sigma_lvl))[0]
        peaks = cls.split_consecutive(bellow_baseline)
        return peaks


@nb.njit
def _get_area(waveform: np.ndarray, baseline_value: float,
              peak_start: int, peak_end: int, dt: int = 10) -> float:
    """Get area of a single identified peak in a waveform. Numbafied.

    Args:
        waveform (np.ndarray): 1D array of all the ADC counts where each
    element is a sample number in the waveform. The length of the array is the
    ammount of samples in the waveform.
        baseline_value (float): value of ADC counts to use as baseline.
        peak_start (int): index of start of the peak
        peak_end (int): index of end of the peak
        dt (int, optional): duration of each sample in the waveform in
    nanoseconds. Defaults to 10.

    Returns:
        float: return calculated integrated ADC counts.
    """
    peak_wf = waveform[peak_start:peak_end]
    area_under = np.sum(baseline_value - peak_wf) * dt
    return area_under


@nb.njit
def _get_amplitude(waveform: np.ndarray, baseline_value: float,
                   peak_start: int, peak_end: int, dt: int = 10) -> float:
    """Get area of a single identified peak in a waveform. Numbafied.

    Args:
        waveform (np.ndarray): 1D array of all the ADC counts where each
    element is a sample number in the waveform. The length of the array is the
    ammount of samples in the waveform.
        baseline_value (float): value of ADC counts to use as baseline.
        peak_start (int): index of start of the peak
        peak_end (int): index of end of the peak
        dt (int, optional): duration of each sample in the waveform in
    nanoseconds. Defaults to 10.

    Returns:
        float: return amplitude of peak in ADC counts (minimum value of 
    ADC counts registered in peak)
    """

    peak_wf = waveform[peak_start:peak_end]
    if len(peak_wf) > 0:
        amplitude = min(peak_wf)
    else:
        amplitude = baseline_value
    return amplitude


@nb.njit
def _split_consecutive(array: np.array, stepsize: int = 1) -> np.ndarray:
    """Splits an array into several where values are consecutive
    to each other, in a numbafied verison.

    Args:
        array (np.array): array of indexes recognised as peaks.
        stepsize (int, optional): minimum consecutive indexes to split into
    different peaks. Defaults to 1.

    Returns:
        np.ndarray: array with splitted peaks.
    """
    split_index = np.where(np.diff(array) != stepsize)[0] + 1
    split_array = np.split(array, split_index)

    return split_array

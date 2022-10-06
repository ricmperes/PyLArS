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
            peaks (_type_): _description_

        Returns:
            _type_: _description_
        """
        lengths = [len(_peak) for _peak in peaks]
        return lengths

    @classmethod
    def get_all_positions(cls, peaks: list) -> list:
        """Calcultes the initial position of the identified peak
        in number of samples.
        (Faster without numba...?)

        Args:
            peaks (_type_): array of identified peaks.

        Returns:
            _type_: list of positions of peaks.
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
        """Calcultes the max amplitude of the identified peak
        in number of samples.
        (Faster without numba...?)

        Args:
            peaks (_type_): array of identified peaks.

        Returns:
            _type_: list of amplitudes of peaks.
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
        """
        split_array = _split_consecutive(array, stepsize)
        return split_array

    @classmethod
    def find_peaks_simple(cls, waveform_array: np.ndarray,
                          baseline_value: float, std_value: float,
                          sigma_lvl: float = 5):
        """Pulse processing to find peaks above sigma times baseline rms.

        Args:
            waveform_array (_type_): _description_
            baseline_value (float): _description_
            std_value (float): _description_
            sigma_lvl (float, optional): _description_. Defaults to 5.

        Returns:
            _type_: _description_
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
        waveform (_type_): _description_
        baseline_value (float): _description_
        peak_start (int): _description_
        peak_end (int): _description_
        dt (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    peak_wf = waveform[peak_start:peak_end]
    area_under = np.sum(baseline_value - peak_wf) * dt
    return area_under


@nb.njit
def _get_amplitude(waveform: np.ndarray, baseline_value: float,
                   peak_start: int, peak_end: int, dt: int = 10) -> float:
    """Get area of a single identified peak in a waveform. Numbafied.

    Args:
        waveform (_type_): _description_
        baseline_value (float): _description_
        peak_start (int): _description_
        peak_end (int): _description_
        dt (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
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
        array (np.array): _description_
        stepsize (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    split_index = np.where(np.diff(array) != stepsize)[0] + 1
    split_array = np.split(array, split_index)

    return split_array

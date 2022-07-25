import numpy as np


class peak_processing():
    """All the things peaks.
    """

    @classmethod
    def get_area(cls, waveform, baseline_value: float,
                 peak_start: int, peak_end: int, dt: int = 10):
        """Get area of a single identified peak in a waveform.

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
        area_under = np.sum(baseline_value - peak_wf) * 10
        return area_under

    @classmethod
    def get_all_areas(cls, waveform, peaks):
        """Compute the areas of all the peaks in a waveform.
        TO DO: use np.apply_along_axis or similar and see if
        there is speed improvement.

        Args:
            waveform (_type_): _description_
            peaks (_type_): _description_

        Returns:
            _type_: _description_
        """
        areas = np.zeros(len(peaks))
        for i, _peak in enumerate(peaks):
            areas[i] = cls.get_area(waveform, _peak[0], _peak[-1])
        return areas

    @classmethod
    def get_all_lengths(cls, peaks):
        """COmpute the lengths of all the peaks in a waveform.
        TO DO: try numba to speed it up.

        Args:
            peaks (_type_): _description_

        Returns:
            _type_: _description_
        """
        lengths = [len(_peak) for _peak in peaks]
        return lengths

    @classmethod
    def get_all_positions(cls, peaks):
        """Calcultes the initial position of the identified peak
        in number of samples.

        Args:
            peaks (_type_): array of identified peaks.

        Returns:
            _type_: list of positions of peaks.
        """
        positions = [_peak[0] for _peak in peaks]
        return positions

    @classmethod
    def split_consecutive(cls, array: np.array, stepsize: int = 1):
        """Splits an array into several where values are consecutive
        to each other.

        Args:
            array (np.array): _description_
            stepsize (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        return np.split(array, np.where(np.diff(array) != stepsize)[0] + 1)

    @classmethod
    def find_peaks_simple(cls, waveform_array, baseline_value: float,
                          std_value: float, sigma_lvl: float = 5):
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

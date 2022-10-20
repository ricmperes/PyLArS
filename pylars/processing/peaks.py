import numpy as np
import numba as nb


class peak_processing():
    """All the things peaks. Peaks are sums of pulses found in waveforms.
    """

    def __init__(self,sensor_gains):
        self.gain = sensor_gains

    
    def get_sum_area(self, waveforms_subtracted: np.ndarray,
                 pulse_start: int, pulse_end: int) -> float:
        """Get summed area over the all the channels.
        
        Args:
            waveforms_subtracted (np.ndarray): array (N samples * M channels)
        """

        waveforms_pe = cls.apply_gain(waveforms_subtracted)
        summed_waveform = np.sum(waveforms_subtracted, axis = 1)

        

        return area_under

    def apply_gain(self,waveforms_subtracted):
        """Transform integrated ADC_counts to pe with gain per channel.

        Args:
            waveforms_subtracted (_type_): _description_

        Returns:
            _type_: _description_
        """

    @classmethod
    def get_all_areas(cls, waveform: np.ndarray, pulses: list,
                      baseline_value: float) -> np.ndarray:
        """Compute the areas of all the pulses in a waveform.
        TO DO: use np.apply_along_axis or similar and see if
        there is speed improvement.
        """
        areas = np.zeros(len(pulses))
        for i, _pulse in enumerate(pulses):
            areas[i] = cls.get_area(
                waveform, baseline_value, _pulse[0], _pulse[-1])
        return areas

    @classmethod
    def get_all_lengths(cls, pulses: list) -> list:
        """Compute the lengths of all the pulses in a waveform.
        (It's faster without @numba.njit)

        Args:
            pulses (_type_): _description_

        Returns:
            _type_: _description_
        """
        lengths = [len(_pulse) for _pulse in pulses]
        return lengths

    @classmethod
    def get_all_positions(cls, pulses: list) -> list:
        """Calcultes the initial position of the identified pulse
        in number of samples.
        (Faster without numba...?)

        Args:
            pulses (_type_): array of identified pulses.

        Returns:
            _type_: list of positions of pulses.
        """
        positions = [_pulse[0] for _pulse in pulses]
        return positions

    @classmethod
    def split_consecutive(cls, array: np.array, stepsize: int = 1):
        """Splits an array into several where values are consecutive
        to each other. Points to numbafied function _split_consecutive().
        """
        split_array = _split_consecutive(array, stepsize)
        return split_array

    @classmethod
    def find_pulses_simple(cls, waveform_array: np.ndarray, baseline_value: float,
                          std_value: float, sigma_lvl: float = 5):
        """Pulse processing to find pulses above sigma times baseline rms.

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
        pulses = cls.split_consecutive(bellow_baseline)
        return pulses

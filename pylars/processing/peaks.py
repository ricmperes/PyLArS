from typing import List, Tuple

import numpy as np


class peak_processing():
    """All the things peaks. Peaks are sums of pulses found in waveforms.

    This class, by definition, is a collection of class methods related
    to peak processing to be used in `peakprocessor`, where a processor
    object is then constructed.
    """

    __version__ = '0.0.1'

    available_posrec_algos = ['CoG']

    @classmethod
    def apply_ADCcounts_to_e(cls, waveforms_subtracted: np.ndarray,
                             ADC_config: dict) -> np.ndarray:
        """Convert ADC counts/sample to charge.

        Applies the charge converting factor to waveforms to turn ADC counts
        (which are integrated over 1 sample) to charge. `waveforms_subtracted`
        can be one or more channels.

        Args:
            waveforms_subtracted (np.ndarray): value of ADC counts per sample
                above calculate local baseline
            ADC_config (dict): dictionary with the ADC config

        Raises:
            ValueError: If the parsed ADC_config dictionary does not have the
                required keys.

        Returns:
            np.ndarray: waveforms in charge.
        """

        try:
            ADC_range = ADC_config['ADC_range']
            ADC_impedance = ADC_config['ADC_impedance']
            F_amp = ADC_config['F_amp']
            ADC_res = ADC_config['ADC_res']
            q_e = ADC_config['q_e']
            dt = ADC_config['dt']
        except BaseException:
            raise ValueError('The ADC_config dictionary is probably missing ' +
                             'something.')

        to_e_constant = (ADC_range * dt / ADC_impedance / F_amp /
                         ADC_res / q_e)

        waveforms_charge = waveforms_subtracted * to_e_constant

        return waveforms_charge

    @classmethod
    def apply_e_to_pe(cls, waveforms_charge: np.ndarray,
                      gains: np.ndarray) -> np.ndarray:
        """Transform waveforms from charge to pe with gain per channel.

        Takes waveforms already converted to charge from ADC counts and an
        array with the gains for each channel in units of [e/pe]. The ammount
        of rows in the waveform array needs to be the same as the length of
        the gains array.

        The gains array is assumed to be on the correct order in respect to
        the order of channels in waveforms_charge.

        Args:
            waveforms_charge (np.ndarray): waveform array in charge units

        Returns:
            np.ndarray: waveform array in pe/sample
        """

        assert len(gains) == np.shape(waveforms_charge)[0], ('''Size of
        gains and channels in waveforms array do not match.''')

        waveforms_pe = (waveforms_charge.T / gains).T

        return waveforms_pe

    @classmethod
    def apply_baseline_subtract(cls, waveforms: np.ndarray,
                                baselines: np.ndarray) -> np.ndarray:
        """Apply baseline subtracting and flipping from negative to positive
        pulses.

        Args:
             waveforms (np.ndarray): waveforms, all channels stacked by rows.
            baselines (np.ndarray): computed baselines, all channels stacked
                by rows.

        Returns:
            np.ndarray: waveforms flipped and where 0 is local baseline.
        """

        assert len(baselines) == np.shape(waveforms)[0], ('''Size of
        baseines and channels in waveforms array do not match.''')

        baselines_expanded = baselines[:, :, np.newaxis]
        waveforms_subtracted = baselines_expanded - waveforms

        return waveforms_subtracted

    @classmethod
    def apply_waveforms_transform(cls, waveforms: np.ndarray,
                                  baselines: np.ndarray,
                                  gains: np.ndarray,
                                  ADC_config: dict) -> np.ndarray:
        """Converts waveforms from ADC counts/sample to pe/s.

        Takes the initials waveforms stacked for all channels and returns
        the waveforms in converted pe/s space.

        Args:
            waveforms (np.ndarray): waveforms, all channels stacked by rows.
            baselines (np.ndarray): computed baselines, all channels stacked
                by rows.
            gains (np.ndarray): gains, all channels stacked by rows.
            ADC_config (dict): dictionary with the specific digitizer configs.

        Returns:
            np.ndarray: waveforms with applied gain and e2pe transformation.
        """

        waveforms_subtracted = cls.apply_baseline_subtract(
            waveforms, baselines)
        waveforms_charge = cls.apply_ADCcounts_to_e(
            waveforms_subtracted, ADC_config)
        waveforms_pe = cls.apply_e_to_pe(waveforms_charge, gains)

        return waveforms_pe

    @classmethod
    def reorder_channel(cls, data_array: np.ndarray,
                        index_reorder: list) -> np.ndarray:
        """Reorder columns of an ndarray, corresponding to changing the order
        of channels to match the order in the sensor layout.

        Based on the following stack overflow thread: https://stackoverflow.
        com/questions/20265229/rearrange-columns-of-numpy-2d-array

        Args:
            data_array (np.ndarray): array where columns are different
                channels
            index_reorder (list): list of indexes where the change in
                `data_array` is i->index_reorder[i].

        Returns:
            np.ndarray: the original array with reordered collumns following
                `index_reorder`.
        """

        if len(np.shape(data_array)) == 1:
            data_array = np.array(data_array).reshape(1, len(data_array))

        idx = np.empty_like(index_reorder)
        idx[index_reorder] = np.arange(len(index_reorder))
        data_array[:] = data_array[:, idx]  # in-place modification of array

        return data_array

    @classmethod
    def get_sum_waveform(cls, waveforms_pe: np.ndarray) -> np.ndarray:
        """Sums the (transformed to pe/s) waveforms of all channels.

        Args:
            waveforms_pe (np.ndarray): array with waveforms from all the
                channels.

        Returns:
            np.ndarray: Summed waveform.
        """

        summed_waveform = np.sum(waveforms_pe, axis=0)

        return summed_waveform

    @classmethod
    def get_sum_peak_start_end_above_min_area(cls, areas: List[float],
                                              positions: List[int], lengths: List[int],
                                              area_min: float) -> Tuple[List[int], List[int]]:
        """Determine the indexes of start and end of a peak, considering
        only peaks with area above `area_min`.

        Returns:
            Tuple[List, List]: lists with the indexes of begin of peaks and
                end of peaks.
        """
        good_peaks_start = []
        good_peaks_end = []
        for _area, _position, _length in zip(areas, positions, lengths):
            if _area > area_min:
                good_peaks_start.append(_position)
                good_peaks_end.append(_position + _length)
        return (good_peaks_start, good_peaks_end)

    @classmethod
    def get_area_per_sensor(cls, waveforms_pe: np.ndarray,
                            peaks_start: List[int],
                            peaks_end: List[int]) -> np.ndarray:
        """Computes the area per sensor of a given waveform set to be used
        for hitpattern needs.

        Args:
            waveforms_pe (np.ndarray): waveforms in pe, one row per channel.
            peaks_start (List[int]): list with the start of peaks in the
                summed waveform.
            peaks_end (List[int]): list with the end of peaks in the
                summed waveform.

        Returns:
            np.ndarray: array with the area per channel for the peaks, each
                row a peak.
        """

        area_per_sensor = np.zeros((len(peaks_start),
                                    np.shape(waveforms_pe)[0]))
        for i, (_p_start, _p_end) in enumerate(zip(peaks_start, peaks_end)):
            area_per_sensor[i, :] = np.sum(waveforms_pe[:, _p_start:_p_end],
                                           axis=1)
        return area_per_sensor

    @classmethod
    def reconstruct_xy_position(cls, area_per_sensor: np.ndarray,
                                sensor_layout: np.ndarray,
                                algo: str = 'CoG') -> np.ndarray:
        """Computes xy position of event given a hitpattern from
        area_per_sensor.

        Args:
            algo (str, optional): The algorithm to use in position
                reconstruction. Defaults to 'CoG'.

        Returns:
            np.ndarray: array with x,y reconstructed position. Each row a
                different peak. pos[:,0] is the list of x, pos[:,1] the
                list of y.
        """

        if algo != 'CoG':
            raise NotImplementedError(f'''The requested reconstruction
            algorithm ({algo}) is not yet implemented,
            try: {cls.available_posrec_algos}''')

        area_tot = np.sum(area_per_sensor, axis=1)
        # might not be exactly the same as calculated in `process_waveform`

        x = np.sum((area_per_sensor * sensor_layout[0, :].T)) / area_tot
        y = np.sum((area_per_sensor * sensor_layout[1, :].T)) / area_tot

        return np.vstack([x, y])


class peak():
    """This is a peak (gipfel).
    """

    def __init__(self, timestamp: int,
                 wf_number: int,
                 area: float,
                 length: int,
                 position: int,
                 amplitude: float,
                 area_per_sensor: np.ndarray):

        self.timestamp = timestamp
        self.wf_number = wf_number
        self.area = area
        self.length = length
        self.position = position
        self.amplitude = amplitude
        self.area_per_sensor = area_per_sensor

import numpy as np
import pytest
from pylars.processing.pulses import pulse_processing


class Test_Pulse_Processing():

    def test_get_area(self, flat_waveform):

        area = pulse_processing.get_area(
            flat_waveform.waveform,
            flat_waveform.baseline_value,
            flat_waveform.pulse_start,
            flat_waveform.pulse_end,
            flat_waveform.dt,
            flat_waveform.negative_polarity,
            flat_waveform.baseline_subtracted)

        area_test = np.sum(np.arange(10, 20) * 10)
        assert area == area_test

    def test_get_amplitude(self, flat_waveform):

        amplitude = pulse_processing.get_amplitude(
            flat_waveform.waveform,
            flat_waveform.baseline_value,
            flat_waveform.pulse_start,
            flat_waveform.pulse_end,
            flat_waveform.negative_polarity,
            flat_waveform.baseline_subtracted)
        assert amplitude == -19.

    def test_split_consecutive(self):

        consecutive = np.array([1, 2, 3, 8, 9, 10, 15, 16, 17, 18, 19, 20])
        split = pulse_processing.split_consecutive(consecutive)
        split_test = [np.array([1, 2, 3]), np.array([8, 9, 10]),
                      np.array([15, 16, 17, 18, 19, 20])]

        for p in range(len(split)):
            assert np.unique(split[p] == split_test[p]) == True

    def test_find_pulses_simple(self):
        pass

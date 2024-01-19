import numpy as np
import pytest

from pylars.processing.waveforms import waveform_processing

class Test_Waveforms():

    def test_get_std_rough(self, flat_waveform, square_pulse):
        std_rough = waveform_processing.get_std_rough(
            flat_waveform.waveform, 10)
        assert std_rough == np.std(flat_waveform.waveform[:10])

        std_rough = waveform_processing.get_std_rough(
            square_pulse.waveform, 10)
        assert std_rough == 0.

    def test_get_baseline_rough(self, flat_waveform, square_pulse):
        baseline_rough = waveform_processing.get_baseline_rough(
            flat_waveform.waveform, 10)
        assert baseline_rough == np.median(flat_waveform.waveform[:10])

        baseline_rough = waveform_processing.get_baseline_rough(
            square_pulse.waveform, 10)
        assert baseline_rough == 0.

    def test_process_waveform(self, square_pulse):
        processed_lists = waveform_processing.process_waveform(
            waveform = square_pulse.waveform, 
            baseline_samples = 10,
            sigma_level = 3, 
            negative_polarity = square_pulse.negative_polarity)
        
        areas, lengths, positions, amplitudes = processed_lists

        assert len(areas) == len(lengths) == 1 
        assert len(positions) == len(amplitudes) == 1

        assert areas[0] == 90
        assert lengths[0] == 10
        assert positions[0] == 10
        assert amplitudes[0] == 1

        
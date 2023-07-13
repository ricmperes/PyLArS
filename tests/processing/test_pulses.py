import pytest
import numpy as np

class test_pulse_processing():
    def test_get_area(self):
        from pylars.processing.pulses import pulse_processing
        waveform = np.arange(100)
        baseline_value = 10
        pulse_start = 10
        pulse_end = 20
        dt = 10
        negative_polarity = True
        baseline_subtracted = False
        area = pulse_processing.get_area(
            waveform,
            baseline_value,
            pulse_start,
            pulse_end,
            dt,
            negative_polarity,
            baseline_subtracted)
        assert area == 1000

    def test_get_amplitude(self):
        from pylars.processing.pulses import pulse_processing
        waveform = np.arange(100)
        baseline_value = 10
        pulse_start = 10
        pulse_end = 20
        negative_polarity = True
        baseline_subtracted = False
        amplitude = pulse_processing.get_amplitude(
            waveform,
            baseline_value,
            pulse_start,
            pulse_end,
            negative_polarity,
            baseline_subtracted)
        assert amplitude == -10
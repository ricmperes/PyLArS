import numpy as np
import pytest


class custom_waveform():
    def __init__(self, waveform, baseline_value, pulse_start, pulse_end,
                 negative_polarity, baseline_subtracted, dt):

        self.waveform = waveform
        self.baseline_value = baseline_value
        self.pulse_start = pulse_start
        self.pulse_end = pulse_end
        self.negative_polarity = negative_polarity
        self.baseline_subtracted = baseline_subtracted
        self.dt = dt


@pytest.fixture
def flat_waveform():
    return custom_waveform(waveform=np.arange(50) * -1,
                           baseline_value=0,
                           pulse_start=10,
                           pulse_end=20,
                           negative_polarity=True,
                           baseline_subtracted=False,
                           dt=10)


@pytest.fixture
def square_pulse():
    wf = np.zeros(30)
    wf[10:20] = 1
    return custom_waveform(waveform=wf,
                           baseline_value=0,
                           pulse_start=10,
                           pulse_end=20,
                           negative_polarity=False,
                           baseline_subtracted=False,
                           dt=10)

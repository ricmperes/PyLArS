r'''Welcome to the core-pillar of PyLArS!

This is the module where every waveform comes to be processed and leaves with
a briefcase containing the identified pulses and their following properties:

  - area
  - length
  - position in the waveform
  - amplitude (max of ADC counts)

#### Planned TO DO:
The additions in the pipeline to this module are the following:

  - Processing of peaks, ie, the sum of pulses throughout a set of channels;

'''

from . import pulses
from . import rawprocessor
from . import waveforms
from . import peaks
from . import peakprocessor
from . import fixwindow
from . import fixwindowprocessor

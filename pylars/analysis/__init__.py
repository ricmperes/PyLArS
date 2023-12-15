r'''This module contains all the pre-established analysis of PyLArS and its
highest level data structures.

These `analysis` are mainly standard SiPM
characterization processing scripts, namely for both LED ON (`breakdown`)
and LED OFF (`darkcount`). In particular, these aim to analyse processed
data to compute and deliver for a collection of voltages and temperatures:

  - Gain
  - Breakdown voltage
  - SPE resolution
  - Dark count rate (DCR)
  - Crosstalk probability (CTP)
  - Standard plots of all the above

### Planned TO DO:
The additions in the pipeline to this module are the following:

  - Gain map structure
  - Automatic gain calculation and save/load functions for gain map

'''

from .breakdown import *
from .darkcount import *
from .ledwindow import *

r'''This module contains the one-liner plots that everyone always desires.
It is subdivided into:

  - `plotwaveforms`: waveforn plot function (with and without identified
pulses);
  - `plotprocessed`: standard plots of a processed dataset, mainly histograms;
  - `plotanalysis`: high level analysis plots (BV fits, SPE fits, DCR curves);

#### Planned TO DO:
The additions in the pipeline to this module are the following:

  - "All the channels" plots (mppc number or channel on x-axis, quantity on
the y-axis);
  - Fingerplot (specific area plot with pe charge axis).

'''

from .plotprocessed import *
from .plotwaveforms import *

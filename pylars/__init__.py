r'''
# What is PyLArS?

PyLArS is a processing and analysis framework particularly aimed at silicon
photomultiplier (SiPM) data but can be easily adapted for the study of other
photosensors. It is developed in the context of the
[Xenoscope](https://www.physik.uzh.ch/en/groups/baudis/Research/Xenoscope.html)
project at and [DARWIN](https://darwin.physik.uzh.ch/) projects at the
University of Zürich.

PyLArS main features are a focus on data structure and automated SiPM data
analysis:

  - Quick data processing using [numba](https://numba.pydata.org/) JIT compiler
at its core processing;
  - Several levels of data structure, aimed at characterization of SiPMs at
different conditions;
  - One-line analysis for breakdown voltage, gain, DCR, CTP, etc;
  - Save and load processed datasets and analysis results;
  - Batch submission of dataset processing;
  - Batch submisison of full run analysis;
  - Easy waveform watching;
  - Main analysis plots available as one-liners;

# Where to start?

  1. Install the package (requirements might not yet be well established)
  ```
  git clone git@github.com:ricmperes/PyLArS.git
  pip install -e pylars
  ```
  2. Take a look at the example
[`notebooks`](https://github.com/ricmperes/PyLArS/tree/main/notebooks) (maybe
check the development branch for any updates yet to be merged and released).
  3. Try it yourself! (for now you might need to change some hardcoded
variables from time to time. For now...)

These steps should get the package running and give you an introduction to
its basic functionalities. Do explore more of the [analysis]
(https://github.com/ricmperes/PyLArS/tree/main/pylars/analysis) available and,
if interested, reach out and contribute to the project!
'''

from . import processing
from . import plotting
from . import utils
from . import analysis

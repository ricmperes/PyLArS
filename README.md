# PyLArS

Comprehensive processing and analysis of SiPM data. Adapted for the [XenoDAQ](https://github.com/Physik-Institut-UZH/AutoXenoDAQ) data acquistion software.

[![PyPI version shields.io](https://img.shields.io/pypi/v/sipmarray.svg)](https://pypi.org/project/pylars-sipm/) [![Static Badge](https://img.shields.io/badge/Docs-%235F9EA0)](https://ricmperes.github.io/PyLArS/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13756803.svg)](https://doi.org/10.5281/zenodo.13756803)



Check the [docs](https://ricmperes.github.io/PyLArS/) and the example 
notebooks for a quick start!

## Instalation
Regular instalation through PyPi: `pip install pylars-sipm`

For a development installation:
```bash
git clone git@github.com:ricmperes/PyLArS.git
cd PyLArs 
pip install .
```
For instal in editable source:
```bash
pip install -e .
```

## How to pylars

To use pylars as "black-box" data processor go to the directory where the raw ROOT files are and run
```bash
pylars
```

For more options (raw and output files, level of RMS, polarity of signal and baseline samples) check the help funtion:
```bash
pylars --help
```


For batch processing use the scripts provided in `scripts`:
  * `make_job_files.py`, option `-r` for run number: creates a directory `jobs` and a slurm compatible `.job` file for each dataset to be submited to a cluster individually.
  * `launch_process.sh`: runs `sbatch [#.job]` for all the files in the `jobs` directory
  * `cleanup.sh`: removes all the `.job` files
 
In case batch processing is conducted in a single machine without slurm submission run:
```python
python make_job_files.py -r [run_number]
cd jobs
ls | xargs chmod +x
ls | xargs -n 1 sh
```

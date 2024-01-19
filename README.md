# PyLArS
> Ricardo Peres, Julian Haas

Comprehensive processing and analysis of SiPM data.

Check the [docs](https://ricmperes.github.io/PyLArS/) and the example 
notebooks for a quick start!

## Instalation
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

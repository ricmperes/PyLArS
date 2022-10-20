# PyLArS
> Ricardo Peres, Julian Haas, 2022

Comprehensive processing and analysis of SiPM data.

Check the [docs](https://ricmperes.github.io/PyLArS/) and the example 
notebooks for a quick start!

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

#!/bin/bash
# script to remove files created to submit a calculation to the cluster
echo Scrubbing the directory!
#jobs
rm ./*.job
rm ./jobs/*.job

#launch script
rm ./launch_process.sh

echo DONE!
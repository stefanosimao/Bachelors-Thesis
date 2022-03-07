#!/bin/bash



mkdir output_files_iters



for bs in 50 150 250;
do

sbatch --output=output_files_iters/iter$bs.out --job-name=iter$bs --error=output_files_iter/iter$bs.err script.job $bs;

done
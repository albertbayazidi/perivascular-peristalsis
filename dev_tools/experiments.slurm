#!/bin/bash
#SBATCH --partition=short
#SBATCH --account=share-ie-imf
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=graphnics_run.txt
#SBATCH --error=output.err  

apptainer exec graphnics.sif ./dev_tools/scripts/run_experiments.sh

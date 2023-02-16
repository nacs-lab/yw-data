#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --mail-type=fail
#SBATCH --output ./{jobname}.%J.out  # %J=jobid.step, %N=node.
#SBATCH --chdir=./
{requirements}

set -e  # abort whole script if any command fails

# === prepare the environement as necessary ===
module purge
module load module_squeezing
source deactivate
source activate env_tenpy
export OMP_NUM_THREADS={cores_per_task}
{environment_setup}

#if you want to redirect output to file, you can use somehting like
# command &> "{jobname}.task_{task_id}.out"

echo "Running task {task_id} of {config_file} on $HOSTNAME at $(date)" &> "{jobname}.task_{task_id}.out"
python {cluster_jobs_module} run {config_file} {task_id}
echo "finished at $(date)"

#!/bin/bash
# Usage: sbatch scripts/multinodes_run.sh

#SBATCH --job-name=wam_reasoning
#SBATCH --nodes=2
#SBATCH --qos lowest
#SBATCH --gpus-per-node=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=1536g
#SBATCH --time=24:00:00
#SBATCH --output=/checkpoint/maestro/wam_runs/slurm/reasoning/wam_reasoning-%j.log
#SBATCH --error=/checkpoint/maestro/wam_runs/slurm/reasoning/wam_reasoning-%j.err
#SBATCH --exclude=rsclearn2838,rsclearn1001,rsclearn1269,rsclearn1322,rsclearn1198,rsclearn2596,rsclearn2825,rsclearn2199

### Slurm options that might be useful
# --qos lowest
# --qos maestro
# --qos maestro_pretrain

### init virtual environment if needed
module load anaconda3/2021.05
conda init bash
source ~/.bashrc

conda activate /checkpoint/maestro/envs/bagel

SCRIPT=scripts/train/rsc/finetune.sh

### Launch to slurm
TIMESTAMP=$(date +"%y%m%d%H%M")
srun --label "${SCRIPT}" "${TIMESTAMP}" $@

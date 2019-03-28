#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --job-name=alt_entropy
#SBATCH -o ./alt_entropy.out
#SBATCH -e ./alt_entropy.err
#SBATCH --mail-type=ALL
#SBATCH --array=0

source $ENT_ENV
ENVIRONMENT="BreakoutNoFrameskip-v4"
SAVE_LOCATION_ALT="${SCRATCH}/Entropy/EntAlt/${ENVIRONMENT}/Seed${SLURM_ARRAY_TASK_ID}"
python3 main_explorer.py --env $ENVIRONMENT --eval-interval 100 --log-dir $SAVE_LOCATION_ALT --seed $SLURM_ARRAY_TASK_ID &

ENVIRONMENT="PongNoFrameskip-v4"
SAVE_LOCATION_ALT="${SCRATCH}/Entropy/EntAlt/${ENVIRONMENT}/Seed${SLURM_ARRAY_TASK_ID}"
python3 main_explorer.py --env $ENVIRONMENT --eval-interval 100 --log-dir $SAVE_LOCATION_ALT --seed $SLURM_ARRAY_TASK_ID &

ENVIRONMENT="FreewayNoFrameskip-v4"
SAVE_LOCATION_ALT="${SCRATCH}/Entropy/EntAlt/${ENVIRONMENT}/Seed${SLURM_ARRAY_TASK_ID}"
python3 main_explorer.py --env $ENVIRONMENT --eval-interval 100 --log-dir $SAVE_LOCATION_ALT --seed $SLURM_ARRAY_TASK_ID

wait    # Wait for both jobs to finish


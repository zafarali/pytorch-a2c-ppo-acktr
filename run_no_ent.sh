#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --job-name=no_entropy
#SBATCH -o ./no_entropy.out
#SBATCH -e ./no_entropy.err
#SBATCH --mail-type=ALL
#SBATCH --array=0,1,2,4,5

source $ENT_ENV
ENVIRONMENT="BreakoutNoFrameskip-v4"
SAVE_LOCATION_NO="${SCRATCH}/Entropy/EntNo/${ENVIRONMENT}/Seed${SLURM_ARRAY_TASK_ID}"
python3 main.py  --env $ENVIRONMENT --eval-interval 100 --log-dir $SAVE_LOCATION_NO --entropy-coef 0.0 --seed $SLURM_ARRAY_TASK_ID &

ENVIRONMENT="PongNoFrameskip-v4"
SAVE_LOCATION_NO="${SCRATCH}/Entropy/EntNo/${ENVIRONMENT}/Seed${SLURM_ARRAY_TASK_ID}"
python3 main.py  --env $ENVIRONMENT --eval-interval 100 --log-dir $SAVE_LOCATION_NO --entropy-coef 0.0 --seed $SLURM_ARRAY_TASK_ID & 

ENVIRONMENT="FreewayNoFrameskip-v4"
SAVE_LOCATION_NO="${SCRATCH}/Entropy/EntNo/${ENVIRONMENT}/Seed${SLURM_ARRAY_TASK_ID}"
python3 main.py  --env $ENVIRONMENT --eval-interval 100 --log-dir $SAVE_LOCATION_NO --entropy-coef 0.0 --seed $SLURM_ARRAY_TASK_ID

wait

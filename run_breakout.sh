#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --job-name=breakout
#SBATCH -o ./breakout.out
#SBATCH -e ./breakout.err
#SBATCH --mail-type=ALL
#SBATCH --array=0,2

source $ENT_ENV
ENVIRONMENT="BreakoutNoFrameskip-v4"
SAVE_LOCATION_ALT="${SCRATCH}/Entropy/EntAlt/${ENVIRONMENT}/Seed${SLURM_ARRAY_TASK_ID}"
SAVE_LOCATION_REG="${SCRATCH}/Entropy/EntReg/${ENVIRONMENT}/Seed${SLURM_ARRAY_TASK_ID}"
SAVE_LOCATION_NO="${SCRATCH}/Entropy/EntNo/${ENVIRONMENT}/Seed${SLURM_ARRAY_TASK_ID}"

echo $SAVE_LOCATION_ALT
echo $SAVE_LOCATION_REG
echo $SAVE_LOCATION_NO
python3 main_explorer.py --env $ENVIRONMENT --eval-interval 100 --log-dir $SAVE_LOCATION_ALT --seed $SLURM_ARRAY_TASK_ID &
python3 main.py  --env $ENVIRONMENT --eval-interval 100 --log-dir $SAVE_LOCATION_REG --seed $SLURM_ARRAY_TASK_ID & 
python3 main.py  --env $ENVIRONMENT --eval-interval 100 --log-dir $SAVE_LOCATION_NO --entropy-coef 0.0 --seed $SLURM_ARRAY_TASK_ID 
wait    # Wait for both jobs to finish


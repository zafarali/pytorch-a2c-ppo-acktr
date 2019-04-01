#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --job-name=alt_a0.75_d0.999995
#SBATCH -o ./alt_entropy_untuned0.75_a0.999995.out
#SBATCH -e ./alt_entropy_untuned0.75_a0.999995.err
#SBATCH --mail-type=ALL
#SBATCH --array=0,1,2

source $ENT_ENV
ENVIRONMENT="BreakoutNoFrameskip-v4"
VERSION="untuned_0.75_decay0.999995"
ALPHA0=0.75
ALPHADELTA=0.999995
SAVE_LOCATION_ALT="${SCRATCH}/Entropy/EntAlt${VERSION}/${ENVIRONMENT}/Seed${SLURM_ARRAY_TASK_ID}"
python3 main_explorer.py --env $ENVIRONMENT --eval-interval 100 --log-dir $SAVE_LOCATION_ALT --seed $SLURM_ARRAY_TASK_ID --explorer-type Decay --explorer-alpha0 $ALPHA0 --explorer-alphadelta $ALPHADELTA &

ENVIRONMENT="PongNoFrameskip-v4"
SAVE_LOCATION_ALT="${SCRATCH}/Entropy/EntAlt${VERSION}/${ENVIRONMENT}/Seed${SLURM_ARRAY_TASK_ID}"
python3 main_explorer.py --env $ENVIRONMENT --eval-interval 100 --log-dir $SAVE_LOCATION_ALT --seed $SLURM_ARRAY_TASK_ID --explorer-type Decay --explorer-alpha0 $ALPHA0 --explorer-alphadelta $ALPHADELTA &

ENVIRONMENT="FreewayNoFrameskip-v4"
SAVE_LOCATION_ALT="${SCRATCH}/Entropy/EntAlt${VERSION}/${ENVIRONMENT}/Seed${SLURM_ARRAY_TASK_ID}"
python3 main_explorer.py --env $ENVIRONMENT --eval-interval 100 --log-dir $SAVE_LOCATION_ALT --seed $SLURM_ARRAY_TASK_ID --explorer-type Decay --explorer-alpha0 $ALPHA0 --explorer-alphadelta $ALPHADELTA

wait    # Wait for both jobs to finish


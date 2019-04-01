#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --job-name=alt_entropy_untuned0.1
#SBATCH -o ./alt_entropy_untuned0.1.out
#SBATCH -e ./alt_entropy_untuned0.1.err
#SBATCH --mail-type=ALL
#SBATCH --array=0,1,2

source $ENT_ENV
ENVIRONMENT="BreakoutNoFrameskip-v4"
VERSION="untuned0.1"
ALPHA0=0.1
ALPHADELTA=1.0
SAVE_LOCATION_ALT="${SCRATCH}/Entropy/EntAlt${VERSION}/${ENVIRONMENT}/Seed${SLURM_ARRAY_TASK_ID}"
python3 main_explorer.py --env $ENVIRONMENT --eval-interval 100 --log-dir $SAVE_LOCATION_ALT --seed $SLURM_ARRAY_TASK_ID --explorer-type Decay --explorer-alpha0 $ALPHA0 --explorer-alphadelta $ALPHADELTA &

ENVIRONMENT="PongNoFrameskip-v4"
SAVE_LOCATION_ALT="${SCRATCH}/Entropy/EntAlt${VERSION}/${ENVIRONMENT}/Seed${SLURM_ARRAY_TASK_ID}"
python3 main_explorer.py --env $ENVIRONMENT --eval-interval 100 --log-dir $SAVE_LOCATION_ALT --seed $SLURM_ARRAY_TASK_ID --explorer-type Decay --explorer-alpha0 $ALPHA0 --explorer-alphadelta $ALPHADELTA &

ENVIRONMENT="FreewayNoFrameskip-v4"
SAVE_LOCATION_ALT="${SCRATCH}/Entropy/EntAlt${VERSION}/${ENVIRONMENT}/Seed${SLURM_ARRAY_TASK_ID}"
python3 main_explorer.py --env $ENVIRONMENT --eval-interval 100 --log-dir $SAVE_LOCATION_ALT --seed $SLURM_ARRAY_TASK_ID --explorer-type Decay --explorer-alpha0 $ALPHA0 --explorer-alphadelta $ALPHADELTA

wait    # Wait for both jobs to finish


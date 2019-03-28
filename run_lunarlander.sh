#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=3:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --job-name=lunarlander
#SBATCH -o ./LOGS/lunarlander.out
#SBATCH -e ./LOGS/lunarlander.err
#SBATCH --mail-type=ALL
#SBATCH --array=0,1

source $ENT_ENV
ENVIRONMENT="LunarLander-v2"
SAVE_LOCATION_ALT="${SCRATCH}/EntAlt/${ENVIRONMENT}/Seed${SLURM_ARRAY_TASK_ID}"
SAVE_LOCATION_REG="${SCRATCH}/EntReg/${ENVIRONMENT}/Seed${SLURM_ARRAY_TASK_ID}"
SAVE_LOCATION_NO="${SCRATCH}/EntNo/${ENVIRONMENT}/Seed${SLURM_ARRAY_TASK_ID}"

echo $SAVE_LOCATION
echo $SAVE_LOCATION$SEED1
srun --exclusive --cpu-bind=cores -c1 --mem=8G python3 main_explorer.py --env $ENVIRONMENT --eval-interval 100 --log-dir $SAVELOCATION_ALT &
srun --exclusive --cpu-bind=cores -c1 --mem=8G python3 main.py  --env $ENVIRONMENT --eval-interval 100 --log-dir $SAVELOCATION_ENT &
srun --exclusive --cpu-bind=cores -c1 --mem=8G python3 main.py  --env $ENVIRONMENT --eval-interval 100 --log-dir $SAVELOCATION_NO --entropy-coef 0.0 &
wait    # Wait for both jobs to finish


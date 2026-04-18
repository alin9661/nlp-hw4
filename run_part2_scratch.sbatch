#!/bin/bash
#SBATCH --job-name=hw4_part2_ec
#SBATCH --account=csci_ga_2590_001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue

##############################################################################
# HW4 Part 2 Extra Credit — Training T5 from Scratch
#
# Same task as Part 2 but with randomly initialized weights instead of
# pretrained. Requires >= 50 F1 on test set for extra credit.
#
# Outputs produced:
#   results/t5_scr_<experiment>_dev.sql   — dev SQL predictions
#   records/t5_scr_<experiment>_dev.pkl   — dev SQL records
#   results/t5_scr_<experiment>_test.sql  — EC test submission
#   records/t5_scr_<experiment>_test.pkl  — EC test submission
#
# Estimated runtime: 3–5 hours (from-scratch needs more epochs).
##############################################################################

# ---------- hyperparameters (edit these) ----------
EXPERIMENT_NAME="ec"
LEARNING_RATE=3e-4
BATCH_SIZE=16
TEST_BATCH_SIZE=16
MAX_EPOCHS=50
PATIENCE=8
WARMUP_EPOCHS=3
SCHEDULER="cosine"
WEIGHT_DECAY=0.01
NUM_BEAMS=4
RESUME=false          # set to true to resume from last checkpoint

# ---------- environment setup ----------
singularity exec --bind /scratch --nv --overlay \
  /scratch/$USER/nlp-25GB-500K-v4.ext3:rw \
  /scratch/$USER/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
  /bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:\$PATH
conda activate nlp_hw4

cd \$SCRATCH/hw4/release/part-2

echo '================================================'
echo ' Extra Credit: Training T5-small from Scratch'
echo ' Experiment: ${EXPERIMENT_NAME}'
echo '================================================'

RESUME_FLAG=\"\"
if [ \"${RESUME}\" = \"true\" ]; then
  RESUME_FLAG=\"--resume\"
fi

python3 train_t5.py \
  --experiment_name ${EXPERIMENT_NAME} \
  --learning_rate ${LEARNING_RATE} \
  --batch_size ${BATCH_SIZE} \
  --test_batch_size ${TEST_BATCH_SIZE} \
  --max_n_epochs ${MAX_EPOCHS} \
  --patience_epochs ${PATIENCE} \
  --num_warmup_epochs ${WARMUP_EPOCHS} \
  --scheduler_type ${SCHEDULER} \
  --weight_decay ${WEIGHT_DECAY} \
  --num_beams ${NUM_BEAMS} \
  \${RESUME_FLAG}

echo ''
echo '================================================'
echo ' Evaluating on dev set'
echo '================================================'

python3 evaluate.py \
  --predicted_sql results/t5_scr_${EXPERIMENT_NAME}_dev.sql \
  --predicted_records records/t5_scr_${EXPERIMENT_NAME}_dev.pkl \
  --development_sql data/dev.sql \
  --development_records records/ground_truth_dev.pkl

echo ''
echo 'Extra Credit complete.'
echo 'Submission files: results/t5_scr_${EXPERIMENT_NAME}_test.sql, records/t5_scr_${EXPERIMENT_NAME}_test.pkl'
echo 'Rename to t5_ft_experiment_ec_test.{sql,pkl} before submitting.'
"

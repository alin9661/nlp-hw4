#!/bin/bash
#SBATCH --job-name=hw4_part2
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
# HW4 Part 2 — Fine-tuning T5 for Text-to-SQL
#
# This script fine-tunes the T5-small model on the flight-booking text-to-SQL
# task, evaluates on the dev set, and generates test set submission files.
#
# Outputs produced:
#   results/t5_ft_<experiment>_dev.sql   — dev SQL predictions
#   records/t5_ft_<experiment>_dev.pkl   — dev SQL records
#   results/t5_ft_<experiment>_test.sql  — test submission (Q7)
#   records/t5_ft_<experiment>_test.pkl  — test submission (Q7)
#
# Estimated runtime: 2–4 hours depending on epochs and batch size.
##############################################################################

# ---------- hyperparameters (edit these) ----------
EXPERIMENT_NAME="experiment"
LEARNING_RATE=3e-4
BATCH_SIZE=16
TEST_BATCH_SIZE=16
MAX_EPOCHS=30
PATIENCE=5
WARMUP_EPOCHS=2
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
echo ' Part 2: Fine-tuning T5-small for Text-to-SQL'
echo ' Experiment: ${EXPERIMENT_NAME}'
echo '================================================'

RESUME_FLAG=\"\"
if [ \"${RESUME}\" = \"true\" ]; then
  RESUME_FLAG=\"--resume\"
fi

python3 train_t5.py \
  --finetune \
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
  --predicted_sql results/t5_ft_${EXPERIMENT_NAME}_dev.sql \
  --predicted_records records/t5_ft_${EXPERIMENT_NAME}_dev.pkl \
  --development_sql data/dev.sql \
  --development_records records/ground_truth_dev.pkl

echo ''
echo 'Part 2 complete.'
echo 'Submission files (test): results/t5_ft_${EXPERIMENT_NAME}_test.sql, records/t5_ft_${EXPERIMENT_NAME}_test.pkl'
"

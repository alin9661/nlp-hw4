#!/bin/bash
#SBATCH --job-name=hw4_part1
#SBATCH --account=csci_ga_2590_001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue

##############################################################################
# HW4 Part 1 — Fine-tuning BERT for Sentiment Classification
#
# This script runs all three stages of Part 1:
#   1. Train BERT on original IMDB data, evaluate on original test set  (Q1)
#   2. Evaluate trained model on transformed (synonym-replaced) test set (Q2)
#   3. Train BERT on augmented data, evaluate on both test sets          (Q3)
#
# Outputs produced:
#   out_original.txt              — Q1 submission
#   out_transformed.txt           — Q2 submission
#   out_augmented_original.txt    — Q3 submission
#   out_augmented_transformed.txt — Q3 submission
#
# Estimated runtime: ~2 hours total (40 min train + 5 min eval) x2 + transforms
##############################################################################

# ---------- hyperparameters (edit these) ----------
LEARNING_RATE=5e-5
NUM_EPOCHS=3
BATCH_SIZE=8
RESUME=false          # set to true to resume from last checkpoint

# ---------- environment setup ----------
singularity exec --bind /scratch --nv --overlay \
  /scratch/$USER/nlp-25GB-500K-v4.ext3:rw \
  /scratch/$USER/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
  /bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:\$PATH
conda activate nlp_hw4

cd \$SCRATCH/hw4/release/part-1

RESUME_FLAG=\"\"
if [ \"${RESUME}\" = \"true\" ]; then
  RESUME_FLAG=\"--resume\"
fi

echo '================================================'
echo ' Q1: Train BERT on original data + evaluate'
echo '================================================'

python3 main.py \
  --train \
  --eval \
  --learning_rate ${LEARNING_RATE} \
  --num_epochs ${NUM_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  \${RESUME_FLAG}

echo ''
echo '================================================'
echo ' Q2: Evaluate on transformed (synonym-replaced) test set'
echo '================================================'

python3 main.py \
  --eval_transformed \
  --batch_size ${BATCH_SIZE}

echo ''
echo '================================================'
echo ' Q3: Train on augmented data + evaluate both test sets'
echo '================================================'

python3 main.py \
  --train_augmented \
  --eval_transformed \
  --learning_rate ${LEARNING_RATE} \
  --num_epochs ${NUM_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  \${RESUME_FLAG}

echo ''
echo 'Evaluating augmented model on original test set...'

python3 main.py \
  --eval \
  --model_dir out_augmented \
  --batch_size ${BATCH_SIZE}

echo ''
echo 'Part 1 complete.'
echo 'Submission files: out_original.txt, out_transformed.txt, out_augmented_original.txt, out_augmented_transformed.txt'
"

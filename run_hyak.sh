#!/bin/bash
# Run the full Noise2Inverse pipeline on UW Hyak as a batch job.

#SBATCH --job-name=n2i
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --output=n2i_%j.out

# paths
REPO=/gscratch/stf/ameliemi/noise2inverse
CONDA=/mmfs1/gscratch/stf/ameliemi/miniforge3/etc/profile.d/conda.sh
LOCAL=/tmp/${USER}_n2i_${SLURM_JOB_ID}
RESULTS=$REPO/results

echo "=== node info ==="
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "local scratch dir: $LOCAL"
df -h /tmp | tail -1

# stage code + inputs onto local disk
mkdir -p "$LOCAL"
rsync -a --exclude='.git' "$REPO"/ "$LOCAL"/
cd "$LOCAL"

# activate the conda environment
source "$CONDA"
conda activate noise2inverse
export MPLBACKEND=Agg          # headless

# bundle whatever outputs exist even if fails part way
package() {
  echo "=== packaging results ==="
  mkdir -p "$RESULTS"
  tar czf "$RESULTS/n2i_${SLURM_JOB_ID}.tgz" -C "$LOCAL" \
      denoised reconstructions weights 2>/dev/null
  echo "results -> $RESULTS/n2i_${SLURM_JOB_ID}.tgz"
  rm -rf "$LOCAL"
}
trap package EXIT

# run the scripts 
# && chain stops at the first failure 
run() { echo "=== running $* ==="; time "$@"; }

run python 01_generate_projections.py && \
run python 02_reconstruct.py          && \
run python 03_train.py                && \
run python 04_evaluate.py             && \
run python 05_metrics.py

echo "=== pipeline exit code: $? ==="

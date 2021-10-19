#!/bin/bash
NUM_CORES=$(nproc --all)
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES

export WANDB_DIR=/tmp/sac_lsp_wandb_histogram_dbg
if [ ! -d $WANDB_DIR ]; then
  mkdir -p $WANDB_DIR
fi

export CUDA_VISIBLE_DEVICES=0

####################################
## Pretrain the SAC LSP Only (faster training baseline) ######
####################################
for env_id in "HopperBulletEnv-v0"; do
  for seed in 111 ; do
    # Without histogram logging
    (sleep 1s && python sac_lsp.py \
      --env-id $env_id \
      --seed $seed \
      --save-model \
      --save-videos \
      --total-steps 500000 \
      --wandb --wandb-project sac_lsp_wandb_histogram_dbg --wandb-entity dosssman \
      --logdir-prefix $WANDB_DIR
    ) >& /dev/null &

    # Logs histograms
    (sleep 1s && python sac_lsp_histogram.py \
        --env-id $env_id \
        --seed $seed \
        --save-model \
        --save-videos \
        --total-steps 500000 \
        --wandb --wandb-project sac_lsp_wandb_histogram_dbg --wandb-entity dosssman \
        --logdir-prefix $WANDB_DIR
    ) >& /dev/null &
    
    # Without histogram logging, now logging video with wandb.Video() too
    # (sleep 1s && python sac_lsp.py \
    #   --exp-name "sac_lsp_wandbvideo" \
    #   --env-id $env_id \
    #   --seed $seed \
    #   --save-model \
    #   --save-videos \
    #   --total-steps 500000 \
    #   --wandb --wandb-project sac_lsp_wandb_histogram_dbg --wandb-entity dosssman \
    #   --logdir-prefix $WANDB_DIR
    # ) >& /dev/null &

  done
done
####################################
## End Pretrain the SAC LSP Only (faster training baseline) ######
####################################

# Clean up
export CUDA_VISIBLE_DEVICES=

export WANDB_DIR=
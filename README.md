# Soft Actor Critic with Latent Space Policy: Snippet for WANDB Histogram logging debug

## Dependencies

1. Install conda env dependencies (wandb==0.10.32 is included) with `conda env create -f conda-env.yml`
2. Activate the environment with `conda activate saclsp`

## Variants of the algorihtms
1. the `sac_lsp.py` logs scalar and videos.
2. The `sac_lsp_histogram.py` additionally logs a few histograms.

## Notes.
- The logging is handled by a custom wrapper `THLogger` in `th_loggers.py`. It handles creation of various folder required for the experiment logging and logging of various training metrics to a TensorboardX SummaryWritter for convenience.
- WANDB is initiliazed to sync the Tensorboard logs around line 53 in the `th_loggers.py` file.
Maybe this is where something is broken ?
- Any metric that needs to be logged during training is passed to the corresponding method of the `THLogger` instance.
For example: `log_stats` for scalar metrics (losses), `log_histogram` for histograms data, `log_videos` for videos, etc...
There is not "explicit" logging via WANDB. Everything relies on syncing the tensorboard logger 
- The runs that are logged in the `sac_lsp_wandb_histogram_dbg` [WANDB project](https://wandb.ai/dosssman/sac_lsp_wandb_histogram_dbg?workspace=user-dosssman) can be reproduced by running the `runs.sh` script with the corresponding parameters.

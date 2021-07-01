# A wrapper around both the TensorboardX logger tool and the Wandb experiment manager logger
# also handlese creation of logging folders for convenience
import os
import json
import wandb
import numpy as np

# Pytorch's TensorboardX Support
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from datetime import datetime
from socket import gethostname

class TBXLogger(object):
    def __init__( self, logdir = None, logdir_prefix = None, exp_name = 'default',
        graph = None, args=None):

        self.logdir_prefix = args.logdir_prefix
        self.logdir = logdir
        # TODO: args is affect in this class, but the parent class also depends on it
        # This is bad.
        self.args = args

        if self.logdir_prefix is None:
            self.logdir_prefix = os.getcwd()
            self.logdir_prefix = os.path.join( self.logdir_prefix, 'logs')

            if not os.path.exists( self.logdir_prefix):
                os.makedirs( self.logdir_prefix)

        if self.logdir is None:
            strfiedDatetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
            self.logdir = exp_name

        # TODO: Dynamically generate the log name based on what was passed to the args.
        if hasattr( args, "env_id"):
            self.logdir +=  "_%s_seed_%d__%s.%s" % ( args.env_id, args.seed, strfiedDatetime, gethostname())
        else:
            # This is for support of non RL experiments
            self.logdir +=  "_seed_%d__%s.%s" % ( args.seed, strfiedDatetime, gethostname())

        self.full_exp_name = self.logdir

        self.logdir = os.path.join( self.logdir_prefix, self.logdir)

        if not os.path.exists( self.logdir):
            os.makedirs( self.logdir)

        # Enable Wandb if needed.
        # Call wandb.int beforethe tensorboard writer is created
        # https://docs.wandb.ai/guides/integrations/tensorboard#how-do-i-configure-tensorboard-when-im-using-it-with-wandb
        if args is not None and args.wandb:
            monitor_gym = True if args.save_videos else False

            wandb.init(project=args.wandb_project,
                config=vars(args), name=self.full_exp_name,
                monitor_gym=monitor_gym, sync_tensorboard=True, save_code=True)
        # Call after wandb.init()
        self.tb_writer = SummaryWriter(self.logdir)

        # Logs Hyperparameters. Kudos to Costa Huang for showing the way.
        if args is not None:
            hyparams ="|Parameter|Value|\n|-|-|\n%s" \
                % ('\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()]))

            self.tb_writer.add_text( "Hyparams", hyparams, 0 )

            # Also dumping hyparams to JSON file
            with open( os.path.join( self.logdir, "hyparams.json"), "w") as fp:
                json.dump( vars( args), fp)

        # Folder for weights saving
        if args.save_model:
            self.models_save_dir = os.path.join( self.logdir, "models")

            if not os.path.exists( self.models_save_dir):
                os.makedirs( self.models_save_dir)

        # Creates folder for videos saving
        self.videos_save_dir = os.path.join( self.logdir, "videos")

        if not os.path.exists( self.videos_save_dir):
            os.makedirs( self.videos_save_dir)

        # Folders for images and plots saving
        self.images_save_dir = os.path.join( self.logdir, "images")

        if not os.path.exists( self.images_save_dir):
            os.makedirs( self.images_save_dir)

    def get_logdir(self):
        return self.logdir

    # Just access it instead ?
    def get_models_savedir(self):
        return self.models_save_dir

    def get_videos_savedir(self):
        return self.videos_save_dir

    def get_images_savedir(self):
        return self.images_save_dir

    # Simple scalar logging
    def log_stats( self, stats_data, step, prefix=None):
        ''' Expects a dictionary with key names and values'''

        for tag, val in stats_data.items():
            fulltag = prefix + "/" + tag if prefix is not None else tag
            self.tb_writer.add_scalar(fulltag, val, global_step=step)

    # Simple histogram logging
    def log_histogram(self, name, hist_data, step, prefix=None):
        fieldname = name if prefix == None else prefix + "/" + name

        self.tb_writer.add_histogram( fieldname, hist_data, step)

    def close(self):
        if hasattr( self, '_tb_writer'):
            if self.tb_writer is not None:
                self.tb_writer.close()

    # Note: by using Wandb tensorboard sync, no need to call wandb.log()
    def log_image(self, name, image_data, step, prefix=None, nrows=None):
        fulltag = prefix + "/" + name if prefix is not None else name
        self.tb_writer.add_image(fulltag, image_data, step)
        # TODO: save to disk logic

    def log_pyplot(self, name, plt_object, step, prefix=None):
        fulltag = prefix + "/" + name if prefix is not None else name
        self.tb_writer.add_figure(fulltag, plt_object, step)
        # TODO: save to disk logic. Can be disk space expensive depending on what is logged
        # fig_savepath = os.path.join(self.get_images_savedir(), f"{name}_at_step_{step}.jpg")
        # plt_object.savefig(fig_savepath)

    def log_video(self, name, video_data, step, fps=60,prefix=None):
        fulltag = prefix + "/" + name if prefix is not None else name
        self.tb_writer.add_video(tag=fulltag, vid_tensor=video_data,
            global_step=step, fps=fps)
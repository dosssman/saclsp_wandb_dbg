# Util for parsing arguments

# A general blueprint for experiment configuration
import os
import sys
import argparse
from socket import gethostname

def get_arg_dict( name, type=None, default = None, help=None, metatype=None, choices=None):
    return { "name": name, "type": type, "default": default, "help": help, "metatype": metatype, "choices": choices}

# Generates Hyper parameters for a RL script
def generate_args( custom_args_list = None):
    parser = argparse.ArgumentParser(description='HWM Experiment Configurator')

    # Recovering executed script filename to be used as default experiment name
    # TODO: What if the scripts is run MPI Style ? Used  within a wrapper.
    script_filename = sys.argv[0] # Broke on 2021-05-05 for some reason ...
    script_filename = (script_filename.split('/')[-1]).replace( ".py", "")

    DEFAULT_ARGS_LIST = [
        # Basic
        get_arg_dict( "exp-name", str, script_filename, "Experiment name"),
        get_arg_dict( "env-id", str, "HopperBulletEnv-v0", "Environment\'s name"),
        get_arg_dict( "seed", int, 0, "Experiment seed"),

        # Basic 2: Depends on nature of algorithm
        get_arg_dict( "total-steps", int, None, 'Total steps sampled by the agent.'),
        get_arg_dict( "epochs", int, 100, "Training epochs."),
        get_arg_dict( "epoch-length", int, 1000, "How many samples in an epoch."),

        # Logging parameterization
        get_arg_dict( "logdir", str, None),
        get_arg_dict( "logdir-prefix", str, None),
        get_arg_dict( "log-name", str, None),
        get_arg_dict( "hostname", str, None),

        # WANDB Support
        get_arg_dict( "wandb", metatype="store_true"),
        get_arg_dict( "wandb-project", str, "hwm"),
        get_arg_dict( "wandb-entity", str, None),

        # DEBUG Params
        get_arg_dict( "notb", metatype="store_true",
            help="Used to disable Tensorboard logging for algorithmic debugs"),

        # Often used in RL
        get_arg_dict( "buffer-size", int, int( 1e6)),
        get_arg_dict( "batch-size", int, 64),
        get_arg_dict( "gamma", float, .99),

        # DL Framework parameterization
        get_arg_dict( "cpu", metatype="store_true"),
        get_arg_dict( "torch-deterministic", bool, True),
        get_arg_dict( "gpu-device", int, 0),

        # NN Parametrization for the lazy
        get_arg_dict( "hidden-sizes", int, [256, 256], metatype="list"),

        # Determinsitic evaluation and such
        get_arg_dict("eval", metatype="store_true"),
        get_arg_dict("eval-interval", int, 50), # After how many episode / update iter / epoch / global step etc ...de we evaluate ?
        get_arg_dict("eval-episodes", int, 1),

        # Saving data
        # Weights of the model etc...
        get_arg_dict("save-model", bool, False, metatype="store_true"), # Weight saving, enabled by default
        get_arg_dict("save-type", str, "episode", metatype="choice",
            choices=["episode", "update", "step", "epoch"],
            help="What milestone do we take as a reference to save the weights ? Default is 'episode', which is made to match with the eval-interval"),
        get_arg_dict("save-interval", int, 50), # Assumes we save every 20 episodes by default, which also coincides with evaluation interval

        # Videos of the agent
        get_arg_dict("save-videos", bool, False, metatype="store_true")
    ]

    # Check for redundancy in the hyparams and remove if already existing
    if custom_args_list is not None:
        for arg_dict in custom_args_list:
            for idx, default_arg_dict in enumerate( DEFAULT_ARGS_LIST):
                if arg_dict["name"] == default_arg_dict["name"]:
                    del DEFAULT_ARGS_LIST[idx]

    # Once we are sure there is no dupes, merge the dictionaries
    ARGS_LST = DEFAULT_ARGS_LIST + custom_args_list

    for arg_dict in ARGS_LST:
        arg_key = '--' + arg_dict["name"]
        if arg_dict["metatype"] is None:
            parser.add_argument(arg_key, type=arg_dict["type"],
                default=arg_dict["default"], help=arg_dict["help"])
        elif arg_dict["metatype"] == "store_true":
            parser.add_argument(arg_key, action='store_true',
                help=arg_dict["help"])
        elif arg_dict["metatype"] == "store_false":
            parser.add_argument(arg_key, action='store_true',
                help=arg_dict["help"])
        elif arg_dict["metatype"] == "list":
            parser.add_argument(arg_key, nargs='+', type=arg_dict["type"],
                default=arg_dict["default"])
        elif arg_dict["metatype"] == "choice":
            parser.add_argument(arg_key, nargs='?', type=arg_dict["type"],
                default=arg_dict["default"], choices=arg_dict["choices"])
        else:
            raise NotImplementedError

    args = parser.parse_args()

    # In case it was not specified. The rest is deferred to algorithmic logic
    if args.total_steps is None:
        args.total_steps = args.epochs * args.epoch_length
    
    # Adding the hostname used for the run to hyperparameters
    # To check which are faster on Wandb for example
    if args.hostname is None:
        args.hostname = gethostname()

    return args
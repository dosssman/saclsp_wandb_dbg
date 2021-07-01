# SAC LSP
# Parsing and logging dependencies
from configurator import generate_args, get_arg_dict
from th_loggers import TBXLogger as TBLogger

# Netowrk dependencies
import os
import copy
import time
import wandb
import numpy as np
import torch as th
import torch.nn.functional as F

# RL modules
from buffers import SimpleReplayBuffer
from utils import layer_init, update_target_network, video_from_img_list, fps_change
from lsp_networks import LSP, QFunction

# Env / Gym deps
import gym
import pybullet_envs

from gym.spaces import Box

start_time = time.time()
# Generating hyparams
CUSTOM_ARGS = [
    # Common parameter, and default overrides
    get_arg_dict("env-id", str, "HopperBulletEnv-v0", help="Generally the Gym Id of the used environment"),

    # LSP SAC Hyper parameterization
    get_arg_dict("hidden-sizes", int, [256,256], metatype="list"),
    get_arg_dict("n-coupling-layers", int, 2), # Number of coupling layers for the LSP
    get_arg_dict("policy-lr", float, 3e-4),
    get_arg_dict("q-lr", float, 1e-3),
    get_arg_dict("sac-batch-size", int, 256),

    ## Eval related
    get_arg_dict("eval-interval", int, 5000), # Timestep interval for evluations
    get_arg_dict("eval-episodes", int, 5), # Number of episode to evaluate upon

    ## Off-policy settings
    get_arg_dict("start-steps", int, int(5e3)),
    get_arg_dict("tau", float, .005),
    get_arg_dict("update-interval", int, 1),
    get_arg_dict("updates-per-step", int, 1),

    ## SAC Specific
    get_arg_dict("alpha", float, 0.2),
    get_arg_dict("autotune", bool, metatype="store_true"),
    get_arg_dict("target-update-interval", int, 1), # Delaying Actor updates
    
    ## Twin Delaying related
    get_arg_dict("actor-update-interval", int, 1), # Delaying Actor updates

    ## NN Parameterization
    get_arg_dict("weights-init", str, "xavier", metatype="choice",
        choices=["xavier","uniform"]),
    get_arg_dict("bias-init", str, "zeros", metatype="choice",
        choices=["zeros","uniform"]),

    ## Special SAC LSP experiment: sets the latent pased to the flow to fixed value of zeros
    get_arg_dict("zero-latent", bool, False, metatype="store_true"),

    ## Video logging hyparams
    get_arg_dict("video-source-fps", int, 60), # Default FPS when recording video from env.render(*)
    get_arg_dict("save-video-fps", int, 4), # FPS of the video to be logged into the tensorboarrd.
]
args = generate_args(CUSTOM_ARGS)

# for loggless debugs
if not args.notb:
    tblogger = TBLogger(exp_name=args.exp_name, args=args)
    print(f"# Logdir: {tblogger.logdir}")

# fix seed
np.random.seed(args.seed)
th.manual_seed(args.seed)
th.cuda.manual_seed_all(args.seed)
th.backends.cudnn.deterministic = True

# set device as gpu
device = th.device( "cuda" if th.cuda.is_available() and not args.cpu else "cpu")

# Environment setup
env = gym.make(args.env_id)
env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)

# Evaluation environment
eval_env = gym.make(args.env_id)
eval_env.seed(args.seed)
eval_env.action_space.seed(args.seed)
eval_env.observation_space.seed(args.seed)

assert isinstance(env.action_space, Box), "continuous action only"

obs_shape = env.observation_space.shape[0]
act_shape = env.action_space.shape[0]

# Instanciate buffer
buffer = SimpleReplayBuffer( obs_shape, act_shape, args.buffer_size, args.sac_batch_size)

# Helper for agent evaluation
def evaluate_agent(eval_env, policy, n_episode_eval=5, deterministic=False,
                    save_videos=True):
    video_data = None
    
    returns, lengths = [], []
    for eval_ep_idx in range(n_episode_eval):
        obs, ret, done, t = eval_env.reset(), 0., False, 0
        if save_videos and eval_ep_idx == 0:
            video_data = [eval_env.render(mode="rgb_array")]
        
        while not done:
            with th.no_grad():
                latent = None if not args.zero_latent else zero_latent_np
                action = policy.get_action(obs, latents=latent, deterministic=deterministic)
        
            obs, rew, done, _ = eval_env.step(action)
            ret += rew
            t += 1

            if save_videos and eval_ep_idx == 0:
                video_data.append(eval_env.render(mode="rgb_array"))
        
        # Episode done
        returns.append(ret), lengths.append(t)
    
    eval_stats = {
        "test_mean_return": np.mean(returns), "test_mean_length": np.mean(lengths),
        "test_max_return": np.max(returns), "test_max_length": np.max(lengths),
        "test_min_return": np.min(returns), "test_min_length": np.min(lengths)
    }

    video_data = np.array(video_data)
    return eval_stats, video_data

## SAC LSP policy and Q networks
net_params = {"obs_shape": obs_shape, "act_shape": act_shape,
    "n_coupling_layers": args.n_coupling_layers, "hidden_sizes": args.hidden_sizes,
    "layer_init": lambda m: layer_init(m, args.weights_init, args.bias_init),
    "device": device}
policy = LSP(**net_params)
qf1, qf2 = QFunction(**net_params), QFunction(**net_params)
qf1_target, qf2_target = copy.deepcopy(qf1), copy.deepcopy(qf2)

# Optimizers
p_optimizer = th.optim.Adam(policy.parameters(), lr=args.policy_lr)
q_optimizer = th.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)

# Special experiment: zeros latent: precreeate the zero tensor to be used as latent
zero_latent = th.zeros([1, act_shape], device=device)
zero_latent_np = np.zeros(act_shape) # for single action sampling

# Automatic entropy tuning
if args.autotune:
    target_entropy = - th.prod(th.Tensor(env.action_space.shape).to(device)).item()
    log_alpha = th.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    a_optimizer = th.optim.Adam([log_alpha], lr=args.q_lr)
else:
    alpha = args.alpha

# Training Loop
# Tracks a few training stats
global_agent_update_iter, global_episode = 0, 0 # SAC LSP related
# Episode init
obs, done = env.reset(), False
train_episode_return, train_episode_length, train_last_log_step = 0., 0, 0

for global_step in range(1, args.total_steps+1):
    # Take one step with the policy
    if global_step <= args.start_steps:
        action = env.action_space.sample()
    else:
        latent = None if not args.zero_latent else zero_latent_np
        action = policy.get_action(obs, latents=latent)

    next_obs, rew, done, info = env.step(action)
    buffer.add_transition( obs, action, rew, next_obs, done)

    obs = next_obs

    # Tracks mean return and length
    train_episode_return += rew
    train_episode_length += 1

    # Update the policy and Q networks
    if buffer.is_ready_for_sample(args.sac_batch_size) and global_step % args.update_interval == 0:
        for _ in range(args.updates_per_step):
            global_agent_update_iter += 1

            observation_batch, action_batch, reward_batch, \
                next_observation_batch, terminal_batch = [th.Tensor(batch).to(device) 
                    for batch in buffer.sample(args.sac_batch_size)]
            batch_size = observation_batch.shape[0]

            # Q Function update
            with th.no_grad():
                latent = None if not args.zero_latent else zero_latent.repeat(batch_size, 1)
                next_actions, next_logprobs = policy.get_actions(next_observation_batch, latents=latent)
                next_obs_qf1_target = qf1_target(next_observation_batch, next_actions).view(-1)
                next_obs_qf2_target = qf2_target(next_observation_batch, next_actions).view(-1)
                min_qf_target = th.min( next_obs_qf1_target, next_obs_qf2_target)
                min_qf_target -= alpha * next_logprobs

                q_backup = reward_batch + (1. - terminal_batch) * args.gamma * min_qf_target

            qf1_values = qf1(observation_batch, action_batch).view(-1)
            qf2_values = qf2(observation_batch, action_batch).view(-1)

            qf1_loss = F.mse_loss(qf1_values, q_backup)
            qf2_loss = F.mse_loss(qf2_values, q_backup)
            qf_loss = (qf1_loss + qf2_loss ) / 2.

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # Delaying the updates of the actor network: The rationale is to have a more accurate
            # critic function before updating the actor itself. Here, we also compensate for the
            # lack of update on the actor side that might occur. (TD3 inspired)
            if global_agent_update_iter % args.actor_update_interval == 0:
                for _ in range(args.actor_update_interval): # Compensation for the delay
                    latent = None if not args.zero_latent else zero_latent.repeat(batch_size, 1)
                    resampled_actions, resampled_logprobs = policy.get_actions( observation_batch, latents=latent)

                    qf1_pi = qf1(observation_batch, resampled_actions).view(-1)
                    qf2_pi = qf2(observation_batch, resampled_actions).view(-1)
                    min_qf_pi = th.min(qf1_pi,qf2_pi)

                    policy_loss = (alpha * resampled_logprobs - min_qf_pi).mean()

                    p_optimizer.zero_grad()
                    policy_loss.backward()
                    p_optimizer.step()

                    if args.autotune:
                        with th.no_grad():
                            latent = None if not args.zero_latent else zero_latent.repeat(batch_size, 1)
                            _, resampled_logprobs = policy.get_actions( observation_batch, latents=latent)
                        alpha_loss = ( -log_alpha * (resampled_logprobs + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()

                        alpha = log_alpha.exp().item()

            if global_agent_update_iter % args.target_update_interval == 0:
                update_target_network( qf1, qf1_target, args.tau)
                update_target_network( qf2, qf2_target, args.tau)
    # Ended Upate of policy and Q networks

    # Logging agent training stats
    if global_step % args.eval_interval == 0:
        # Evaluate the current policy network's performance
        eval_stats, eval_video_data = evaluate_agent(eval_env, policy, args.eval_episodes)
        if global_agent_update_iter > 0: # Make sure at least one update was made, otherwise no policy loss etc...
            print( "[%08d] Agent | PLoss: %.3f | QLoss: %.3f | Eval Mean Ep. Ret: %.3f | Eval Mean Ep. Len.: %.3f"
                % (global_step, policy_loss.item(), qf_loss.item(),
                eval_stats["test_mean_return"], eval_stats["test_mean_length"]))
        else: # Kind of redundant, just for the few first steps too ...
            print( "[%08d] Agent | PLoss: N/A | QLoss: N/A | Eval Mean Ep. Ret: %.3f | Eval Mean Ep. Len.: %.3f"
                % (global_step, eval_stats["test_mean_return"], eval_stats["test_mean_length"]))

        if not args.notb: # Also log to tensorboard / wandb
            tblogger.log_stats(eval_stats, global_step, "eval")
            if global_agent_update_iter > 0: # log the losses since at least one update
                with th.no_grad():
                    next_min_qf = th.min(next_obs_qf1_target, next_obs_qf2_target)
                    one_minus_done_gamma_next_min_qf = (1. - terminal_batch) * args.gamma * next_min_qf
                    next_neglogpi = next_logprobs.neg()
                    alpha_next_neglogpi = alpha * next_neglogpi
                    gamma_alpha_next_neglogpi = args.gamma * alpha_next_neglogpi
                    one_minus_done_gamma_alpha_next_neglogpi = (1. - terminal_batch) * gamma_alpha_next_neglogpi
                    rew_gamma_alpha_next_neglopi = reward_batch + gamma_alpha_next_neglogpi
                    rew_one_minus_done_gamma_alpha_next_neglopi = reward_batch + one_minus_done_gamma_alpha_next_neglogpi

                train_stats = {
                    "policy_loss": policy_loss.item(),
                    "qf_loss": qf_loss.item(),
                    # QF loss additional debugs
                    "qf_rewards_mean": reward_batch.mean(),
                    "qf_rewards_var": reward_batch.var(),
                    
                    "qf_dones_mean": terminal_batch.mean(),
                    "qf_dones_var": terminal_batch.var(),

                    "next_min_qf_mean": next_min_qf.mean().item(),
                    "next_min_qf_var": next_min_qf.var().item(),

                    "one_minus_done_gamma_next_min_qf_mean": one_minus_done_gamma_next_min_qf.mean().item(),
                    "one_minus_done_gamma_next_min_qf_var": one_minus_done_gamma_next_min_qf.var().item(),

                    "next_neglogpi_mean": next_neglogpi.mean().item(),
                    "next_neglogpi_var": next_neglogpi.var().item(),

                    "alpha_next_neglogpi_mean": alpha_next_neglogpi.mean().item(),
                    "alpha_next_neglogpi_var": alpha_next_neglogpi.var().item(),

                    "gamma_alpha_next_neglogpi_mean": gamma_alpha_next_neglogpi.mean().item(),
                    "gamma_alpha_next_neglogpi_var": gamma_alpha_next_neglogpi.var().item(),

                    "one_minus_done_gamma_alpha_next_neglogpi_mean": one_minus_done_gamma_alpha_next_neglogpi.mean().item(),
                    "one_minus_done_gamma_alpha_next_neglogpi_var": one_minus_done_gamma_alpha_next_neglogpi.var().item(),

                    "rew_gamma_alpha_next_neglopi_mean": rew_gamma_alpha_next_neglopi.mean().item(),
                    "rew_gamma_alpha_next_neglopi_var": rew_gamma_alpha_next_neglopi.var().item(),

                    "rew_one_minus_done_gamma_alpha_next_neglopi_mean": rew_one_minus_done_gamma_alpha_next_neglopi.mean().item(),
                    "rew_one_minus_done_gamma_alpha_next_neglopi_var": rew_one_minus_done_gamma_alpha_next_neglopi.var().item(),
                    
                    "one_minus_done_gamma_next_qf_neglogpi_mean": (rew_one_minus_done_gamma_alpha_next_neglopi + one_minus_done_gamma_next_min_qf).mean().item(),
                    "one_minus_done_gamma_next_qf_neglogpi_var": (rew_one_minus_done_gamma_alpha_next_neglopi + one_minus_done_gamma_next_min_qf).var().item(),

                    "q_target_mean": q_backup.mean().item(),
                    "q_target_var": q_backup.var().item(),
                    "qf_values_mean": ((qf1_values + qf2_values) / 2).mean().item(),
                    "qf_values_var": ((qf1_values + qf2_values) / 2).var().item(),
                    
                    # Policy additional debus
                    "logpi_mean": resampled_logprobs.mean().item(),
                    "logpi_var": resampled_logprobs.var().item(),
                    "alpha_logpi_mean": alpha * resampled_logprobs.mean().item(),
                    "alpha_logpi_var": alpha * resampled_logprobs.var().item(),
                    "neg_min_qf_pi_mean": min_qf_pi.neg().mean().item(),
                }
                if args.autotune:
                    train_stats["alpha_loss"] = alpha_loss.item()
                train_stats["alpha"] = alpha
                tblogger.log_stats(train_stats, global_step, "train")
                
                # # Additional debug logging
                # NOTE: As of 2021-06-22, this will result in all logged
                # field to not show up on the wandb UI.
                # hist_debug_train_stats = {
                #     "qf_rewards": reward_batch,
                #     "qf_dones": terminal_batch,

                #     "next_min_qf": next_min_qf.cpu().numpy(),
                #     "one_minus_done_gamma_next_min_qf": one_minus_done_gamma_next_min_qf.cpu().numpy(),
                #     "alpha_next_neglogpi": alpha_next_neglogpi.cpu().numpy(),
                #     "gamma_alpha_next_neglogpi": gamma_alpha_next_neglogpi.cpu().numpy(),
                #     "one_minus_done_gamma_alpha_next_neglogpi": one_minus_done_gamma_alpha_next_neglogpi.cpu().numpy(),
                #     "rew_gamma_alpha_next_neglopi": rew_gamma_alpha_next_neglopi.cpu().numpy(),
                #     "rew_one_minus_done_gamma_alpha_next_neglopi": rew_one_minus_done_gamma_alpha_next_neglopi.cpu().numpy(),

                #     # Policy realted:
                #     "logpi": resampled_logprobs.detach().cpu().numpy(),
                #     "alpha_logpi": alpha * resampled_logprobs.detach().cpu().numpy(),
                #     "neg_qf1_pi_mean": qf1_pi.neg().detach().cpu().numpy(),
                #     "neg_qf2_pi_mean": qf2_pi.neg().detach().cpu().numpy(),
                #     "neg_min_qf_pi": min_qf_pi.neg().detach().cpu().numpy()
                # }
                # for k, v in hist_debug_train_stats.items():
                #     tblogger.log_histogram(k, v, global_step, "debug")
            
            # Log the various training steps
            tblogger.log_stats({
                "global_step": global_step,
                "global_agent_update_iter": global_agent_update_iter,
                "global_episode": global_episode,
                "duration": time.time() - start_time
            }, global_step)

            if args.save_videos and len(eval_video_data) > 0:         
                # NOTE: attempto at memory leak fix when logging video to TBX
                # reduce the fps of the video
                tblogger.log_video(f"test_episode_video",
                    np.expand_dims(fps_change(eval_video_data,
                        target_fps=args.save_video_fps).transpose(0,3,1,2),axis=0),
                        global_step, fps=args.save_video_fps)
                
                # Save video to disk
                video_from_img_list(eval_video_data,
                    f"test_episode_video_at_gstep_{global_step}",
                    tblogger.get_videos_savedir())
                
            # Saving model if required
            if args.save_model:
                ## SAVing policy and HWM network
                policy_savepath = os.path.join( tblogger.get_models_savedir(),
                    "policy_at_global_step_%d.h5" % (global_step))
                th.save(policy, policy_savepath)
                
                if args.wandb: # Also uploads to WANDB. Better not abuse though.
                    wandb.save(policy_savepath, base_path=tblogger.get_logdir())
    
    if done:
        global_episode += 1

        # Logs the running train episode return and lengths
        if not args.notb and (global_step - train_last_log_step) >= args.eval_interval:
            tblogger.log_stats({
                "train_episode_return": train_episode_return,
                "train_episode_length": train_episode_length
            }, global_step, "eval")
            train_last_log_step = global_step
        
        # Reset the episode stats
        obs, done = env.reset(), False
        train_episode_return, train_episode_length = 0., 0

if not args.notb:
    tblogger.close()

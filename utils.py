import os
import cv2
import math
import numpy as np
import torch as th
import torch.nn as nn

# RL helper functions
# Initializing network weights
def layer_init(layer, init_scheme, bias_init_scheme, weight_gain=1, bias_const=0):
    if isinstance(layer, nn.Linear):
        if init_scheme == "xavier":
            th.nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        elif init_scheme == "orthogonal":
            th.nn.init.orthogonal_(layer.weight, gain=weight_gain)
        if bias_init_scheme == "zeros":
            th.nn.init.constant_(layer.bias, bias_const)

# Pytorch: updating target network functions
def update_target_network(vf, vf_target, tau):
    for target_param, param in zip(vf_target.parameters(), vf.parameters()):
        target_param.data.copy_((1. - tau) * target_param.data + tau * param.data)

# Video saving tools
default_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
def video_from_img_list(ep_img_obs, filename, savedir, fps=60, fourcc=default_fourcc):
    h,w,_ = ep_img_obs[0].shape
    video_file_path = os.path.join(savedir, f"{filename}.mp4")
    video = cv2.VideoWriter(video_file_path, fourcc, float(fps), (w, h))
    for frame in ep_img_obs:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()

    return video_file_path

# Add colored border to indicate segment
# normalized: is the color value passed between [0,1] ? Makes the necessary adjectment
def draw_color_over_frame(frame, color, normalized=True, border_thicc=10):
    frame = frame.copy()
    color_value = np.array(np.array(color) * 255., dtype=np.uint8) if not normalized else color # Set normalized to False in case the frame's channels are in the form [255,255,255] uint8 instead of scaled float.
    frame[:border_thicc, :, :] = frame[:, :border_thicc, :] = frame[-border_thicc:, :, :] = frame[:, -border_thicc:, :] = color_value # Python ...
    
    return frame

# Reduces the FPS of the video, workaround for TB logger memory leak
# when saving high-quality, 60 FPS videos.
# Namely, slice the data to match lower framerates
def fps_change(video_np_data, source_fps=60, target_fps=4):
    assert source_fps > target_fps, f"Attempting meaningless FPS change from {source_fps} to {target_fps}"
    slice_step = int(math.ceil(source_fps / target_fps))
    
    return video_np_data[::slice_step, :, :, :]

# Reduce the size of the video itself
# NOTE: Expects images as unit8 arrays
def scale_frame_video(video_np_data, scale_factor=.25):
    T, H, W, C = video_np_data.shape
    scaled_dim = [int(H * scale_factor), int(W * scale_factor)]
    scaled_video_np_data = np.zeros([T, *scaled_dim, C], dtype=np.uint8)
    for t, frame in enumerate(video_np_data):
        scaled_video_np_data[t] = cv2.resize(frame, scaled_dim[::-1],
                                             interpolation=cv2.INTER_AREA)
    return scaled_video_np_data

# Video saving tools
default_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
def video_from_img_list(ep_img_obs, filename, savedir, fps=60, fourcc=default_fourcc):
    h,w,_ = ep_img_obs[0].shape
    video_file_path = os.path.join(savedir, f"{filename}.mp4")
    video = cv2.VideoWriter(video_file_path, fourcc, float(fps), (w, h))
    for frame in ep_img_obs:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()

    return video_file_path

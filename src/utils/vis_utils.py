import os
import pickle
from pathlib import Path
from typing import List, Union

import cv2
import lmdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from matplotlib.animation import ArtistAnimation, FFMpegWriter
from numpy import ndarray
from omegaconf import DictConfig
from torch import Tensor
from tqdm import tqdm

def lmdb_key_list(episode_name: str, begin_frame: int, end_frame: int) -> List:
    """
    Returns list of keys for RGB videos

    Args:
        episode_name (str): Episode name.
        begin_frame (int): Begin frame.
        end_frame (int): End frame.

    Returns:
        List: List of keys mapping to RGB frames in lmdb environment.
    """
    return [f"{Path(episode_name.split('.')[0])}/{frame_idx + 1:07d}.jpg".encode('ascii') \
            for frame_idx in range(begin_frame, end_frame + 1)]


def get_rgb_frames(lmdb_keys: List[str], lmdb_env: lmdb.Environment) -> List:
    """
    Returns list of RGB frames

    Args:
        lmdb_keys (List[str]): List of keys mapping to RGB frames in lmdb environment.
        lmdb_env (lmdb.Environment): lmdb environment.

    Returns:
        frames (List): List of RGB frames.
    """
    frames = []
    for key in lmdb_keys:
        with lmdb_env.begin() as txn:
            frame = txn.get(key)
        frame = cv2.imdecode(
            np.frombuffer(frame, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)
    return frames

def save_video(name, start, end, output_path, rgb_lmdb_env, fps=25):
    # Assemble into matplotlib figure
    lmdb_keys = lmdb_key_list(
            episode_name=name,
            begin_frame=int(start * 25),
            end_frame=int(end * 25),
        )
    frames = get_rgb_frames(lmdb_keys, rgb_lmdb_env)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    
    # Frame display settings
    ax1.set_xlim(0, frames[0].shape[1])
    ax1.set_ylim(0, frames[0].shape[0])
    ax1.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    
    animated_frames = []
    for frame in frames:
        animated_frame = []
        animated_frame.append(ax1.imshow(np.flipud(frame), animated=True, interpolation="nearest"))
        animated_frames.append(animated_frame)
    
    fig.tight_layout()
    anim = ArtistAnimation(fig, animated_frames, 
            interval=50,
            blit=True,
            repeat=False)
    
    # Save the video locally
    writer = FFMpegWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)
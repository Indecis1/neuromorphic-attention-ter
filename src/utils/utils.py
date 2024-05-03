import functools
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def plot_animation_in_subplot(fig, ax, frames):
    def animate(ax, frame):
        ax.set_data(frame)
        return ax

    if frames.shape[1] == 2:
        rgb = np.zeros((frames.shape[0], 3, *frames.shape[2:]))
        rgb[:, 1:, ...] = frames
        frames = rgb
    if frames.shape[1] in [1, 2, 3]:
        frames = np.moveaxis(frames, 1, 3)
    ax_img_show = ax.imshow(frames[0])
    ax.axis("off")
    anim = animation.FuncAnimation(fig, functools.partial(animate, ax_img_show), frames=frames, interval=100)
    return anim


def upscale(spike_output: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """
    Take a 1D array representing the spike output of the network where each cell represent a dim-pixel binning as input
    and return a 2D-array representing the ungroup form
    :param spike_output: a 1-D array representing the pixel grouped
    :param img_width: The number of pixel in the width dimension each value in spike_output represent
    :param img_height: The number of pixel in the height dimension each value in spike_output represent
    :return: The upscale version of the image
    """
    upscale_img = []
    block = []
    for i, spike_num in enumerate(spike_output):
        block.extend([spike_num] * img_width)
        if (i + 1) % img_width == 0:
            upscale_img.extend([*block] * img_height)
            block = []
    upscale_img = np.array(upscale_img)
    upscale_img_side = int(math.sqrt(upscale_img.shape[0]))
    return np.reshape(upscale_img, (upscale_img_side, upscale_img_side))


def normalise_spike_output(data):
    data = data / data.max()
    return data
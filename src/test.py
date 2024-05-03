import functools

import matplotlib.pyplot as plt
import numpy as np
import tonic
from matplotlib import animation

dataset = tonic.datasets.DVSGesture(save_to='./', train=True)
data, label = dataset[0]

sensor_size = tonic.datasets.DVSGesture.sensor_size
frame_transform = tonic.transforms.ToFrame(
    sensor_size=sensor_size, time_window=40000,
)

frames = frame_transform(data)
ani = tonic.utils.plot_animation(frames)


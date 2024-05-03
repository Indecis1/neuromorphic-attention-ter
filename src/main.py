import math

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.proc.monitor.process import Monitor
from matplotlib import pyplot as plt
import numpy as np
import tonic

from src.process_processModel.Network import NetworkProcess
from src.process_processModel.SpikeInput import SpikeInputProcess
from src.process_processModel.SpikeOutput import SpikeOutputProcess
from src.utils import utils


# Loading input events
sensor_size = (128, 128)
dataset = tonic.datasets.DVSGesture(save_to='./', train=True)
raw_events, label = dataset[0]

events = np.zeros((len(raw_events), 4))
for e in range(len(raw_events)):
    events[e][0] = int(raw_events[e][0] * (sensor_size[0] / 128))
    events[e][1] = int(raw_events[e][1] * (sensor_size[1] / 128))
    events[e][2] = int(raw_events[e][2])
    events[e][3] = int(raw_events[e][3] / 1000)
events = events.astype(int)

events = events
events[:, 3] += 3

events = np.array(events)

model_param = {
    "shape_in": (sensor_size[0] * sensor_size[1],),
    "kernel": (4, 4),
    "stride": (4, 4),
    "padding": (0, 0),
    "sensor_size": sensor_size,

}

spike_input = SpikeInputProcess(vth=1, events=events, sensor_size=sensor_size)
model = NetworkProcess(**model_param)
output_proc = SpikeOutputProcess(shape=model.s_out.shape)

spike_input.spikes_out.connect(model.s_in)
model.s_out.connect(output_proc.spikes_in)

plot_img_width = model_param["kernel"][0]
plot_img_height = model_param["kernel"][1]
print("output_shape: ", output_proc.spikes_accum.shape)

# Monitor states of our lif neurons
monitor_model = Monitor()
num_steps = np.max(events[:, 3]) + 1
monitor_model.probe(model.lif1_v, num_steps)

# Run condition : we only need to run our network for a few timesteps
run_condition = RunSteps(num_steps=num_steps)

# Run config : we use CPU
run_cfg = Loihi1SimCfg(select_tag="floating_pt")

# Running simulation
model.run(condition=run_condition, run_cfg=run_cfg)

# Visualize stuff
data_model = monitor_model.get_data()

# Function to transform events to frame
frame_transform = tonic.transforms.ToFrame(
    sensor_size=tonic.datasets.DVSGesture.sensor_size, time_window=40000,
)
# We transform events to frames
frames = frame_transform(raw_events)

# We create a figure where we will plot the animation and the attention level of the network
fig, axes = plt.subplots(1, 3, figsize=(40, 15))
ax = axes[0]
anim = utils.plot_animation_in_subplot(fig, ax, frames)

ax = axes[1]
# data = utils.upscale(output_proc.spikes_accum.get(), plot_img_width, plot_img_height)
data = output_proc.spikes_accum.get().reshape((32, 32))

print("data_shape before upscale: ", output_proc.spikes_accum.get().shape)
print("data_shape: ", data.shape)
im = ax.imshow(data)
cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel("Focus level", rotation=-90, va="bottom")
# ax.set_xlim((0, data.shape[0]-0.5))
# ax.set_ylim((0, data.shape[1]-0.5))

ax = axes[2]
# We ungroup pixels and normalise the data
# data = utils.normalise_spike_output(utils.upscale(output_proc.spikes_accum.get(), plot_img_width, plot_img_height))
data = utils.normalise_spike_output(data)
# We show the attention level
im = ax.imshow(data)
cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel("Focus level", rotation=-90, va="bottom")
# ax.set_xlim((0, data.shape[0]-0.5))
# ax.set_ylim((0, data.shape[1]-0.5))

fig.suptitle("An event-based video on the left, the attention level of the network corresponding to the event-based "
             "video on the center and the normalise version of attention level on the right")

plt.show()

# Stop the execution
model.stop()

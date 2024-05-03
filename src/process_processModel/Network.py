import random

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
import numpy as np
from lava.magma.core.resources import CPU
from lava.proc.conv import utils
from lava.proc.conv.process import Conv
from lava.proc.dense.process import Dense
from lava.proc.lif.process import LIF

from src.process_processModel.Flatten import Flatten
from src.process_processModel.TransformProcess import TransformProcess


class NetworkProcess(AbstractProcess):
    def __init__(self, **kwargs):
        super().__init__()

        self.shape_in = kwargs.get("shape_in", (1,))
        self.shape_out = kwargs.get("shape_out", (1,))
        kernel = kwargs.get("kernel", (1,))
        stride = kwargs.get("stride", (1,))
        padding = kwargs.get("padding", (0,))
        self.sensor_size = kwargs.get("sensor_size", (16, 16))

        self.conv_param = {
            "input_shape": (self.sensor_size[0], self.sensor_size[1], 1),
            "weight": np.array([
                [
                    [[random.random()]] * kernel[0]
                ] * kernel[1]
            ]),
            "stride": np.array(stride),
            "padding": np.array(padding),
            # "dilation": (0, 0),
        }
        conv_output_shape = utils.output_shape(input_shape=self.conv_param["input_shape"], out_channels=self.conv_param["weight"].shape[0], kernel_size=kernel, stride=stride, padding=padding, dilation=(1,1))
        self.shape_out = (conv_output_shape[0] * conv_output_shape[1] * conv_output_shape[2], )
        # self.w_dense0 = np.array([[1] * int((self.sensor_size[0]/kernel[0])) * int((self.sensor_size[1]/kernel[1]))] * self.shape_out[0])
        # self.w_dense0[1, 1] = 0
        self.s_in = InPort(shape=self.shape_in)
        self.s_out = OutPort(shape=self.shape_out)

        self.lif1_u = Var(shape=self.shape_out)
        self.lif1_v = Var(shape=self.shape_out)


@implements(proc=NetworkProcess)
@requires(CPU)
class PyNetworkModel(AbstractSubProcessModel):

    def __init__(self, proc):

        self.conv_param = proc.conv_param

        self.transform = TransformProcess(proc.sensor_size)
        self.conv0 = Conv(**self.conv_param)

        self.flatten = Flatten(shape_in=self.conv0.output_shape)
        # self.dense0 = Dense(weights=proc.w_dense0)
        self.output = LIF(shape=self.flatten.s_out.shape, vth=0.5, dv=0.5, du=1.0, bias_mant=0)

        proc.s_in.connect(self.transform.s_in)
        self.transform.s_out.connect(self.conv0.s_in)
        self.conv0.a_out.connect(self.flatten.s_in)
        self.flatten.s_out.connect(self.output.a_in)
        # self.flatten.s_out.connect(self.dense0.s_in)
        # self.dense0.a_out.connect(self.output.a_in)
        self.output.s_out.connect(proc.s_out)

        proc.lif1_u.alias(self.output.u)
        proc.lif1_v.alias(self.output.v)

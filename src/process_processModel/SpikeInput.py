import numpy as np
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


# Classes to feed our events to an SNN
class SpikeInputProcess(AbstractProcess):
    """Takes an array of events and converts it to input spikes"""

    def __init__(self, vth: int, events: np.ndarray, sensor_size=(16, 16)):
        super().__init__()
        shape = (sensor_size[0] * sensor_size[1],)
        self.spikes_out = OutPort(shape=shape)  # Input spikes to the SNN
        self.v = Var(shape=shape, init=0)
        self.vth = Var(shape=(1,), init=vth)
        self.events = Var(shape=(len(events), 4), init=events)


@implements(proc=SpikeInputProcess, protocol=LoihiProtocol)
@requires(CPU)
class PySpikeInputModel(PyLoihiProcessModel):
    spikes_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    v: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    vth: int = LavaPyType(int, int, precision=32)
    events: np.ndarray = LavaPyType(np.ndarray, int, precision=32)

    def __init__(self, proc_params, sensor_size=(16, 16)):
        super().__init__(proc_params=proc_params)
        self.sensor_size = sensor_size

    def post_guard(self):
        """Guard function for PostManagement phase.
        """
        return True

    def run_post_mgmt(self):
        """Post-Management phase: executed only when guard function above
            returns True.
        """
        self.v = np.zeros(self.v.shape)

    def run_spk(self):
        """Spiking phase: executed unconditionally at every time-step
        """
        events_at_this_timestep = self.events[self.events[:, 3] == self.time_step]
        for e in events_at_this_timestep:
            self.v[e[0] * self.sensor_size[0] + e[1]] += self.vth
        s_out1 = self.v >= self.vth
        # s_out2 = np.argwhere(s_out1 == True)
        self.spikes_out.send(data=s_out1)


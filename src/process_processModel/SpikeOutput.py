import numpy as np
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


# Classes to gather the output from our SNN
class SpikeOutputProcess(AbstractProcess):
    """Process to gather spikes from output LIF neurons"""

    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs.get("shape", (1,))
        self.spikes_in = InPort(shape=shape)
        self.spikes_accum = Var(shape=shape)  # Accumulated spikes for classification


@implements(proc=SpikeOutputProcess, protocol=LoihiProtocol)
@requires(CPU)
class PySpikeOutputModel(PyLoihiProcessModel):
    spikes_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, precision=8)
    spikes_accum: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=32)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)

    def post_guard(self):
        """Guard function for PostManagement phase.
        """
        return False

    def run_post_mgmt(self):
        """Post-Management phase: executed only when guard function above
        returns True.
        """
        pass

    def run_spk(self):
        """Spiking phase: executed unconditionally at every time-step
        """
        spk_in = self.spikes_in.recv()
        self.spikes_accum = self.spikes_accum + spk_in

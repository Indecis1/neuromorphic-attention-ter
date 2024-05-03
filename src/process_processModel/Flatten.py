import numpy as np
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort, InPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


class Flatten(AbstractProcess):

    def __init__(self, shape_in):
        super().__init__()
        shape_out = 1
        for s in shape_in:
            shape_out *= s
        shape_out = (shape_out, )
        self.s_in = InPort(shape=shape_in)
        self.s_out = OutPort(shape=shape_out)


@implements(proc=Flatten, protocol=LoihiProtocol)
@requires(CPU)
class PyFlattenProcessModel(PyLoihiProcessModel):

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, precision=8)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int, precision=8)

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
        data: np.ndarray = self.s_in.recv()
        data = data.flatten()
        self.s_out.send(data=data)

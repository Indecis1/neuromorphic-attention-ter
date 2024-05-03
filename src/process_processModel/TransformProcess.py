import numpy as np
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


class TransformProcess(AbstractProcess):

    def __init__(self, sensor_size=(16, 16)):
        super().__init__(sensor_size=sensor_size)
        shape_in = (sensor_size[0] * sensor_size[1],)
        shape_out = (sensor_size[0], sensor_size[1], 1)
        self.sensor_size = sensor_size
        self.s_in = InPort(shape=shape_in)
        self.s_out = OutPort(shape=shape_out)


@implements(proc=TransformProcess, protocol=LoihiProtocol)
@requires(CPU)
class PyTransformProcessModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, precision=8)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int, precision=8)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        self.sensor_size = proc_params.get("sensor_size")

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
        data = data.reshape(self.sensor_size[0], self.sensor_size[1], 1)
        self.s_out.send(data=data)


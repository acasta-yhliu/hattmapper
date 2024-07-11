from PIL import Image
import qiskit_ibm_runtime.fake_provider
from qiskit.transpiler import CouplingMap

from architecture import Architecture

CouplingMap.draw(Architecture.coupling_map).save("arch.png")
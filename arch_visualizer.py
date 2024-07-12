from PIL import Image
import qiskit_ibm_runtime.fake_provider
from qiskit.transpiler import CouplingMap

from architecture import Architecture
#just to draw the architecture we are using to visualize hardware structure and how we map
CouplingMap.draw(Architecture.coupling_map).save("arch.png")
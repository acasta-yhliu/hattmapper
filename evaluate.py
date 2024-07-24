from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver

from utility import Simulation
from utility import Evaluation
from utility import load_molecule
from ternary_tree_mapper import HamiltonianTernaryTreeMapper
from ternary_bonsai_mapper import HamiltonianTernaryBonsaiMapper
from connectivity_conscious_mapper import HamiltonianTernaryConnectivityMapper
from original_bonsai_mapper import HamiltonianOriginalBonsaiMapper
from ternary_union_mapper import HamiltonianTernaryUnionMapper
from qiskit.synthesis import LieTrotter, QDrift

import time

from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper
# "H 0 0 0; Li 1.6 0 0;"
fermionic_hamiltonian = (
    PySCFDriver(
        atom= load_molecule("tests/methane.json"),
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    .run()
    .hamiltonian.second_q_op()
)

start = time.time()
simulation = Evaluation(fermionic_hamiltonian, HamiltonianTernaryBonsaiMapper(fermionic_hamiltonian))
end = time.time()
print(str(end - start))

start = time.time()
simulation = Evaluation(fermionic_hamiltonian, HamiltonianTernaryUnionMapper(fermionic_hamiltonian))
end = time.time()
print(str(end - start))

# start = time.time()
# simulation = Evaluation(fermionic_hamiltonian, HamiltonianTernaryConnectivityMapper(fermionic_hamiltonian))
# end = time.time()
# print(str(end - start))

# start = time.time()
# simulation = Evaluation(fermionic_hamiltonian, HamiltonianOriginalBonsaiMapper(fermionic_hamiltonian))
# end = time.time()
# print(str(end - start))
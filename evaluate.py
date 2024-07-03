from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver

from utility import Simulation
from ternary_tree_mapper import HamiltonianTernaryTreeMapper
from ternary_bonsai_mapper import HamiltonianTernaryBonsaiMapper
from qiskit.synthesis import LieTrotter, QDrift

from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper

fermionic_hamiltonian = (
    PySCFDriver(
        atom="H 0 0 0; H 0 0 0.735",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    .run()
    .hamiltonian.second_q_op()
)

# simulation = Simulation(
#     fermionic_hamiltonian, HamiltonianTernaryBonsaiMapper(fermionic_hamiltonian)
# )

simulation = Simulation(fermionic_hamiltonian, JordanWignerMapper())

print(simulation.simulate())

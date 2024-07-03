from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver

from utility import Simulation
from ternary_tree_mapper import HamiltonianTernaryTreeMapper

fermionic_hamiltonian = (
    PySCFDriver(
        atom="H 0 0 0; Li 0 0 1.6",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    .run()
    .hamiltonian.second_q_op()
)

simulation = Simulation(
    fermionic_hamiltonian, HamiltonianTernaryTreeMapper(fermionic_hamiltonian)
).build()

print(simulation.circuit_gates, simulation.circuit_depth)

print(simulation.simulate())

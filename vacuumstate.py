from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers.mode_based_mapper import ModeBasedMapper
from qiskit.quantum_info import SparsePauliOp, Statevector

from hatt_mapper import HATTMapper
from hatt_pairing_mapper import HATTPairingMapper

molecules = (
    ("H_2", "H 0 0 0; H 0 0 0.735"),
    ("LiH", "H 0 0 0; Li 0 0 1.6"),
    ("H_2O", "O 0.0 0.0 0.0; H 0.757 0.586 0.0; H -0.757 0.586 0.0"),
)


def assert_vacuum_state(hamiltonian: FermionicOp, mapper: ModeBasedMapper):
    pauli_table = mapper.pauli_table(hamiltonian.register_length)
    for m2j, m2j1 in pauli_table:
        op = SparsePauliOp([m2j, m2j1], coeffs=[0.5, 0.5j])  # type: ignore

        # prepare the vacuum state
        state = Statevector.from_label("0" * hamiltonian.register_length)
        if state.evolve(op).data.any():  # type: ignore
            return False
    return True


for _, geometry in molecules:
    print(geometry)

    hamiltonian: FermionicOp = (
        PySCFDriver(
            atom=geometry,
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        )
        .run()
        .hamiltonian.second_q_op()
    )

    assert assert_vacuum_state(
        hamiltonian, HATTMapper(hamiltonian)
    ), f"union mapper failed on {geometry}"
    assert assert_vacuum_state(
        hamiltonian, HATTPairingMapper(hamiltonian)
    ), f"bonsai mapper failed on {geometry}"

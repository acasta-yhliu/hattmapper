from utility import pauli_weight, PaulihedralDriver, FermihedralMapper
from hatt_pairing_mapper import HATTPairingMapper
from hatt_naive_mapper import TernaryTreeMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper

from qiskit_nature.second_q.hamiltonians.lattices import (
    BoundaryCondition,
    SquareLattice,
)
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel

print("Perform experiment on Pauli weight...")


t = -1.0  # the interaction parameter
v = 0.0  # the onsite potential
u = 5.0  # the interaction parameter U

geometry = [
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 3),
    (2, 5),
    (3, 4),
    (2, 7),
    (3, 5),
    (4, 4),
    (3, 6),
    (4, 5),
]

pw_file = open("tests/pauli-weight/fermihubbard.csv", "w")
print("Geometry,Modes,Tree,JW,BK,BTT,FH", file=pw_file)

circ_file = open("tests/circuit-complexity/fermihubbard.csv", "w")
print("Geometry,Modes,Tree,JW,BK,BTT,FH", file=circ_file)

for nrows, ncols in geometry:
    print(f"{nrows}x{ncols}")

    square_lattice = SquareLattice(
        rows=nrows, cols=ncols, boundary_condition=BoundaryCondition.PERIODIC
    )

    fhm = FermiHubbardModel(
        square_lattice.uniform_parameters(
            uniform_interaction=t,
            uniform_onsite_potential=v,
        ),
        onsite_interaction=u,
    )

    hamiltonian: FermionicOp = fhm.second_q_op().simplify()

    ttmapper = HATTPairingMapper(hamiltonian)

    mappers = (
        ttmapper,
        JordanWignerMapper(),
        BravyiKitaevMapper(),
        TernaryTreeMapper(),
        FermihedralMapper.fermi_hubbard(),
    )

    weights = []
    complexities = []

    for m in mappers:
        weight = (
            "--"
            if isinstance(m, FermihedralMapper)
            and not m.solved(hamiltonian.register_length)
            else str(pauli_weight(hamiltonian, m))
        )

        weights.append(weight)

        complexity = (
            "--"
            if isinstance(m, FermihedralMapper)
            and not m.solved(hamiltonian.register_length)
            else PaulihedralDriver(hamiltonian, m).summary
        )

        complexities.append(complexity)

    print(
        f"{nrows}x{ncols},{hamiltonian.register_length},{','.join(weights)}",
        file=pw_file,
    )

    print(
        f"{nrows}x{ncols},{hamiltonian.register_length},{','.join(complexities)}",
        file=circ_file,
    )

pw_file.close()
circ_file.close()
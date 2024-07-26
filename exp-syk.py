from utility import pauli_weight, PaulihedralDriver
from ternary_bonsai_mapper import HamiltonianTernaryBonsaiMapper
from ternary_tree_mapper import TernaryTreeMapper
from qiskit_nature.second_q.operators import MajoranaOp, FermionicOp
from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper
from numpy import random
from itertools import permutations
from math import factorial

pw_file = open("tests/pauli-weight/syk.csv", "w")
print("Geometry,Modes,Tree,JW,BK,BTT", file=pw_file)

circ_file = open("tests/circuit-complexity/syk.csv", "w")
print("Geometry,Modes,Tree,JW,BK,BTT", file=circ_file)

for modes in range(4, 12):
    print(f"{modes} Modes")

    def to_acop(m: int, modes: int):
        if m % 2 == 0:
            j = m // 2
            return FermionicOp({f"+_{j}": 1.0, f"-_{j}": 1.0}, num_spin_orbitals=modes)
        else:
            j = (m - 1) // 2
            return FermionicOp(
                {f"+_{j}": 1.0j, f"-_{j}": -1.0j}, num_spin_orbitals=modes
            )

    hamiltonian = FermionicOp({}, num_spin_orbitals=modes)
    for i, j, k, l in permutations(range(modes * 2), 4):
        coefficient = random.normal(0, 1) / factorial(4)
        hamiltonian += coefficient * (
            to_acop(i, modes)
            @ to_acop(j, modes)
            @ to_acop(k, modes)
            @ to_acop(l, modes)
        )
    hamiltonian = hamiltonian.simplify()

    ttmapper = HamiltonianTernaryBonsaiMapper(hamiltonian)

    mappers = (
        ttmapper,
        JordanWignerMapper(),
        BravyiKitaevMapper(),
        TernaryTreeMapper(),
    )

    weights = []
    complexities = []

    for m in mappers:
        weight = str(pauli_weight(hamiltonian, m))

        weights.append(weight)

        complexity = PaulihedralDriver(hamiltonian, m).summary

        complexities.append(complexity)

    print(
        f"{modes},{hamiltonian.register_length},{','.join(weights)}",
        file=pw_file,
    )

    print(
        f"{modes},{hamiltonian.register_length},{','.join(complexities)}",
        file=circ_file,
    )

pw_file.close()
circ_file.close()

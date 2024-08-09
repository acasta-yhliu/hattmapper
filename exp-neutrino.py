from utility import pauli_weight, PaulihedralDriver, FermihedralMapper, load_hamiltonian
from hatt_pairing_mapper import HATTPairingMapper
from hatt_naive_mapper import TernaryTreeMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper


print("Perform experiment on Pauli weight...")

pw_file = open("tests/pauli-weight/neutrino.csv", "w")
print("Geometry,Modes,Tree,JW,BK,BTT,FH", file=pw_file)

circ_file = open("tests/circuit-complexity/neutrino.csv", "w")
print("Geometry,Modes,Tree,JW,BK,BTT,FH", file=circ_file)

for nx in (3, 4, 5, 6, 7):
    for nf in (2, 3):
        filename = f"tests/neutrino/oneD_NX_{nx}_NF_{nf}.txt"
        print(f"NX = {nx}, NF = {nf}")

        hamiltonian: FermionicOp = load_hamiltonian(filename)

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
            f"{nx}x{nf} F,{hamiltonian.register_length},{','.join(weights)}",
            file=pw_file,
        )

        print(
            f"{nx}x{nf} F,{hamiltonian.register_length},{','.join(complexities)}",
            file=circ_file,
        )

pw_file.close()
circ_file.close()
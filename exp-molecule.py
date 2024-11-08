from utility import load_molecule, pauli_weight, PaulihedralDriver, FermihedralMapper
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from hatt_pairing_mapper import HATTPairingMapper
from hatt_naive_mapper import TernaryTreeMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper


molecules = (
    ("H_2", "H 0 0 0; H 0 0 0.735"),
    ("LiH (freeze)", "H 0 0 0; Li 0 0 1.6"),
    ("LiH", "H 0 0 0; Li 0 0 1.6"),
    ("H_2O", "O 0.0 0.0 0.0; H 0.757 0.586 0.0; H -0.757 0.586 0.0"),
    ("CH_4", load_molecule("tests/methane.json")),
    ("O_2", "O 0.616 0.0 0.0; O -0.616 0.0 0.0"),
    ("NaF", load_molecule("tests/NaF.json")),
    ("CO_2", load_molecule("tests/co2.json")),
)


pw_file = open("tests/pauli-weight/molecule.csv", "w")
print("Geometry,Modes,Tree,JW,BK,BTT,FH", file=pw_file)

circ_file = open("tests/circuit-complexity/molecule.csv", "w")
print("Geometry,Modes,Tree,JW,BK,BTT,FH", file=circ_file)

for casename, atom in molecules:
    print(casename)

    problem = PySCFDriver(
        atom=atom,
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    ).run()

    if casename == "LiH (freeze)":
        problem = FreezeCoreTransformer(
            freeze_core=True, remove_orbitals=[-3, 3, 2, -2]
        ).transform(problem)

    hamiltonian: FermionicOp = problem.hamiltonian.second_q_op()

    ttmapper = HATTPairingMapper(hamiltonian)

    mappers = (
        ttmapper,
        JordanWignerMapper(),
        BravyiKitaevMapper(),
        TernaryTreeMapper(),
        FermihedralMapper.molecule(),
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
        f"{casename},{hamiltonian.register_length},{','.join(weights)}",
        file=pw_file,
    )

    print(
        f"{casename},{hamiltonian.register_length},{','.join(complexities)}",
        file=circ_file,
    )

pw_file.close()
circ_file.close()

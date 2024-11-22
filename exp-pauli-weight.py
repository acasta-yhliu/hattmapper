from hatt_mapper import HATTMapper
from hatt_naive_mapper import HATTNaiveMapper
from termcolor import colored
from utility import load_molecule, pauli_weight, load_hamiltonian

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.hamiltonians.lattices import (
    BoundaryCondition,
    SquareLattice,
)
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel

molecules = (
    ("H_2", "H 0 0 0; H 0 0 0.735"),
    ("LiH (freeze)", "H 0 0 0; Li 0 0 1.6"),
    ("LiH", "H 0 0 0; Li 0 0 1.6"),
    ("H_2O", "O 0.0 0.0 0.0; H 0.757 0.586 0.0; H -0.757 0.586 0.0"),
    ("CH_4", load_molecule("tests/CH_4.json")),
    ("O_2", "O 0.616 0.0 0.0; O -0.616 0.0 0.0"),
    ("NaF", load_molecule("tests/NaF.json")),
    ("CO_2", load_molecule("tests/co2.json")),
)

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

record_file = open("tests/pauli-weight-compare.csv", "w")


def testcase(casename: str, hamiltonian: FermionicOp):
    # too large, we won't test on it
    if hamiltonian.register_length > 24:
        print(f"  {colored('Skip', attrs=['bold'])} {casename}")
        return

    unopt_mapper = HATTNaiveMapper(hamiltonian)
    opt_mapper = HATTMapper(hamiltonian)

    print(
        f"{casename},{pauli_weight(hamiltonian, unopt_mapper)},{pauli_weight(hamiltonian, opt_mapper)}",
        file=record_file,
    )

    print(f"  {colored('Done', attrs=['bold'])} {casename}")


def execute_molecules():
    print(f"{colored('Molecule', attrs=['bold'])} test cases")
    for casename, atom in molecules:
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

        hamiltonian: FermionicOp = problem.hamiltonian.second_q_op()  # type: ignore
        testcase(casename, hamiltonian)


def execute_fermi_hubbard():
    print(f"{colored('Fermi-Hubbard', attrs=['bold'])} test cases")
    for nrows, ncols in geometry:
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
        testcase(f"{nrows}\\times{ncols}", hamiltonian)


def execute_neutrino():
    print(f"{colored('Neutrino', attrs=['bold'])} test cases")
    for nx in (3, 4, 5, 6, 7):
        for nf in (2, 3):
            filename = f"tests/neutrino/oneD_NX_{nx}_NF_{nf}.txt"
            print(f"NX = {nx}, NF = {nf}")

            hamiltonian: FermionicOp = load_hamiltonian(filename)
            testcase(f"{nx}\\times{nf}F", hamiltonian)


if __name__ == "__main__":
    print("Case,Unopt,Opt", file=record_file)
    execute_molecules()
    execute_fermi_hubbard()
    execute_neutrino()
    record_file.close()

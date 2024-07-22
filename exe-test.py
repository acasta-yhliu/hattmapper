from typing import Literal
from utility import load_molecule, pauli_weight, Simulation
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from ternary_bonsai_mapper import HamiltonianTernaryBonsaiMapper
from ternary_tree_mapper import TernaryTreeMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("classic")

from tqdm import tqdm
import sys


hamiltonian: FermionicOp = (
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

ttmapper = HamiltonianTernaryBonsaiMapper(hamiltonian)

ttsim = Simulation(hamiltonian, ttmapper)
jwsim = Simulation(hamiltonian, JordanWignerMapper())
bksim = Simulation(hamiltonian, BravyiKitaevMapper())
bttsim = Simulation(hamiltonian, TernaryTreeMapper())

print(jwsim.simulate(depolarerr_1q=0.0004, depolarerr_2q=0.0435))
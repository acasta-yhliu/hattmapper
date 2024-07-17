from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from statistics import mean, variance
import json

BK = {
    "0000": 245,
    "0001": 143,
    "0010": 798,
    "0011": 232,
    "0100": 266,
    "0101": 120,
    "0110": 155,
    "0111": 457,
    "1000": 121,
    "1001": 52,
    "1010": 109,
    "1011": 87,
    "1100": 44,
    "1101": 45,
    "1110": 59,
    "1111": 67,
}


def read(name: str):
    with open(name, "r") as f:
        return json.load(f)["measurementProbabilities"]


def process(data: dict[str, int], mapper: FermionicMapper):
    problem = PySCFDriver("H 0 0 0; H 0 0 0.735").run()

    hamiltonian = mapper.map(problem.hamiltonian.second_q_op())

    energies = []

    for state, n in data.items():
        initial_state = Statevector.from_label(state)
        exp_energy = initial_state.expectation_value(hamiltonian)  # type: ignore

        for _ in range(n):
            energies.append(exp_energy.real)

    return (
        mean(energies),
        variance(energies),
    )


# print(read("tests/result-jw.json"))
# print(process(read("tests/result-jw.json"), JordanWignerMapper()))
print(process(BK, BravyiKitaevMapper()))

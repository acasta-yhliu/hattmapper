from typing import Literal
import json

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, EvolutionSynthesis
from qiskit.quantum_info import SparsePauliOp

from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper

from qiskit_aer import AerSimulator, noise



class Simulation:
    def __init__(
        self, fermionic_hamiltonian: FermionicOp, mapper: FermionicMapper
    ) -> None:
        self.fermionic_hamiltonian = fermionic_hamiltonian
        self.fermionic_mappper = mapper

        # this is also the observable
        self.qubit_hamiltonian: SparsePauliOp = mapper.map(fermionic_hamiltonian)  # type: ignore

    def build(
        self,
        time: float = 1.0,
        basis_gates: list[str] = ["cx", "rx", "ry", "rz"],
        synthesis: EvolutionSynthesis = LieTrotter(),
    ):
        weight = 0
        for pauli in self.qubit_hamiltonian.paulis:  # type: ignore
            label = pauli.to_label()  # type: ignore
            weight += len(label) - label.count("I")

        self.pauli_weight: int = weight

        pe = PauliEvolutionGate(self.qubit_hamiltonian, time=time, synthesis=synthesis)
        circ = QuantumCircuit(pe.num_qubits)
        circ.append(pe, range(pe.num_qubits))
        circ.measure_all()

        self.circuit = transpile(circ, basis_gates=basis_gates, optimization_level=3)
        self.basis_gates = basis_gates

        return self

    @property
    def circuit_gates(self) -> dict[str, int]:
        return {str(k): v for k, v in self.circuit.count_ops().items()}

    @property
    def circuit_depth(self) -> int:
        return self.circuit.depth()

    def simulate(
        self,
        shots: int = 1000,
        *,
        depolarerr_1q=0.0001,
        depolarerr_2q=0.001,
        device: Literal["CPU", "GPU"] = "GPU"
    ):
        # create the Aer backend with noise
        error_1 = noise.depolarizing_error(depolarerr_1q, 1)
        error_2 = noise.depolarizing_error(depolarerr_2q, 2)

        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_1, ["u3"])
        noise_model.add_all_qubit_quantum_error(error_2, ["cx"])

        backend = AerSimulator(
            method="statevector", noise_model=noise_model, device=device
        )

        result = backend.run([self.circuit], shots=shots).result()
        print(result)

        return None


def load_molecule(file: str):
    f = open(file)
    molecule = json.load(f)
    represent = ""
    try:
        molecule["PC_Compounds"][0]["coords"][0]["conformers"][0]["z"]
    except:
        for i in range(len(molecule["PC_Compounds"][0]["atoms"]["element"])):
            represent = (
                represent
                + str(molecule["PC_Compounds"][0]["atoms"]["element"][i])
                + " "
                + str(molecule["PC_Compounds"][0]["coords"][0]["conformers"][0]["x"][i])
                + " "
                + str(molecule["PC_Compounds"][0]["coords"][0]["conformers"][0]["y"][i])
                + " "
                + "0"
                + "; "
            )
    else:
        for i in range(len(molecule["PC_Compounds"][0]["atoms"]["element"])):
            represent = (
                represent
                + str(molecule["PC_Compounds"][0]["atoms"]["element"][i])
                + " "
                + str(molecule["PC_Compounds"][0]["coords"][0]["conformers"][0]["x"][i])
                + " "
                + str(molecule["PC_Compounds"][0]["coords"][0]["conformers"][0]["y"][i])
                + " "
                + str(molecule["PC_Compounds"][0]["coords"][0]["conformers"][0]["z"][i])
                + "; "
            )
    return represent

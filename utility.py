from typing import Literal
import json

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, EvolutionSynthesis
from qiskit.quantum_info import SparsePauliOp, Statevector

from qiskit_nature.second_q.operators import FermionicOp, SparseLabelOp
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.problems import BaseProblem
from qiskit_nature.second_q.hamiltonians import Hamiltonian
from qiskit_algorithms import NumPyMinimumEigensolver

from qiskit_aer import noise
from qiskit_aer.primitives import Estimator
from qiskit_aer.library import set_statevector

from architecture import Architecture

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _1q_gates(gate_name: str):
    single_qubit_gates = {
        "id",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "rx",
        "ry",
        "rz",
        "u1",
        "u2",
        "u3",
    }
    return gate_name in single_qubit_gates


def _2q_gates(gate_name: str):
    two_qubit_gates = {
        "cx",
        "cy",
        "cz",
        "ch",
        "swap",
        "crx",
        "cry",
        "crz",
        "cu1",
        "cu3",
    }
    return gate_name in two_qubit_gates


class FermionicHamiltonian(Hamiltonian):
    def __init__(self, fermionic_op: FermionicOp) -> None:
        super().__init__()
        self.fermionic_op = fermionic_op

    def second_q_op(self) -> SparseLabelOp:
        return self.fermionic_op

    @property
    def register_length(self) -> int | None:
        return self.fermionic_op.register_length


class Simulation:
    def __init__(
        self,
        fermionic_hamiltonian: FermionicOp,
        mapper: FermionicMapper,
        time: float = 1.0,
        basis_gates: list[str] = ["cx", "u3", "set_statevector"],
        synthesis: EvolutionSynthesis = LieTrotter(),
    ) -> None:
        print("Hamiltonian simulation:")

        self.basis_gates = basis_gates
        self.fermionic_hamiltonian = fermionic_hamiltonian
        self.fermionic_mappper = mapper

        # this is also the observable
        self.qubit_hamiltonian: SparsePauliOp = mapper.map(fermionic_hamiltonian)  # type: ignore

        print("  Solve ground state:")

        result = GroundStateEigensolver(mapper, NumPyMinimumEigensolver()).solve(
            BaseProblem(FermionicHamiltonian(self.fermionic_hamiltonian))
        )

        assert result.groundstate is not None
        preparation = result.groundstate[0]

        initial_state = Statevector.from_instruction(preparation)

        print(f"    Ground energy = {result.groundenergy}")
        # print(f"    Ground state  = {initial_state}")

        print("  Construct and transpile Pauli evolution:")
        print(f"    Time duration    = {time}")
        print(f"    Basis gates      = {', '.join(basis_gates)}")
        print(f"    Synthesis method = {synthesis.__class__.__name__}")

        pe = PauliEvolutionGate(self.qubit_hamiltonian, time=time, synthesis=synthesis)
        self.circuit = QuantumCircuit(Architecture.nqubits)
        self.circuit.set_statevector(initial_state)  # type: ignore
        self.circuit.append(pe, range(Architecture.nqubits))
        self.circuit = transpile(
            self.circuit, basis_gates=basis_gates, optimization_level=3, coupling_map=Architecture.coupling_map
        )

        print("  Circuit summary:")
        print(f"    Depth = {self.circuit_depth}")
        counted_gates = [
            f"{counts} {op.upper()}" for op, counts in self.circuit_gates.items()
        ]
        print(f"    Gates = {', '.join(counted_gates)}")

    @property
    def pauli_weight(self) -> int:
        weight = 0
        for pauli in self.qubit_hamiltonian.paulis:  # type: ignore
            label = pauli.to_label()  # type: ignore
            weight += len(label) - label.count("I")

        return weight

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
        depolarerr_1q=0.001,
        depolarerr_2q=0.0001,
        device: Literal["CPU", "GPU"] = "GPU",
    ):
        # create the Aer backend with noise
        error_1 = noise.depolarizing_error(depolarerr_1q, 1)
        error_2 = noise.depolarizing_error(depolarerr_2q, 2)

        noise_model = noise.NoiseModel()

        instructions = [str(ins) for ins, _ in self.circuit.count_ops().items()]

        gate_1q = [ins for ins in instructions if _1q_gates(ins)]
        gate_2q = [ins for ins in instructions if _2q_gates(ins)]

        print("  Noise model, gates:")
        print(f"    Single qubit ({depolarerr_1q:>5.0e}) = {', '.join(gate_1q)}")
        print(f"    Double qubit ({depolarerr_2q:>5.0e}) = {', '.join(gate_2q)}")

        noise_model.add_all_qubit_quantum_error(error_1, gate_1q)
        noise_model.add_all_qubit_quantum_error(error_2, gate_2q)

        print(f"  Begin simulation, {shots} shots")

        result = (
            Estimator(
                backend_options={"noise_model": noise_model, "device": device},
                skip_transpilation=True,
            )
            .run(self.circuit, self.qubit_hamiltonian, shots=shots)
            .result()
        )

        return result.values[0], result.metadata[0]["variance"]


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


class Evaluation:
    def __init__(
        self,
        fermionic_hamiltonian: FermionicOp,
        mapper: FermionicMapper,
        time: float = 1.0,
        basis_gates: list[str] = ["cx", "u3", "set_statevector"],
        synthesis: EvolutionSynthesis = LieTrotter(),
    ) -> None:
        print("Hamiltonian simulation:")

        self.basis_gates = basis_gates
        self.fermionic_hamiltonian = fermionic_hamiltonian
        self.fermionic_mappper = mapper

        # this is also the observable
        self.qubit_hamiltonian: SparsePauliOp = mapper.map(fermionic_hamiltonian)  # type: ignore

        print("  Solve ground state:")

        result = GroundStateEigensolver(mapper, NumPyMinimumEigensolver()).solve(
            BaseProblem(FermionicHamiltonian(self.fermionic_hamiltonian))
        )

        assert result.groundstate is not None
        preparation = result.groundstate[0]

        initial_state = Statevector.from_instruction(preparation)

        print(f"    Ground energy = {result.groundenergy}")

        print("  Construct and transpile Pauli evolution:")
        print(f"    Time duration    = {time}")
        print(f"    Basis gates      = {', '.join(basis_gates)}")
        print(f"    Synthesis method = {synthesis.__class__.__name__}")

        pe = PauliEvolutionGate(self.qubit_hamiltonian, time=time, synthesis=synthesis)
        self.circuit = QuantumCircuit(pe.num_qubits)
        self.circuit.append(pe, range(pe.num_qubits))
        self.circuit = transpile(
            self.circuit, basis_gates=basis_gates, optimization_level=3, #coupling_map=FakeBrooklynV2().coupling_map
        )

        print("  Circuit summary:")
        print(f"    Depth = {self.circuit_depth}")
        counted_gates = [
            f"{counts} {op.upper()}" for op, counts in self.circuit_gates.items()
        ]
        print(f"    Gates = {', '.join(counted_gates)}")


    @property
    def circuit_gates(self) -> dict[str, int]:
        return {str(k): v for k, v in self.circuit.count_ops().items()}

    @property
    def circuit_depth(self) -> int:
        return self.circuit.depth()
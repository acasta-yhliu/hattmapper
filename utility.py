from typing import Literal, cast
import json

from qiskit import QuantumCircuit, transpile, qpy
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, EvolutionSynthesis
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info import Pauli, SparsePauliOp

from qiskit_nature.second_q.operators import FermionicOp, SparseLabelOp
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.mappers.mode_based_mapper import ModeBasedMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.problems import BaseProblem
from qiskit_nature.second_q.hamiltonians import Hamiltonian
from qiskit_algorithms import NumPyMinimumEigensolver

from qiskit_aer import noise
from qiskit_aer.primitives import Estimator
from qiskit_aer.library import set_statevector
import numpy as np

import tempfile
import subprocess

from io import BytesIO

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


def pauli_weight(hamiltonian: FermionicOp, mapper: FermionicMapper):
    op = mapper.map(hamiltonian)
    weight = 0
    for pauli in op.paulis:  # type: ignore
        label = pauli.to_label()  # type: ignore
        weight += len(label) - label.count("I")
    return weight


class Simulation:
    def __init__(
        self,
        fermionic_hamiltonian: FermionicOp,
        mapper: FermionicMapper,
        time: float = 1.0,
        basis_gates: list[str] = ["cx", "u3", "set_statevector"],
        synthesis: EvolutionSynthesis = LieTrotter(),
        groundstate: bool = True,
    ) -> None:
        print("Hamiltonian simulation:")

        self.basis_gates = basis_gates
        self.fermionic_hamiltonian = fermionic_hamiltonian
        self.fermionic_mappper = mapper

        # this is also the observable
        self.qubit_hamiltonian: SparsePauliOp = mapper.map(fermionic_hamiltonian)  # type: ignore

        self._ground_energy = None

        if groundstate:
            print("  Solve ground state:")

            result = GroundStateEigensolver(mapper, NumPyMinimumEigensolver()).solve(
                BaseProblem(FermionicHamiltonian(self.fermionic_hamiltonian))
            )

            assert result.groundstate is not None
            preparation = result.groundstate[0]

            initial_state = Statevector.from_instruction(preparation)

            print(f"    Ground energy = {result.groundenergy}")

            self._ground_energy = result.groundenergy

        print("  Construct and transpile Pauli evolution:")
        print(f"    Time duration    = {time}")
        print(f"    Basis gates      = {', '.join(basis_gates)}")
        print(f"    Synthesis method = {synthesis.__class__.__name__}")

        pe = PauliEvolutionGate(self.qubit_hamiltonian, time=time, synthesis=synthesis)
        self.circuit = QuantumCircuit(pe.num_qubits)
        if groundstate:
            self.circuit.set_statevector(initial_state)  # type: ignore
        self.circuit.append(pe, range(pe.num_qubits))
        self.circuit = transpile(
            self.circuit, basis_gates=basis_gates, optimization_level=3
        )

        print("  Circuit summary:")
        print(f"    Depth = {self.circuit_depth}")
        counted_gates = [
            f"{counts} {op.upper()}" for op, counts in self.circuit_gates.items()
        ]
        print(f"    Gates = {', '.join(counted_gates)}")

    @property
    def ground_energy(self) -> float:
        if self._ground_energy is None:
            return 0
        return self._ground_energy

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
        quiet: bool = False,
    ):
        # create the Aer backend with noise
        error_1 = noise.depolarizing_error(depolarerr_1q, 1)
        error_2 = noise.depolarizing_error(depolarerr_2q, 2)

        noise_model = noise.NoiseModel()

        instructions = [str(ins) for ins, _ in self.circuit.count_ops().items()]

        gate_1q = [ins for ins in instructions if _1q_gates(ins)]
        gate_2q = [ins for ins in instructions if _2q_gates(ins)]

        if not quiet:
            print("  Noise model, gates:")
            print(f"    Single qubit ({depolarerr_1q:>5.0e}) = {', '.join(gate_1q)}")
            print(f"    Double qubit ({depolarerr_2q:>5.0e}) = {', '.join(gate_2q)}")

        noise_model.add_all_qubit_quantum_error(error_1, gate_1q)
        noise_model.add_all_qubit_quantum_error(error_2, gate_2q)

        if not quiet:
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


class Execution:
    def __init__(
        self,
        fermionic_hamiltonian: FermionicOp,
        mapper: FermionicMapper,
        time: float = 1.0,
        basis_gates: list[str] = ["cx", "u3"],
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

        initial_state_data = Statevector.from_instruction(preparation).data
        initial_state_data[np.abs(initial_state_data) < Statevector.atol] = 0
        initial_state = Statevector(initial_state_data)

        print(f"    Ground energy = {result.groundenergy}")

        self.ground_energy = result.groundenergy

        print("  Construct and transpile Pauli evolution:")
        print(f"    Time duration    = {time}")
        print(f"    Basis gates      = {', '.join(basis_gates)}")
        print(f"    Synthesis method = {synthesis.__class__.__name__}")

        pe = PauliEvolutionGate(self.qubit_hamiltonian, time=time, synthesis=synthesis)
        self.circuit = QuantumCircuit(pe.num_qubits, pe.num_qubits)
        self.circuit.prepare_state(initial_state)
        self.circuit.append(pe, range(pe.num_qubits))
        self.circuit.measure(range(pe.num_qubits), range(pe.num_qubits))
        self.circuit = transpile(
            self.circuit, basis_gates=basis_gates, optimization_level=3
        )

        print("  Circuit summary:")
        print(f"    Depth = {self.circuit.depth()}")
        counted_gates = [
            f"{counts} {op.upper()}"
            for op, counts in {
                str(k): v for k, v in self.circuit.count_ops().items()
            }.items()
        ]
        print(f"    Gates = {', '.join(counted_gates)}")


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


def paulihedral_transpile(
    hamiltonian: SparsePauliOp, basis_gates: list[str] = ["u3", "cx"], time: float = 1.0
) -> tuple[QuantumCircuit, list[int]]:
    with tempfile.NamedTemporaryFile("w", delete=False) as hf:
        nq: int = hamiltonian.num_qubits  # type: ignore
        counter = [0 for _ in range(nq)]
        for pauli in hamiltonian:
            for i, op in enumerate(pauli.paulis[0].to_label()):  # type: ignore
                if op != "I":
                    counter[i] += 1
        # use negative to sort in reversed order
        permutation = sorted(range(len(counter)), key=lambda k: -counter[k])

        for pauli in hamiltonian:
            labels = "".join(map(pauli.paulis[0].to_label().__getitem__, permutation))  # type: ignore
            hf.write(f"{labels} {pauli.coeffs[0]}\n")
        hf.close()

        stdout_output = BytesIO(
            subprocess.run(
                [
                    "paulicc",
                    hf.name,
                    "-t",
                    str(time),
                    "-b",
                    ",".join(basis_gates),
                ],
                capture_output=True,
            ).stdout
        )

        return cast(QuantumCircuit, qpy.load(stdout_output)[0]), permutation


class PaulihedralDriver:
    def __init__(
        self,
        hamiltonian: FermionicOp,
        mapper: FermionicMapper,
        basis_gates: list[str] = ["u3", "cx"],
    ) -> None:
        self.hamiltonian = hamiltonian
        self.mapper = mapper

        self.circuit, self.permutation = paulihedral_transpile(
            self.mapper.map(self.hamiltonian), basis_gates  # type: ignore
        )

    @property
    def summary(self):
        return f"{self.circuit_gates['cx']}/{self.circuit_gates['u3']}/{self.circuit_depth}"

    @property
    def circuit_gates(self):
        return {str(k): v for k, v in self.circuit.count_ops().items()}

    @property
    def circuit_depth(self):
        return self.circuit.depth()


FERMIHEDRAL_MOLECULE = {
    8: [
        "_Z__Z_Z_",
        "Z___Z_X_",
        "___ZYY__",
        "____X__X",
        "_Y__Z_Z_",
        "____X__Z",
        "Y___Z_X_",
        "X___Z_X_",
        "_X__Z_Z_",
        "__Y_X__Y",
        "__Z_X__Y",
        "___ZYZ__",
        "___ZYX__",
        "____Z_Y_",
        "___YY___",
        "___XY___",
    ],
    6: [
        "ZZZZ_X",
        "ZZ_Z_Y",
        "Z__X__",
        "Z__Y__",
        "ZZXZ_X",
        "ZZYZ_X",
        "Y___X_",
        "Y___Y_",
        "YX__ZZ",
        "YY__ZZ",
        "X_____",
        "YZ__Z_",
    ],
    4: ["XZ_X", "YZ_X", "ZZZX", "_ZZY", "ZZX_", "Z_Y_", "ZXXZ", "ZYXZ"],
    2: ["YX", "XX", "_Z", "_Y"],
    12: [
        "____Y____Z__",
        "YY__X_______",
        "Z___X__Y____",
        "____YX___X__",
        "__ZZZ_______",
        "__X_Z_X_____",
        "YZ__X_______",
        "__Y_Z_____X_",
        "____YZ___X__",
        "__ZXZ_______",
        "X___X______Y",
        "X___X______X",
        "__Y_Z_____Z_",
        "YX__X_______",
        "____Y___YY__",
        "____Y___XY__",
        "____Y___ZY__",
        "____YY___X__",
        "__Y_Z_____Y_",
        "__X_Z_Y_____",
        "__X_Z_Z_____",
        "X___X______Z",
        "__ZYZ_______",
        "Z___X__X____",
    ],
    10: [
        "_____XY__X",
        "___X_XZ___",
        "_____XY__Y",
        "_____XY__Z",
        "_____XX_X_",
        "_____XX_Y_",
        "_____Z_Z__",
        "_____XX_Z_",
        "___Y_XZ___",
        "___Z_XZ___",
        "_X__XY____",
        "_ZY__Y____",
        "YY___Y____",
        "ZY___Y____",
        "_ZX__Y____",
        "_ZZ__Y____",
        "_____Z_X__",
        "_____Z_Y__",
        "_X__YY____",
        "XY___Y____",
    ],
}

FERMIHEDRAL_FERMIHUBBARD = {
    4: ["Z_XZ", "Z_YZ", "___X", "___Y", "ZXZZ", "ZYZZ", "X__Z", "Y__Z"],
    6: [
        "__Y___",
        "__X___",
        "ZZZYZZ",
        "ZZZXZZ",
        "Y_Z__Z",
        "X_Z__Z",
        "ZYZ_ZZ",
        "ZXZ_ZZ",
        "__Z__Y",
        "__Z__X",
        "Z_Z_YZ",
        "Z_Z_XZ",
    ],
    8: [
        "_ZZ__YZ_",
        "_ZZ__XZ_",
        "_ZZ_YZZ_",
        "_ZZ_XZZ_",
        "_YZ___Z_",
        "_XZ___Z_",
        "ZZZYZZZ_",
        "ZZZXZZZ_",
        "__Y_____",
        "__X_____",
        "YZZ_ZZZ_",
        "XZZ_ZZZ_",
        "__Z___Y_",
        "__Z___X_",
        "ZZZZZZZY",
        "ZZZZZZZX",
    ],
    10: [
        "_____XY__Y",
        "___X_XZ___",
        "XY___Y____",
        "YY___Y____",
        "_____XY__Z",
        "_____XY__X",
        "_____Z_Z__",
        "_____Z_X__",
        "_____XX_X_",
        "_____Z_Y__",
        "_ZX__Y____",
        "_ZY__Y____",
        "_____XX_Y_",
        "_____XX_Z_",
        "_ZZ__Y____",
        "ZY___Y____",
        "___Z_XZ___",
        "___Y_XZ___",
        "_X__YY____",
        "_X__XY____",
    ],
    12: [
        "__Y_Z_____X_",
        "__ZZZ_______",
        "X___X______Z",
        "YY__X_______",
        "____Y___YY__",
        "____Y___XY__",
        "X___X______X",
        "X___X______Y",
        "__ZXZ_______",
        "__ZYZ_______",
        "Z___X__Y____",
        "Z___X__X____",
        "__X_Z_Z_____",
        "__X_Z_X_____",
        "____YX___X__",
        "____YZ___X__",
        "__Y_Z_____Y_",
        "__Y_Z_____Z_",
        "YX__X_______",
        "YZ__X_______",
        "____Y___ZY__",
        "__X_Z_Y_____",
        "____Y____Z__",
        "____YY___X__",
    ],
    14: [
        "_____X__ZY___X",
        "_____X__YY___X",
        "__Y______Z___X",
        "__X______Z___X",
        "_______Z_X___X",
        "_____X__XY___X",
        "_Y___________Y",
        "__Z______Z___X",
        "_______X_X___X",
        "_______Y_X___X",
        "_Z____Z______Y",
        "_Z____Y______Y",
        "___XX________Z",
        "___YX________Z",
        "XX___________Y",
        "_Z____X______Y",
        "____Y_______ZZ",
        "___ZX________Z",
        "YX___________Y",
        "ZX___________Y",
        "____Y_______XZ",
        "____Y_______YZ",
        "____Z______Y_Z",
        "____Z______Z_Z",
        "_____Y___Y___X",
        "_____Z___Y___X",
        "____Z_____ZX_Z",
        "____Z_____YX_Z",
    ],
    16: [
        "_____Y_________Y",
        "_____Z_Y_Y______",
        "_____Z___X___X__",
        "_____Z___Z_X__Z_",
        "_____Z_X_Y__Z___",
        "_____Z_X_Y__Y___",
        "_____Z___Z_X__Y_",
        "_____Z___Z_X__X_",
        "Z____X__X_______",
        "X____X__X_______",
        "____ZY_________X",
        "____XY_________X",
        "_____Z_Z_Y______",
        "_____Z_X_Y__X___",
        "_____Y____X____Z",
        "_____Y____Z____Z",
        "Y____XX_X_______",
        "Y____XY_X_______",
        "__X__Y____Y____Z",
        "____YY_________X",
        "_Y___X__Y_______",
        "_X___X__Y_______",
        "__Y__Y____Y____Z",
        "__Z__Y____Y____Z",
        "___X_X__Z_______",
        "___Z_X__Z_______",
        "_____Z___X___Y__",
        "_____Z___X___Z__",
        "___Y_X__Z_______",
        "_Z___X__Y_______",
        "_____Z___Z_Y____",
        "_____Z___Z_Z____",
    ],
    18: [
        "____YZ______X_____",
        "____YX______XX____",
        "____Z_Y___X______Y",
        "____Z_Y___Z______Y",
        "____Z__________Y_Z",
        "____Z___________YX",
        "_X__X___Y_____Y___",
        "_X__X___Y_____Z___",
        "____Z___________ZX",
        "____Z___________XX",
        "YZY_X_____________",
        "YZZ_X_____________",
        "____YX______XY____",
        "____YX______XZ____",
        "____Z_Z__________Y",
        "____Z_X__________Y",
        "___ZY_______Y_____",
        "___XY_______Y_____",
        "_X__X__XX_________",
        "_X__X__YX_________",
        "_Y__X______Y______",
        "_Y__X______X______",
        "YZX_X_____________",
        "XZ__X_____________",
        "____Y____Y__Z_____",
        "____YY______X_____",
        "____Z_Y___Y______Y",
        "____Z__________X_Z",
        "____Y____Z__Z_____",
        "____Y____X__Z_____",
        "_X__X___Y_____X___",
        "_X__X__ZX_________",
        "_Y__X______Z______",
        "___YY_______Y_____",
        "_X__X___Z_________",
        "ZZ__X_____________",
    ],
}


class FermihedralMapper(FermionicMapper, ModeBasedMapper):
    def __init__(self, solutions: dict[int, list[str]]) -> None:
        super().__init__()
        self.solutions = solutions

    def map(self, second_q_ops: FermionicOp, *, register_length: int | None = None) -> SparsePauliOp:  # type: ignore
        return super().map(second_q_ops, register_length=register_length)  # type: ignore

    def solved(self, nmodes: int):
        return nmodes in self.solutions

    def get_solution(self, nmodes: int):
        assert (
            nmodes in self.solutions
        ), f"undefined Fermihedral Fermion-to-qubit mapping of {nmodes} modes"

        return [string.replace("_", "I") for string in self.solutions[nmodes]]

    def pauli_table(self, register_length: int):
        table = []

        pt = self.get_solution(register_length)

        for i in range(0, len(pt), 2):
            table.append((Pauli(pt[i]), Pauli(pt[i + 1])))

        return table

    @staticmethod
    def molecule():
        return FermihedralMapper(FERMIHEDRAL_MOLECULE)

    @staticmethod
    def fermi_hubbard():
        return FermihedralMapper(FERMIHEDRAL_FERMIHUBBARD)


def load_hamiltonian(filename: str):
    data = {}

    def transform_operator(op: str):
        opindex = int(op)
        if opindex < 0:
            return f"-_{abs(opindex) - 1}"
        else:
            return f"+_{abs(opindex) - 1}"

    with open(filename) as hfile:
        for line in map(str.strip, hfile.readlines()):
            data[" ".join(map(transform_operator, line.split(" ")))] = 1.0

    return FermionicOp(data)

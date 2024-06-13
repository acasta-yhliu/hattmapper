from sys import stdout
from typing import Callable, Literal, cast
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.problems import BaseProblem
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers.bravyi_kitaev_mapper import BravyiKitaevMapper
from qiskit_nature.second_q.mappers.jordan_wigner_mapper import JordanWignerMapper
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

from .ternary_tree_mapper import HamiltonianTernaryTreeMapper
from .extra_modes_tree_mapper import HamiltonianTernaryTreeExtraMapper

from dataclasses import dataclass
import math


@dataclass
class EvaluationResult:
    name: str
    basis_gates: list[str]

    method: list[str]
    pauli_weight: list[int]
    gate_counts: list[list[int]]
    depth: list[int]
    energy: list[float]

    def report(self, format: Literal["default", "csv"], output: str | None):
        if output is None:
            output_file = stdout
        else:
            output_file = open(output, "w")

        print(self.format(format), file=output_file)

        if output is not None:
            output_file.close()

    def format(self, format: Literal["default", "csv"]):
        if format == "default":
            return self._format_default()
        elif format == "csv":
            return self._format_csv()

    def _format_default(self):
        for i in range(len(self.method)):
            print(self.method[i] + " Pauli Weight: " + str(self.pauli_weight[i]))

    def _format_csv(self):
        print("warning: when reporting to csv, evaluation name is ignored")

        lines = []

        lines.append(
            f"Method,\"Pauli Weight\",{','.join(self.basis_gates)},Depth,Energy"
        )

        for i in range(len(self.method)):
            lines.append(
                f"\"{self.method[i]}\",{self.pauli_weight[i]},{','.join(map(str,self.gate_counts[i]))},{self.depth[i]},{self.energy[i]}"
            )

        return "\n".join(lines)


def _qiskit_lie_trotter(
    hamiltonian: SparsePauliOp, time: float, time_steps: int, basis_gates: list[str]
) -> QuantumCircuit:
    pe = PauliEvolutionGate(hamiltonian, time=time, synthesis=LieTrotter(time_steps))
    circ = QuantumCircuit(pe.num_qubits)
    circ.append(pe, range(pe.num_qubits))

    circuit = transpile(circ, basis_gates=basis_gates, optimization_level=3)
    return circuit


def evaluate(
    name: str,
    problem: BaseProblem | FermionicOp,
    *,
    extra_qubits: int = 0,
    basis_gates: list[str] = ["cx", "rx", "ry", "rz"],
    compile: Callable[
        [SparsePauliOp, float, int, list[str]], QuantumCircuit
    ] = _qiskit_lie_trotter,
):
    def pauli_weight(op: SparsePauliOp):
        weight = 0
        for pauli in op.paulis:
            label = pauli.to_label()  # type: ignore
            weight += len(label) - label.count("I")
        return weight

    records = []

    if isinstance(problem, BaseProblem):
        hamiltonian = problem.hamiltonian.second_q_op()
    else:
        hamiltonian = problem

    for mapper_name, mapper in [
        (
            "Our Method",
            HamiltonianTernaryTreeMapper(
                cast(FermionicOp, hamiltonian),
                nqubits=extra_qubits + hamiltonian.register_length,
            ),
        ),
        ("Method 2:", 
            HamiltonianTernaryTreeExtraMapper(
            cast(FermionicOp, hamiltonian), 
            nqubits=extra_qubits + math.ceil(hamiltonian.register_length * 1.5),)),
        ("Bravyi-Kitaev", BravyiKitaevMapper()),
        ("Jordan-Wigner", JordanWignerMapper()),
    ]:
        if isinstance(problem, BaseProblem):
            '''ground_energy = (
                GroundStateEigensolver(mapper, NumPyMinimumEigensolver())
                .solve(problem)
                .groundenergy
            )'''
            ground_energy = 0
        else:
            ground_energy = 0
            
        qh = mapper.map(hamiltonian)

        TIME = 1.0
        TIME_STEPS = 1

        circuit = compile(qh, TIME, TIME_STEPS, basis_gates)
        ops = {str(k): v for k, v in circuit.count_ops().items()}

        records.append(
            (
                mapper_name,
                pauli_weight(qh),
                [ops[gate] for gate in basis_gates],
                circuit.depth(),
                ground_energy,
            )
        )

    def split_n(l, n):
        return tuple([x[i] for x in l] for i in range(n))

    return EvaluationResult(name, basis_gates, *split_n(records, 5))

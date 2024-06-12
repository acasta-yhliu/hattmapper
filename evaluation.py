from sys import stdout
from typing import Literal, cast
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.problems import BaseProblem
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers.bravyi_kitaev_mapper import BravyiKitaevMapper
from qiskit_nature.second_q.mappers.jordan_wigner_mapper import JordanWignerMapper
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

from .ternary_tree_mapper import HamiltonianTernaryTreeMapper

from argparse import ArgumentParser
from dataclasses import dataclass


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
        raise NotImplementedError()

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


parser = ArgumentParser()
parser.add_argument(
    "-f",
    "--format",
    choices=["default", "csv"],
    default="default",
    type=str,
    help="report format of the evaluation result",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=None,
    help="output file for the evaluation result, default is stdout",
)
parser.add_argument(
    "-b",
    "--basis-gates",
    type=str,
    default="cx,rx,ry,rz",
    help="comma-split basis gates, default is %(default)s",
)


def evaluate(
    name: str,
    problem: BaseProblem | FermionicOp,
    *,
    basis_gates: list[str] = ["cx", "rx", "ry", "rz"],
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
        ("Our Method", HamiltonianTernaryTreeMapper(cast(FermionicOp, hamiltonian))),
        ("Bravyi-Kitaev", BravyiKitaevMapper()),
        ("Jordan-Wigner", JordanWignerMapper()),
    ]:
        if isinstance(problem, BaseProblem):
            ground_energy = (
                GroundStateEigensolver(mapper, NumPyMinimumEigensolver())
                .solve(problem)
                .groundenergy
            )
        else:
            ground_energy = 0

        qh = mapper.map(hamiltonian)

        TIME = 1.0
        TIME_STEPS = 1

        pe = PauliEvolutionGate(qh, time=TIME, synthesis=LieTrotter(TIME_STEPS))
        circ = QuantumCircuit(pe.num_qubits)
        circ.append(pe, range(pe.num_qubits))

        circuit = transpile(circ, basis_gates=basis_gates, optimization_level=3)
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


if __name__ == "__main__":
    args = parser.parse_args()

    evaluate(
        "LiH",
        PySCFDriver(
            atom="H 0 0 0; Li 0 0 1.6",
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        ).run(),
        basis_gates=args.basis_gates.split(","),
    ).report(args.format, output=args.output)

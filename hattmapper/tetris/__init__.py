from hattmapper.utility import load_molecule
from hattmapper import HATTMapper

from qiskit import transpile, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper

from .mypauli import pauliString
from .synthesis_lookahead import synthesis_lookahead, load_coupling_map

import sys
import os
from termcolor import colored


def load_sycamore_coupling_map():
    reduced = True
    pth = os.path.join("arch", "sycamore_64.txt")

    coupling = []
    n = 0
    with open(pth, "r") as file:
        lines = file.readlines()
        num_nodes, num_edges = map(int, lines[0].split()[:2])
        n = num_nodes

        # Add edges to the graph
        for edge in lines[1:]:
            node1, node2 = map(int, edge.split()[:2])
            coupling.append([node1, node2])
            coupling.append([node2, node1])

    return coupling


def coupling_from(arch: str):
    if arch == "sycamore":
        return load_sycamore_coupling_map()
    else:
        return load_coupling_map(arch)


class TetrisDriver:
    def __init__(
        self,
        hamiltonian: FermionicOp,
        mapper: FermionicMapper,
        basis_gates: list[str] = ["u3", "cx"],
        arch: str = "manhattan",
    ) -> None:
        self.hamiltonian = hamiltonian
        self.mapper = mapper
        self.basis_gates = basis_gates
        self.arch = arch

        self.transpile(self.mapper.map(self.hamiltonian))  # type: ignore

    def transpile(
        self, hamiltonian: SparsePauliOp, use_bridge=False, swap_coefficient=3, k=1
    ):
        coup = coupling_from(self.arch)
        pauli_sequence = []
        for op in hamiltonian.paulis:
            pauli_sequence.append(pauliString(op.to_label(), 1))  # type: ignore
        qc, _ = synthesis_lookahead(
            [pauli_sequence],
            arch=self.arch,
            use_bridge=use_bridge,
            swap_coefficient=swap_coefficient,
            k=k,
        )
        assert isinstance(qc, QuantumCircuit)
        pnq = qc.num_qubits
        self.circuit = transpile(
            qc,
            basis_gates=["u3", "cx"],
            coupling_map=coup,
            initial_layout=list(range(pnq)),
            optimization_level=3,
        )

    @property
    def summary(self):
        return {
            "cx": self.circuit_gates["cx"],
            "u3": self.circuit_gates["u3"],
            "depth": self.circuit_depth,
        }

    @property
    def circuit_gates(self):
        return {str(k): v for k, v in self.circuit.count_ops().items()}

    @property
    def circuit_depth(self):
        return self.circuit.depth()

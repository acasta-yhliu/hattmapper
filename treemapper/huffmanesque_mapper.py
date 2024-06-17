from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp, MajoranaOp
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.mappers.mode_based_mapper import ModeBasedMapper

from functools import reduce
import math
from tqdm import tqdm

from queue import PriorityQueue 
import heapdict


def _walk_string(
    i: int, mapping: dict[int, tuple[str, int]], nqubits: int, nstrings: int
):
    string = ["I" for _ in range(nqubits)]

    while i in mapping:
        op, i = mapping[i]
        string[i - nstrings] = op

    return "".join(string)


def _select_nodes(
    terms: list[tuple[int, ...]], nodes: set[int], nqubits: int, queue: heapdict, qubit_id: int
):
    selection: tuple[int, int, int] | None = None
    (xx, px) = queue.popitem()
    (yy, py) = queue.popitem()
    (zz, pz) = queue.popitem()
    selection = xx, yy, zz
    queue[qubit_id] = min(px,py,pz)
    assert selection is not None
    return selection


def _compile_fermionic_op(fermionic_op: FermionicOp, nqubits: int | None = None):
    if nqubits is None:
        nqubits = fermionic_op.register_length
    nstrings = 2 * nqubits + 1

    # turn the Hamiltonian into Majorana form and ignore the coefficients
    terms = [
        tuple(ms[1] for ms in term[0])
        for term in MajoranaOp.from_fermionic_op(fermionic_op).terms()
        if not math.isclose(abs(term[1]), 0)
    ]
    queue = heapdict.heapdict()
    frequencies = [0] * nstrings
    for term in terms:
        for op in term:
            frequencies[op] += 1
    
    for i in range(nstrings):
        queue[i] = frequencies[i]
        
    queue[nstrings - 1] = float("inf")

    # generate all terms, all initial nodes (strings)
    nodes = set(range(nstrings))

    # mapping, node -> branch, parent
    mapping: dict[int, tuple[str, int]] = {}

    for round in range(nqubits):
        # the qubit that will become the new parent
        qubit_id = nstrings + round

        # select the node with lowest Pauli weight
        selection = _select_nodes(terms, nodes, nqubits, queue, qubit_id)

        # update nodes and terms, record solution
        for node, op in zip(selection, "XYZ"):
            nodes.remove(node)
            mapping[node] = (op, qubit_id)
        nodes.add(qubit_id)

        # reduce the Hamiltonian
        for i in range(len(terms)):
            term = tuple(idx for idx in terms[i] if idx not in selection)
            terms[i] = (
                term if (len(terms[i]) - len(term)) % 2 == 0 else (term + (qubit_id,))
            )
        terms = list(filter(lambda x: len(x) != 0, terms))

    # generate solution
    return [_walk_string(i, mapping, nqubits, nstrings) for i in range(nstrings - 1)]


class HamiltonianTernaryTreeHuffmanMapper(ModeBasedMapper, FermionicMapper):
    def __init__(
        self, loader: FermionicOp | list[str], nqubits: int | None = None
    ) -> None:
        if isinstance(loader, FermionicOp):
            raw_pauli_table = _compile_fermionic_op(loader, nqubits)
        else:
            raw_pauli_table = loader

        self.raw_pauli_table = raw_pauli_table

        self.nqubits = nqubits if nqubits is not None else len(raw_pauli_table[0])

    def map(self, second_q_ops: FermionicOp, *, _: int | None = None) -> SparsePauliOp:  # type: ignore
        return super().map(second_q_ops, register_length=self.nqubits)  # type: ignore

    def pauli_table(self, register_length: int):
        table = []

        for i in range(0, len(self.raw_pauli_table), 2):
            table.append(
                (Pauli(self.raw_pauli_table[i]), Pauli(self.raw_pauli_table[i + 1]))
            )

        return table

    def save(self, path: str):
        with open(path, "w") as pauli_table_file:
            pauli_table_file.write("\n".join(self.raw_pauli_table))

    @staticmethod
    def load(path: str):
        with open(path, "r") as pauli_table_file:
            lines = list(map(str.strip, pauli_table_file.readlines()))
            return HamiltonianTernaryTreeHuffmanMapper(lines)

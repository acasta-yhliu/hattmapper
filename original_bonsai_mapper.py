from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp, MajoranaOp
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.mappers.mode_based_mapper import ModeBasedMapper

from architecture import Architecture

from itertools import permutations
from functools import reduce
import math
from tqdm import tqdm


def _walk_string(
    i: int, mapping: dict[int, tuple[str, int]], nqubits: int
):
    string = ["I" for _ in range(Architecture.nqubits)]
    qubit = int(i/2)
    if qubit == nqubits:
        x = 0
        while x < nqubits:
            string[x] = "Z"
            x = x * 3 + 3
    else:
        ops = ["X", "Y", "Z"]
        while qubit > 0:  
            parent = int((qubit - 1)/3)
            op = ops[qubit % 3 - 1]
            string[parent] = op
            qubit = parent
        qubit = int(i/2)
        string[qubit] = ops[i % 2]
        qubit = qubit * 3 + (i % 2) + 1
        while qubit < nqubits:
            string[qubit] = "Z"
            qubit = qubit * 3 + 3
        

    return "".join(string)

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
    # generate all terms, all initial nodes (strings)
    # mapping, node -> branch, parent
    mapping: dict[int, tuple[str, int]] = {}
    #mapping for parent -> (x,y,z)
    return [_walk_string(i, mapping, nqubits) for i in range(nstrings - 1)]


class HamiltonianOriginalBonsaiMapper(ModeBasedMapper, FermionicMapper):
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
            return HamiltonianOriginalBonsaiMapper(lines)

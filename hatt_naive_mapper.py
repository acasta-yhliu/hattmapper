from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp, MajoranaOp
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.mappers.mode_based_mapper import ModeBasedMapper

from itertools import combinations
from functools import reduce
from math import comb, isclose
from tqdm import tqdm


def _walk_string(
    i: int, mapping: dict[int, tuple[str, int]], nqubits: int, nstrings: int
):
    string = ["I" for _ in range(nqubits)]

    while i in mapping:
        op, i = mapping[i]
        string[i - nstrings] = op

    return "".join(string)


def _walk_ternary_string(
    i: int, mapping: dict[int, tuple[str, int]], nqubits: int, nstrings: int
):
    string = ["I" for _ in range(nqubits)]

    index = None

    while i in mapping:
        op, i = mapping[i]
        qubit_id = i - nstrings
        string[qubit_id] = op

        if op != "Z" and (index is None):
            index = 2 * qubit_id if op == "X" else 2 * qubit_id + 1

    # assert index is not None

    return "".join(string), index


def _select_nodes(
    terms: list[tuple[int, ...]], nodes: set[int], round: int, nqubits: int
):
    minimum_pauli_weight = float("inf")
    selection: tuple[int, int, int] | None = None

    for xx, yy, zz in tqdm(
        combinations(nodes, 3),
        total=comb(len(nodes), 3),
        leave=False,
        desc=f"Qubit {round + 1}/{nqubits}",
        colour="#03925e",
        ascii="░▒█",
    ):
        # calculate Pauli weight of each selection
        pauli_weight = 0
        for term in terms:
            term = reduce(
                lambda x, y: (x[0] ^ y[0], x[1] ^ y[1]),
                map(
                    lambda i: (
                        (True, False)
                        if i == xx
                        else (
                            (True, True)
                            if i == yy
                            else ((False, True) if i == zz else (False, False))
                        )
                    ),
                    term,
                ),
                (False, False),
            )
            if term[0] or term[1]:
                pauli_weight += 1

        # is the selection better ?
        if pauli_weight < minimum_pauli_weight:
            minimum_pauli_weight = pauli_weight
            selection = xx, yy, zz

    assert selection is not None
    return selection


def _compile_fermionic_op(
    fermionic_op: FermionicOp | MajoranaOp, nqubits: int | None = None
):
    if nqubits is None:
        nqubits = fermionic_op.register_length

    majorana_op = (
        MajoranaOp.from_fermionic_op(fermionic_op)
        if isinstance(fermionic_op, FermionicOp)
        else fermionic_op
    )

    nstrings = 2 * nqubits + 1
    # turn the Hamiltonian into Majorana form and ignore the coefficients
    terms = [
        tuple(ms[1] for ms in term[0])
        for term in majorana_op.terms()
        if not isclose(abs(term[1]), 0)
    ]
    # generate all terms, all initial nodes (strings)
    nodes = set(range(nstrings))

    # mapping, node -> branch, parent
    mapping: dict[int, tuple[str, int]] = {}

    for round in range(nqubits):
        # the qubit that will become the new parent
        qubit_id = nstrings + round

        # select the node with lowest Pauli weight
        selection = _select_nodes(terms, nodes, round, nqubits)

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


class HATTNaiveMapper(ModeBasedMapper, FermionicMapper):
    def __init__(
        self, loader: FermionicOp | MajoranaOp | list[str], nqubits: int | None = None
    ) -> None:
        if isinstance(loader, FermionicOp) or isinstance(loader, MajoranaOp):
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
            return HATTNaiveMapper(lines)


class TernaryTreeMapper(FermionicMapper, ModeBasedMapper):
    def __init__(self, *, pair: bool = False) -> None:
        super().__init__()
        self.pair = pair

    def map(self, second_q_ops: FermionicOp, *, register_length: int | None = None) -> SparsePauliOp:  # type: ignore
        return super().map(second_q_ops, register_length=register_length)  # type: ignore

    def pauli_table(self, register_length: int):
        table = []

        pt = self.majorana_table(register_length)

        for i in range(0, len(pt), 2):
            table.append((Pauli(pt[i]), Pauli(pt[i + 1])))

        return table

    def majorana_table(self, register_length: int):
        nqubits = register_length

        mapping: dict[int, tuple[str, int]] = {}

        # initial slots : all the slots of the root
        free_slots: list[tuple[int, str]] = [
            (2 * nqubits + 1, "Z"),
            (2 * nqubits + 1, "Y"),
            (2 * nqubits + 1, "X"),
        ]

        # 1. insert all qubits
        for n in range(2 * nqubits + 2, 3 * nqubits + 1):
            parent, branch = free_slots.pop(0)
            mapping[n] = (branch, parent)
            free_slots.extend([(n, "X"), (n, "Y"), (n, "Z")])

        for n in range(2 * nqubits + 1):
            parent, branch = free_slots.pop(0)
            mapping[n] = (branch, parent)

        if not self.pair:
            return [
                _walk_string(i, mapping, nqubits, 2 * nqubits + 1)
                for i in range(2 * nqubits)
            ]
        else:
            return {
                index: string
                for string, index in [
                    _walk_ternary_string(i, mapping, nqubits, 2 * nqubits + 1)
                    for i in range(2 * nqubits + 1)
                ] if index is not None
            }

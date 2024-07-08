from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp, MajoranaOp
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.mappers.mode_based_mapper import ModeBasedMapper

from itertools import combinations, permutations
from functools import reduce
import math
from tqdm import tqdm


def _walk_string(
    i: int, mapping: dict[int, tuple[str, int]], nqubits: int, nstrings: int
):
    string = ["I" for _ in range(nqubits)]

    while i in mapping:             #move up the tree until we get to the root (the root has no parent and is not in mapping)
        op, i = mapping[i]
        string[i - nstrings] = op

    return "".join(string)


def print_tree(i: int, tree: dict[int, tuple[int, int, int]], nstrings: int, string: str):
    if i in tree:       #print recursively. This helps retain some structure that we can observe.
        string[i - nstrings] = "X"
        print_tree(tree[i][0], tree, nstrings, string)
        string[i - nstrings] = "Y"
        print_tree(tree[i][1], tree, nstrings, string)
        string[i - nstrings] = "Z"
        print_tree(tree[i][2], tree, nstrings, string)    
        string[i - nstrings] = "I"
    else:
        for j in range(len(string)):
            if string[j] == "I":
                string[j] = " "
        string.reverse()
        print(f"{str(i) : <3}" + " " + "".join(string))
        string.reverse()
    
    

def _select_nodes(
    terms: list[tuple[int, ...]], nodes: set[int], round: int, nqubits: int, tree: dict[int, tuple[int, int, int]], mapping: dict[int, tuple[str, int]]
):
    minimum_pauli_weight = float("inf")
    selection: tuple[int, int, int] | None = None

    for xx, zz in tqdm(     # bash every permutation of xx, zz and greedily choose best
        permutations(nodes, 2),
        total=math.perm(len(nodes), 2),
        leave=False,
        desc=f"Qubit {round + 1}/{nqubits}",
        colour="#03925e",
        ascii="░▒█",
    ):
        # calculate Pauli weight of each selection
        vac = nqubits * 2
        while vac in mapping:
            _, vac = mapping[vac]
        if xx == vac:               # the VAC operator should have a mapping to only Z operators. This enforces that.
            continue
        
        i = xx
        while i in tree:            #find the Z child of the X branch
            i = tree[i][2]
        if i % 2 == 0:              #now the Y branch will be the corresponding other majorana operator
            yy = i + 1
        else:
            yy = i - 1
        while yy in mapping:        # we know whatthe other majorana operator is. We now find what node it is under.
            _, yy = mapping[yy]
        if zz == yy:                #check to make sure it isn't the same
            continue
            
        pauli_weight = 0
        for term in terms:
            # for each mode in the term, map it to the corresponding gate based on the selection (or identity).
            # XOR on simplectic vectors is equivalent to gates minus phase change.
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
            # considered as a simplectic vector; if either is true, then we have a gate acting on this qubit.
            if term[0] or term[1]:
                pauli_weight += 1

        # is the selection better ?
        # if yes, update selection.
        if pauli_weight < minimum_pauli_weight:
            minimum_pauli_weight = pauli_weight
            selection = xx, yy, zz
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
    # generate all terms, all initial nodes (strings)
    nodes = set(range(nstrings))

    # mapping, node -> branch, parent
    mapping: dict[int, tuple[str, int]] = {}
    #mapping for parent -> (x,y,z)
    tree: dict[int, tuple[int, int, int]] = {}
    
    for round in range(nqubits):
        # the qubit that will become the new parent
        qubit_id = nstrings + round

        # select the node with lowest Pauli weight
        selection = _select_nodes(terms, nodes, round, nqubits, tree, mapping)

        # update nodes and terms, record solution
        for node, op in zip(selection, "XYZ"):
            nodes.remove(node)
            mapping[node] = (op, qubit_id)
            tree[qubit_id] = selection
        nodes.add(qubit_id)

        # reduce the Hamiltonian
        # this allows us to consider individual qubits when computing intermediary pauli weights.
        for i in range(len(terms)):
            term = tuple(idx for idx in terms[i] if idx not in selection)
            terms[i] = (
                term if (len(terms[i]) - len(term)) % 2 == 0 else (term + (qubit_id,))  
                #if two modes were in the selection, they are siblings under a common node, and thus will cancel each other out. Otherwise, add in the new node.
            )
        terms = list(filter(lambda x: len(x) != 0, terms))

    # generate solution
    # next statement helps see tree structure
    #print_tree(nstrings + nqubits - 1, tree, nstrings, ["I" for _ in range(nqubits)])
    return [_walk_string(i, mapping, nqubits, nstrings) for i in range(nstrings - 1)]


class HamiltonianTernaryBonsaiMapper(ModeBasedMapper, FermionicMapper):
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
            return HamiltonianTernaryBonsaiMapper(lines)

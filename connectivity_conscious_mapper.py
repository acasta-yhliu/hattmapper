from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp, MajoranaOp
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.mappers.mode_based_mapper import ModeBasedMapper

from itertools import combinations
from functools import reduce
from math import comb, isclose
from tqdm import tqdm

import random

## We define a subroutine that takes in a physical connectivity graph P 
## as input, and computes a "close-enough" spanning ternary subtree to minimize
## SWAP gates and reduce circuit complexity.
def _qubit_spanning_tree(P: dict[int, set[int]]) -> tuple[set[int], set[tuple[int, int]]]:
    # Determine root node r: node which has the min max topological dist between any other node in P
    r = _find_root(P)

    # Define initial layer, L_0 = r, with height h = 0, and tree T = (V, E), where V and E are empty
    L: list[set[int]] = [{r}]
    h = 0
    V_T: set[int] = {}
    E_T: set[tuple[int, int]] = {}

    # Body of subroutine
    while len(L[h]) > 0:
        L_h_1 = {}
        for v in L[h]:
            # let the set of unassigned neighbors of v be N_v, and ...
            N_v = {}
            for w, N_w in P: # iterate to find neighbors of v that are not in V_T (a.k.a. unassigned)
                if w in P.get(v) and w not in V_T:
                    N_v.add(w)
            if len(N_v) > 3:
                # we want to select three random neighbors, if there exist more than 3
                N_v = set(random.sample(N_v, 3))
            L_h_1 = L_h_1.union(N_v)
            V_T = V_T.union(N_v)
            E: set[tuple[int, int]] = {}
            # track the new edges of our spanning tree
            for w in N_v:
                edge = (v, w)
                E.add(edge)
            E_T = E_T.union(E)
        L.append(L_h_1)
        h += 1
    
    # enumerate/iterate through all vertices in V_P but not in V_T
    for u in V_P.difference(V_T):
        # find vertices in V_T that are still available to connect
        # NOTE: NVM, this and the next part seem kind of annoying, 
        # will make a separate function for them later
        #A = {}
        #for node in V_T:

        C: set[int] = {}

        if len(C) > 1:
            C = set(random.sample(C, 1))

        V_T = V_T.union(C)
        E: set[tuple[int]] = {}
        for v in C: # going to change this later, definitely a better built-in way to do extract the singleton
            edge = (u, v)
            E.add(edge)
        E_T = E_T.union(E)
    
    # return T
    return (V_T, E_T)


## Finds a root node r for the subroutine above.
## NOTE: this approach is quite naive (mainly for general prototyping) 
##       and can possibly be optimized for runtime performance.
def _find_root(P: dict[int, set[int]]) -> int:
    min_max_dist = 10000 # make it a large number for now
    candidate_root: int | None = None
    for node, _ in P:
        dist = _bfs(node, P) # run BFS to determine the max distance of any node from this node
        if dist < min_max_dist:
            min_max_dist = dist
            candidate_root = node

    assert candidate_root is not None
    return candidate_root

## Runs BFS from a specified node on the given graph P,
## returns longest topological dist from given node
def _bfs(node: int, P: dict[int, set[int]]) -> int:
    furthest = 0 # initialize furthest distance to be 0 at first
    visited = {}
    queue = []
    dist_queue = []
    queue.append(node)
    dist_queue.append(0)
    while len(queue) > 0: # keep exploring until all nodes have been "visited"
        curr = queue.popleft()
        curr_dist = dist_queue.popleft()
        for elt in P.get(curr): # examine all of the node's neighbors
            if elt not in visited: # check to make sure the neighbor is not already "visited"
                queue.append(elt)
                dist_queue.append(curr_dist + 1)
                if curr_dist + 1 > furthest: # update furthest dist if curr dist is larger
                    furthest = curr_dist + 1
        visited.add(curr) # mark current node as "visited"

    return furthest


def _walk_string(
    i: int, mapping: dict[int, tuple[str, int]], nqubits: int, nstrings: int
):
    string = ["I" for _ in range(nqubits)]

    while i in mapping:
        op, i = mapping[i]
        string[i - nstrings] = op

    return "".join(string)


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


def _compile_fermionic_op(fermionic_op: FermionicOp, nqubits: int | None = None):
    if nqubits is None:
        nqubits = fermionic_op.register_length

    nstrings = 2 * nqubits + 1
    # turn the Hamiltonian into Majorana form and ignore the coefficients
    terms = [
        tuple(ms[1] for ms in term[0])
        for term in MajoranaOp.from_fermionic_op(fermionic_op).terms()
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


class HamiltonianTernaryTreeMapper(ModeBasedMapper, FermionicMapper):
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
            return HamiltonianTernaryTreeMapper(lines)

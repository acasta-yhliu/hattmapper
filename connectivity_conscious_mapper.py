from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp, MajoranaOp
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.mappers.mode_based_mapper import ModeBasedMapper

from qiskit.transpiler import CouplingMap
import qiskit_ibm_runtime.fake_provider
from architecture import Architecture

from itertools import combinations, permutations
from functools import reduce
import math
from tqdm import tqdm

import random
import copy

## We define a subroutine that takes in a physical connectivity graph P 
## as input, and computes a "close-enough" mapping to minimize
## SWAP gates and reduce circuit complexity.
def mapper(
           tree: dict[int, tuple[int, int, int]], 
           mapping: dict[int, tuple[str,int]],
           heights: dict[int, int],
           nqubits: int,
           ) -> dict[int,int]:
    # Determine root node r: node which has the min max topological dist between any other node in P
    h = max(heights.values())
    treepath: list[int] = []
    i = list(heights.keys())[list(heights.values()).index(h)]
    P = Architecture.adj_list
    while i in mapping:
        treepath.append(i)
        _, i = mapping[i]
    treepath.append(i)
    treepath.pop(0)
    r, longest = _find_root(P, h)
    #assigns the longest path in the tree to the longest path in our bfs from the root.
    physical: dict[int,int] = {}
    longest.reverse()
    if len(treepath) == len(longest):
        for i in range(len(treepath)):
            physical[treepath[i]] = longest[i]
    #root is always the root of the tree
    print(treepath)
    print(physical)
    physical[nqubits * 3] = r
    # Body of subroutine
    for i in range(nqubits * 3, nqubits * 2 + 1, -1):
        if i in physical:
            continue
        _, parent = mapping[i]
        for w in P[physical[parent]]: # iterate to find neighbors of v that are unassigned
            if w not in physical.values():  #find one thats not assigned? assign it.
                physical[i] = w
                break
    
        
    # # enumerate/iterate through all vertices that have not been mapped to
    # for u in set(P.keys()).difference(physical.values()):
    #     #all this does currently is: finds a qubit that has not been mapped onto the tree yet randomly, and assign it to that in the set.
    #     #this definitely could be optimized, such that the qubit is assigned as a child of node with qubit x such that physical distance to x is minimized
    #     #(thus lowering swaps needed)
    #     C: set[int] = set(range(nqubits * 2 + 1, nqubits * 3 + 1, 1)).difference(set(physical.keys()))
        
    #     if len(C) > 1:
    #         C = set(random.sample(C, 1))
    #     if len(C) == 1:
    #         [v] = C
    #         physical[v] = u
            
    # if a tree node has not yet recieved a mapping, map it to the physically closest qubit to its parent
    for i in range(nqubits * 2 + 1, nqubits * 3 + 1, 1):
        if i in physical:
            continue
        min_dist = float("inf")
        for u in set(P.keys()).difference(physical.values()):
            if Architecture.coupling_map.shortest_undirected_path(physical[mapping[i]], u) < min_dist:
                min_dist = Architecture.coupling_map.shortest_undirected_path(physical[mapping[i]], u)
                closest = u
        physical[i] = closest
        
    
    # return T
    print(physical)
    return physical
    


## Finds a root node r for the subroutine above.
## NOTE: this approach is quite naive (mainly for general prototyping) 
##       and can possibly be optimized for runtime performance.
def _find_root(P: dict[int, set[int]], h: int) -> tuple[int, list[int]]:
    min_max_dist = float("inf") # make it a large number for now
    candidate_root: int | None = None
    candidate_path: list[int] | None = None
    for node in P.keys():
        (dist, path) = _bfs(node, P) # run BFS to determine the max distance of any node from this node
        if dist < min_max_dist and dist >= h - 1:
            min_max_dist = dist
            candidate_root = node
            candidate_path = path
            
    #if there isn't a long enough path, just choose the most central
    #this might not work?
    if min_max_dist != h - 1:
        for node in P.keys():
            (dist, path) = _bfs(node, P) # run BFS to determine the max distance of any node from this node
            if dist < min_max_dist:
                min_max_dist = dist
                candidate_root = node
                candidate_path = path
                
    assert candidate_root is not None
    assert candidate_path is not None
    return (candidate_root, candidate_path)

## Runs BFS from a specified node on the given graph P,
## returns longest topological dist from given node
## and the nodes along that path
def _bfs(node: int, P: dict[int, set[int]]) -> tuple[int, list[int]]:
    furthest = 0 # initialize furthest distance to be 0 at first
    longest_path: list[int] = [] # will store the longest path


    visited: set[int] = set()
    queue = [] # the list that serves as a queue that determines order of exploration in BFS
    dist_queue = [] # keeps track of the distance we're at
    paths: list[list[int]] = []

    queue.append(node)
    dist_queue.append(0)

    init_path = [node]
    paths.append(init_path) # initial path only contains root node for now

    while len(queue) > 0: # keep exploring until all nodes have been "visited"
        curr = queue[0]
        queue.pop(0)
        curr_dist = dist_queue[0]
        dist_queue.pop(0)
        curr_path = paths[0]
        paths.pop(0)
        for elt in P[curr]: # examine all of the node's neighbors
            if elt not in visited:
                queue.append(elt)
                dist_queue.append(curr_dist + 1)

                curr_path_copy = copy.deepcopy(curr_path)
                curr_path_copy.append(elt)
                paths.append(curr_path_copy)

                if curr_dist + 1 > furthest: # update furthest dist if curr dist is larger
                    furthest = curr_dist + 1
                    longest_path = curr_path_copy
                visited.add(curr) # mark current node as "visited"

    return (furthest, longest_path)


def _walk_string(
    i: int, mapping: dict[int, tuple[str, int]], nstrings: int, phys: dict[int, int]
):
    string = ["I" for _ in range(Architecture.nqubits)]

    while i in mapping:             #move up the tree until we get to the root (the root has no parent and is not in mapping)
        op, i = mapping[i]
        string[phys[i]] = op

    return "".join(string)


def print_tree(i: int, tree: dict[int, tuple[int, int, int]], nstrings: int, string: str, phys: dict[int,int]):
    if i in tree:       #print recursively. This helps retain some structure that we can observe.
        string[phys[i]] = "X"
        print_tree(tree[i][0], tree, nstrings, string)
        string[phys[i]] = "Y"
        print_tree(tree[i][1], tree, nstrings, string)
        string[phys[i]] = "Z"
        print_tree(tree[i][2], tree, nstrings, string)    
        string[phys[i]] = "I"
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
    
    physical: dict[int, int] = {}
    
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
    #print_tree(nstrings + nqubits - 1, tree, nstrings, ["I" for _ in range(nqubits)], physical)
    
    heights: dict[int,int] = {}
    for i in range(nqubits + nstrings - 1, -1, -1):
        if i in mapping:
            _, j = mapping[i]
            heights[i] = heights[j] + 1
        else:
            heights[i] = 0
    
    
    P = Architecture.adj_list
    physical = mapper(tree, mapping, heights, nqubits)
    return [_walk_string(i, mapping, nstrings, physical) for i in range(nstrings - 1)]


class HamiltonianTernaryConnectivityMapper(ModeBasedMapper, FermionicMapper):
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
            return HamiltonianTernaryConnectivityMapper(lines)

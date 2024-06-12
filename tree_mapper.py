from qiskit.quantum_info import Pauli
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper

from itertools import product, combinations
from functools import reduce
from math import comb
from tqdm import tqdm

class PauliTable:
    def __init__(self, pauli_table: list[str], *, check: bool = False) -> None:
        if check:
            assert len(pauli_table) % 2 == 0

            nqubits = len(pauli_table) // 2

            for pauli_string in pauli_table:
                assert len(pauli_string) == nqubits
                assert set(pauli_string).issubset({"I", "X", "Y", "Z"})

        self.pauli_table = pauli_table

    def mapper(self) -> FermionicMapper:
        @classmethod
        def pauli_table(cls, register_length: int) -> list[tuple[Pauli, Pauli]]:
            return [(Pauli(self.pauli_table[i]), Pauli(self.pauli_table[i+1])) for i in range(0, len(self.pauli_table), 2)]

        return type(f"PauliTableMapper", (FermionicMapper,), {"pauli_table": pauli_table})()
    
    def save(self, path: str):
        with open(path, "w") as pauli_table_file:
            pauli_table_file.write("\n".join(self.pauli_table))

    @staticmethod
    def load(path: str, *, check: bool = False):
        with open(path, "r") as pauli_table_file:
            lines = list(map(str.strip, pauli_table_file.readlines()))
            return PauliTable(lines, check=check)

def _walk_string(i : int, mapping : dict[int, tuple[str, int]], nqubits: int, nstrings: int):
        string = ["I" for _ in range(nqubits)]

        while i in mapping:
            op, i = mapping[i]
            string[i - nstrings] = op

        return "".join(string)

def _select_nodes(terms: list[tuple[int, ...]], nodes: set[int], round: int):
    minimum_pauli_weight = float("inf")
    selection : tuple[int, int, int] | None = None

    for xx, yy, zz in tqdm(combinations(nodes, 3), total=comb(len(nodes), 3), position=1, leave=False, desc=f"growing node {round}"):
        # calculate Pauli weight of each selection
        pauli_weight = 0
        for term in terms:
            term = reduce(lambda x, y: (x[0] ^ y[0], x[1] ^ y[1]), map(lambda i: (True, False) if i == xx else ((True, True) if i == yy else ((False, True) if i == zz else (False, False))), term), (False, False))
            if term[0] or term[1]:
                pauli_weight += 1
        
        # is the selection better ?
        if pauli_weight < minimum_pauli_weight:
            minimum_pauli_weight = pauli_weight
            selection = xx, yy, zz

    assert selection is not None
    return selection

def compile_fermionic_op(fermionic_op : FermionicOp, nqubits: int | None = None) -> PauliTable:
    if nqubits is None:
        nqubits = fermionic_op.register_length
    nstrings = 2 * nqubits + 1

    # turn the Hamiltonian into Majorana form and ignore the coefficients
    majorana_terms: set[tuple[int, ...]] = set()
    for terms, _ in fermionic_op.terms():
        majorana_terms.update(map(lambda x: tuple(sorted(x)), product(*((2 * i, 2 * i + 1) for _, i in terms))))
    
    # generate all terms, all initial nodes (strings)
    terms = list(majorana_terms)
    nodes = set(range(nstrings))

    # mapping, node -> branch, parent
    mapping : dict[int, tuple[str, int]] = {}

    for round in tqdm(range(nqubits), position=0, desc="ternary tree"):
        # the qubit that will become the new parent
        qubit_id = nstrings + round

        # select the node with lowest Pauli weight
        selection = _select_nodes(terms, nodes, round)

        # update nodes and terms, record solution
        for node, op in zip(selection, "XYZ"):
            nodes.remove(node)
            mapping[node] = (op, qubit_id)
        nodes.add(qubit_id)
        
        # reduce the Hamiltonian
        for i in range(len(terms)):
            term = tuple(idx for idx in terms[i] if idx not in selection)
            terms[i] = term if (len(terms[i]) - len(term)) % 2 == 0 else (term + (qubit_id,))
        terms = list(filter(lambda x: len(x) != 0, terms))

    # generate solution
    return PauliTable([_walk_string(i, mapping, nqubits, nstrings) for i in range(nstrings - 1)])



if __name__ == "__main__":
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_nature.units import DistanceUnit
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers.bravyi_kitaev_mapper import BravyiKitaevMapper
    from qiskit_nature.second_q.mappers.jordan_wigner_mapper import JordanWignerMapper
    from qiskit_algorithms import NumPyMinimumEigensolver
    from qiskit_nature.second_q.algorithms import GroundStateEigensolver

    import logging

    logging.basicConfig(level=logging.INFO)

    logging.info("perform testing for ternary tree mapper with H2 problem")
    problem = PySCFDriver(
        atom="H 0 0 0; Li 0 0 1.6",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    ).run()

    hamiltonian: FermionicOp = problem.hamiltonian.second_q_op()

    def pauli_weight(op: SparsePauliOp):
        weight = 0
        for pauli in op.paulis:
            label = pauli.to_label() # type: ignore
            weight += len(label) - label.count("I")
        return weight

    for name, mapper in [("tree", compile_fermionic_op(hamiltonian).mapper()), ("bravyi-kitaev", BravyiKitaevMapper()), ("jordan-wigner", JordanWignerMapper())]:
        result = GroundStateEigensolver(mapper, NumPyMinimumEigensolver()).solve(problem)
        print(name, "weight =", pauli_weight(mapper.map(hamiltonian)),"ground energy =", result.groundenergy)
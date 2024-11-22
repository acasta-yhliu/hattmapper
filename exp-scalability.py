from hatt_mapper import HATTMapper
from hatt_naive_mapper import HATTNaiveMapper
from qiskit_nature.second_q.operators import FermionicOp

from collections import defaultdict

import time

RESULT = defaultdict(list)


class Stopwatch:
    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time

        RESULT[self.name].append(self.elapsed_time)


MAXMODES = 20

for modes in range(2, MAXMODES):
    print(f"{modes} Modes")

    def to_acop(m: int, modes: int):
        if m % 2 == 0:
            j = m // 2
            return FermionicOp({f"+_{j}": 1.0, f"-_{j}": 1.0}, num_spin_orbitals=modes)
        else:
            j = (m - 1) // 2
            return FermionicOp(
                {f"+_{j}": 1.0j, f"-_{j}": -1.0j}, num_spin_orbitals=modes
            )

    hamiltonian = FermionicOp({}, num_spin_orbitals=modes)
    for i in range(modes):
        hamiltonian += to_acop(i, modes)

    with Stopwatch("naive"):
        HATTNaiveMapper(hamiltonian)

    with Stopwatch("optimized"):
        HATTMapper(hamiltonian)

with open("tests/scalability.csv", "w") as of:
    for n, o in zip(RESULT["naive"], RESULT["optimized"]):
        print(f"{n},{o}", file=of)

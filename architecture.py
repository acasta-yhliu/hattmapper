from typing import Literal
import json

from qiskit.transpiler import CouplingMap
import qiskit_ibm_runtime.fake_provider

import warnings

class Architecture():
    hardware = qiskit_ibm_runtime.fake_provider.FakeAthensV2()
    nqubits = hardware.num_qubits
    coupling_map = hardware.coupling_map
    adj_list: dict[int, set[int]] = {}
    for [x,y] in coupling_map:
        if x not in adj_list:
            adj_list[x] = set()
        adj_list[x].add(y)
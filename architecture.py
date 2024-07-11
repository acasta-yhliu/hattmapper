import qiskit_ibm_runtime.fake_provider

#This class allows us to quickly change around our files for quicker testing.
class Architecture():
    hardware = qiskit_ibm_runtime.fake_provider.FakeAthensV2()
    nqubits = hardware.num_qubits
    coupling_map = hardware.coupling_map
    adj_list: dict[int, set[int]] = {}
    for [x,y] in coupling_map:
        if x not in adj_list:
            adj_list[x] = set()
        adj_list[x].add(y)
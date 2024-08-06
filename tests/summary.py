import csv
import sys
from statistics import mean

if len(sys.argv) != 2:
    print("Help: summary.py {casename}")
    sys.exit(1)

casename = sys.argv[1]


def tovalue(record: str, field: str):
    if record == "--":
        return SUPERBIG
    s = record.split("/")
    return int(s[0]) if field == "cx" else int(s[2])


def toint(a: str):
    if a == "--":
        return SUPERBIG
    else:
        return int(a)


SUPERBIG = 9999999999999999999


def improve(hatt: int, old: int):
    return (old - hatt) * 100 / old


with open(f"pauli-weight/{casename}.csv", "r") as csvfile:
    pauli_weight_result = csv.DictReader(csvfile)
    mappers = ("JW", "BK", "BTT", "Tree")

    with open(f"circuit-complexity/{casename}.csv", "r") as circ_file:
        circuit_complexity_result = csv.DictReader(circ_file)

        improve_jw = []
        improve_bk = []
        improve_btt = []

        for row_pauli, row_circ in zip(pauli_weight_result, circuit_complexity_result):
            # Handle pauli weight
            pauli_weights = []
            cnot_counts = []
            circ_depth = []
            for mapper in mappers:
                pauli_weights.append(toint(row_pauli[mapper]))
                cnot_counts.append(tovalue(row_circ[mapper], "cx"))
                circ_depth.append(tovalue(row_circ[mapper], "depth"))

            improve_jw.append(
                (
                    improve(pauli_weights[-1], pauli_weights[0]),
                    improve(cnot_counts[-1], cnot_counts[0]),
                    improve(circ_depth[-1], circ_depth[0]),
                )
            )

            improve_bk.append(
                (
                    improve(pauli_weights[-1], pauli_weights[1]),
                    improve(cnot_counts[-1], cnot_counts[1]),
                    improve(circ_depth[-1], circ_depth[1]),
                )
            )

            improve_btt.append(
                (
                    improve(pauli_weights[-1], pauli_weights[2]),
                    improve(cnot_counts[-1], cnot_counts[2]),
                    improve(circ_depth[-1], circ_depth[2]),
                )
            )

def slice(a, index):
    return [i[index] for i in a]

print("improve to JW: ", mean(slice(improve_jw, 0)), mean(slice(improve_jw, 1)), mean(slice(improve_jw, 2)))
print("improve to BK: ", mean(slice(improve_bk, 0)), mean(slice(improve_bk, 1)), mean(slice(improve_bk, 2)))
print("improve to BTT: ", mean(slice(improve_btt, 0)), mean(slice(improve_btt, 1)), mean(slice(improve_btt, 2)))

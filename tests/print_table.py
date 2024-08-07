import csv
import sys

if len(sys.argv) != 2:
    print("Help: print_table.py {casename}")
    sys.exit(1)

casename = sys.argv[1]


def tovalue(record: str, field: str):
    if record == "--":
        return SUPERBIG
    s = record.split("/")
    return int(s[0]) if field == "cx" else int(s[2])


def format_title(title: str):
    return ",".join(map(lambda x: f"\\textbf{{\\textsf{{{x}}}}}", title.split(",")))

SUPERBIG = 9999999999999999999

def resultstr(result: list[int]):
    minvalue = min(result)
    return ",".join(map(lambda x: f"\\textbf{{{x}}}" if x == minvalue else ("--" if x == SUPERBIG else str(x)), result))


lines = []

def to_int(a : str):
    if a == "--":
        return SUPERBIG
    else:
        return int(a)

with open(f"pauli-weight/{casename}.csv", "r") as csvfile:
    pauli_weight_result = csv.DictReader(csvfile)
    if "FH" in pauli_weight_result.fieldnames:  # type: ignore
        title = format_title("JW,BK,BTT,FH,HATT")
        mappers = ("JW", "BK", "BTT", "FH", "Tree")
    else:
        title = format_title("JW,BK,BTT,HATT")
        mappers = ("JW", "BK", "BTT", "Tree")

    with open(f"circuit-complexity/{casename}.csv", "r") as circ_file:
        circuit_complexity_result = csv.DictReader(circ_file)

        # format title
        lines.append(
            "\\textbf{Case},\\textbf{Modes},\\textbf{Pauli Weight},"
            + "," * (len(mappers) - 1)
            + "\\textbf{$CNOT$ Gate Count},"
            + "," * (len(mappers) - 1)
            + "\\textbf{Circuit Depth},"
            + "," * (len(mappers) - 1)
        )

        lines.append(f",,{title},{title},{title}")

        for row_pauli, row_circ in zip(pauli_weight_result, circuit_complexity_result):
            # Handle pauli weight
            pauli_weights = []
            cnot_counts = []
            circ_depth = []
            for mapper in mappers:
                pauli_weights.append(to_int(row_pauli[mapper]))
                cnot_counts.append(tovalue(row_circ[mapper], "cx"))
                circ_depth.append(tovalue(row_circ[mapper], "depth"))

            lines.append(f"{row_pauli['Geometry']},{row_pauli['Modes']},{resultstr(pauli_weights)},{resultstr(cnot_counts)},{resultstr(circ_depth)}")


for line in lines:
    print(line)

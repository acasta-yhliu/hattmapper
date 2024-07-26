import csv
from typing import Callable
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# plt.style.use("classic")
plt.rc("font", size=20, family="serif", serif="cmr10")
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["mathtext.fontset"] = "cm"

TIMES = "\\times"

physics = "Fermi-Hubbard Model"


def fh_format(row: dict[str, str]):
    return f"{row['Modes']}\n(${row['Geometry'].replace('x', TIMES)}$)"


def format_mapper(mapper: str):
    if mapper == "Tree":
        mapper = "HATT"
    return mapper


pw_records = []
filename = f"pauli-weight/syk.csv"

with open(filename, "r") as csvfile:
    pauli_weight_result = csv.DictReader(csvfile)
    for row in pauli_weight_result:
        for mapper in ("JW", "BK", "BTT", "FH", "Tree"):
            if mapper in row and row[mapper] != "--":
                cr = int(row[mapper])
                pw_records.append(
                    {
                        "case": fh_format(row),
                        "name": format_mapper(mapper),
                        "weight": cr,
                    }
                )

pw_records = pd.DataFrame(pw_records)

plt.subplots(3, 1, figsize=(20, 12), layout="constrained")

plt.subplot(3, 1, 1)
plt.title(f"(a) Pauli Weight")
sns.barplot(pw_records, x="case", y="weight", hue="name")
plt.grid(True, axis="y")
plt.xticks([])
plt.xlabel("")
plt.ylabel("Pauli Weight")
plt.legend(prop={"family": "sans serif"})


circ_records = []

filename = f"circuit-complexity/syk.csv"


def extract_record(record: str):
    if record == "--":
        return float("inf"), float("inf")
    s = record.split("/")
    return int(s[0]), int(s[2])


with open(filename, "r") as csvfile:
    pauli_weight_result = csv.DictReader(csvfile)
    for row in pauli_weight_result:
        for mapper in ("JW", "BK", "BTT", "FH", "Tree"):
            if mapper in row and row[mapper] != "--":
                cr = extract_record(row[mapper])
                circ_records.append(
                    {
                        "case": fh_format(row),
                        "name": format_mapper(mapper),
                        "cx": cr[0],
                        "depth": cr[1],
                    }
                )

circ_records = pd.DataFrame(circ_records)

plt.subplot(3, 1, 2)
plt.title(f"(b) Circuit Complexity - $CNOT$ Gate Count")
sns.barplot(circ_records, x="case", y="cx", hue="name")
plt.grid(True, axis="y")
plt.xlabel("")
plt.xticks([])
plt.ylabel("$CNOT$ Gate Count")
plt.legend(prop={"family": "sans serif"})

plt.subplot(3, 1, 3)
plt.title("(c) Circuit Complexity - Circuit Depth")
sns.barplot(circ_records, x="case", y="depth", hue="name")
plt.grid(True, axis="y")
plt.xlabel("Modes (Geometry)")
plt.ylabel("Circuit Depth")
plt.legend(prop={"family": "sans serif"})
plt.savefig(f"./syk.pdf")


def summary_file(
    filename: str, extractor=lambda x: float("inf") if x == "--" else int(x)
):
    with open(filename, "r") as csvfile:
        pauli_weight_result = csv.DictReader(csvfile)
        for row in pauli_weight_result:
            opt_tree = extractor(row["Tree"])

            others = [
                extractor(row[i])
                for i in pauli_weight_result.fieldnames
                if i in ("JW", "BK", "BTT", "FH")
            ]

            best = min(others)

            print((best - opt_tree) * 100 / best)

            # print(opt_tree, others, min(others))


def extract_circuit(record: str, i: int):
    if record == "--":
        return [float("inf"), float("inf")][i]
    s = record.split("/")
    return [int(s[0]), int(s[2])][i]


summary_file("circuit-complexity/syk.csv", lambda x: extract_circuit(x, 0))
summary_file("circuit-complexity/syk.csv", lambda x: extract_circuit(x, 1))
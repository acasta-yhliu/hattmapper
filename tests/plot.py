import csv
from typing import Callable
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def fh_format(row: dict[str, str]):
    return f"{row['Modes']}\n({row['Geometry']})"


def mol_format(row: dict[str, str]):
    return f"{row['Modes']}\n(${row['Geometry']}$)"


def plot_pauli_weight(
    name: str,
    physics: str,
    formatter: Callable[[dict[str, str]], str],
    xlabel: str,
    yscale: str = "linear",
):
    cases = []
    tree_weight = []
    jw_weight = []
    bk_weight = []
    btt_weight = []
    fh_weight = []

    filename = f"pauli-weight/{name}.csv"
    with open(filename, "r") as csvfile:
        pauli_weight_result = csv.DictReader(csvfile)
        for row in pauli_weight_result:
            cases.append(formatter(row))
            tree_weight.append(int(row["Tree"]))
            jw_weight.append(int(row["JW"]))
            bk_weight.append(int(row["BK"]))
            btt_weight.append(int(row["BTT"]))

            if "FH" in row and row["FH"] != "--":
                fh_weight.append(int(row["FH"]))

    plt.clf()
    plt.figure(figsize=(12, 4))
    plt.title(f"Pauli Weight ({physics})")
    plt.plot(cases, tree_weight, label="Tree")
    plt.plot(cases, jw_weight, label="JW")
    plt.plot(cases, bk_weight, label="BK")
    plt.plot(cases, btt_weight, label="BTT")
    if len(fh_weight) != 0:
        plt.plot(cases[: len(fh_weight)], fh_weight, label="FH")
    plt.xlabel(xlabel)
    plt.ylabel("Pauli Weight")
    plt.yscale(yscale)
    plt.legend()
    plt.savefig(f"pauli-weight/{name}.pdf")


def plot_circuit_complexity(
    name: str,
    physics: str,
    formatter: Callable[[dict[str, str]], str],
    xlabel: str,
    yscale: str = "linear",
):
    records = []

    filename = f"circuit-complexity/{name}.csv"

    def extract_record(record: str):
        s = record.split("/")
        return int(s[0]), int(s[2])


    with open(filename, "r") as csvfile:
        pauli_weight_result = csv.DictReader(csvfile)
        for row in pauli_weight_result:
            for mapper in ("JW", "BK", "BTT", "FH", "Tree"):
                if mapper in row and row[mapper] != "--":
                    cr = extract_record(row[mapper])
                    records.append(
                        {
                            "case": formatter(row),
                            "name": mapper,
                            "cx": cr[0],
                            "depth": cr[1],
                        }
                    )
    
    records = pd.DataFrame(records)

    plt.clf()
    plt.title(f"Circuit Complexity ({physics})")
    sns.barplot(records, x="case", y="cx", hue="name")
    plt.xlabel(xlabel)
    plt.ylabel("CX Gate Counts")
    plt.yscale(yscale)
    plt.legend()
    plt.savefig(f"circuit-complexity/{name}-cx.pdf")

    plt.clf()
    plt.title(f"Circuit Complexity ({physics})")
    sns.barplot(records, x="case", y="depth", hue="name")
    plt.xlabel(xlabel)
    plt.ylabel("Circuit Depth")
    plt.yscale(yscale)
    plt.legend()
    plt.savefig(f"circuit-complexity/{name}-depth.pdf")


plot_pauli_weight("fermihubbard", "Fermi-Hubbard Model", fh_format, "Modes (Geometry)")
plot_circuit_complexity(
    "fermihubbard", "Fermi-Hubbard Model", fh_format, "Modes (Geometry)"
)

plot_pauli_weight(
    "molecule",
    "Electron Structure Problem",
    mol_format,
    "Modes (Molecule)",
    yscale="log",
)
plot_circuit_complexity(
    "molecule",
    "Electron Structure Problem",
    mol_format,
    "Modes (Molecule)",
    yscale="log",
)

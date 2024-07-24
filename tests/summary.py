import csv

filename = f"circuit-complexity/molecule.csv"


def tovalue(record: str, field: str):
    if record == "--":
        return "--"
    s = record.split("/")
    return s[0] if field == "cx" else s[2]

def format_title(title: str):
    return ",".join(map(lambda x: f"\\textbf{{\\textsf{{{x}}}}}", title.split(",")))

print(format_title("Molecule,Modes,JW,BK,BTT,FH,OptTree"))
with open(filename, "r") as csvfile:
    pauli_weight_result = csv.DictReader(csvfile)
    for row in pauli_weight_result:
        values = []
        for mapper in ("JW", "BK", "BTT", "FH", "Tree"):
            values.append(tovalue(row[mapper], "depth"))
        print(f"${row['Geometry']}$,{row['Modes']},{','.join(values)}")

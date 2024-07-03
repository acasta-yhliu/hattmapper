from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver

from argparse import ArgumentParser
from treemapper import evaluate
from moleculeFromJson import loadMolecule

parser = ArgumentParser()
parser.add_argument(
    "-f",
    "--format",
    choices=["default", "csv", "txt"],
    default="default",
    type=str,
    help="report format of the evaluation result",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=None,
    help="output file for the evaluation result, default is stdout",
)
parser.add_argument(
    "-b",
    "--basis-gates",
    type=str,
    default="cx,rx,ry,rz",
    help="comma-split basis gates, default is %(default)s",
)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.output != None:
        open(args.output, "w").close()
    open("out.txt", "w").close()
    LiHTest = evaluate(
        "LiH",
        PySCFDriver(
            atom="H 0 0 0; Li 0 0 1.6",
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        ).run(),
        basis_gates=args.basis_gates.split(","),
    )
    LiHTest.report(args.format, output=args.output)
    LiHTest.report("txt", "out.txt")
    
    H2OTest = evaluate(
        "H2O",
        PySCFDriver(
            atom='\
            O 0.0 0.0 0.0; \
            H 0.757 0.586 0.0; \
            H -0.757 0.586 0.0\
            ',
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        ).run(),
        basis_gates=args.basis_gates.split(","),
    )
    H2OTest.report(args.format, output=args.output)
    H2OTest.report("txt", "out.txt")
    
    CH4Test = evaluate(
        "Methane",
        PySCFDriver(
            atom=loadMolecule("tests/methane.json"),
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        ).run(),
        basis_gates=args.basis_gates.split(","),
    )
    CH4Test.report(args.format, output=args.output)
    CH4Test.report("txt", "out.txt")
    
    N2Test = evaluate(
        "N2",
        PySCFDriver(
            atom=loadMolecule('tests/n2.json'),
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        ).run(),
        basis_gates=args.basis_gates.split(","),
    )
    N2Test.report(args.format, output=args.output)
    N2Test.report("txt", "out.txt")
    
    COTest = evaluate(
        "Carbon Monoxide",
        PySCFDriver(
            atom=loadMolecule("tests/carbonmonoxide.json"),
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        ).run(),
        basis_gates=args.basis_gates.split(","),
    )
    COTest.report(args.format, output=args.output)
    COTest.report("txt", "out.txt")
    
    O2Test = evaluate(
        "O2",
        PySCFDriver(
            atom='\
            O 0.616 0.0 0.0; \
            O -0.616 0.0 0.0\
            ',
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        ).run(),
        basis_gates=args.basis_gates.split(","),
    )
    O2Test.report(args.format, output=args.output)
    O2Test.report("txt", "out.txt")
    
    EthaneTest = evaluate(
        "Ethane",
        PySCFDriver(
            atom=loadMolecule("tests/ethane.json"),
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        ).run(),
        basis_gates=args.basis_gates.split(","),
    )
    EthaneTest.report(args.format, output=args.output)
    EthaneTest.report("txt", "out.txt")
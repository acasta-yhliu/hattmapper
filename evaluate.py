from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver

from argparse import ArgumentParser
from treemapper import evaluate

parser = ArgumentParser()
parser.add_argument(
    "-f",
    "--format",
    choices=["default", "csv"],
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

    evaluate(
        "LiH",
        PySCFDriver(
            atom="H 0 0 0; Li 0 0 1.6",
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        ).run(),
        basis_gates=args.basis_gates.split(","),
    ).report(args.format, output=args.output)
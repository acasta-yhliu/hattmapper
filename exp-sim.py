from typing import Literal
from utility import load_molecule, pauli_weight, Simulation, FermihedralMapper
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from hatt_pairing_mapper import HATTPairingMapper
from hatt_naive_mapper import TernaryTreeMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qiskit_nature.second_q.transformers import FreezeCoreTransformer

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("classic")

from tqdm import tqdm
import sys

molecule = sys.argv[1]

if molecule == "H2":
    hamiltonian: FermionicOp = (
        PySCFDriver(
            atom="H 0 0 0; H 0 0 0.735",
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        )
        .run()
        .hamiltonian.second_q_op()
    )
else:
    freeze_list = [0]
    remove_list = [-3, -2]

    problem = PySCFDriver(
        atom="H 0 0 0; Li 0 0 1.6",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    ).run()

    hamiltonian: FermionicOp = (
        FreezeCoreTransformer(freeze_core=True, remove_orbitals=[-3, 3, 2, -2])
        .transform(problem)
        .hamiltonian.second_q_op()
    )

# print(hamiltonian.register_length)

ttmapper = HATTPairingMapper(hamiltonian)

ttsim = Simulation(hamiltonian, ttmapper)
jwsim = Simulation(hamiltonian, JordanWignerMapper())
bksim = Simulation(hamiltonian, BravyiKitaevMapper())
bttsim = Simulation(hamiltonian, TernaryTreeMapper())
fhsim = Simulation(hamiltonian, FermihedralMapper.molecule())


class Result:
    def __init__(self, x: int, y: int, base: float) -> None:
        self.bias = np.zeros((x, y))
        self.var = np.zeros((x, y))
        self.base = base

    def __setitem__(self, pos, value):
        e, var = value
        self.bias[pos] = e - self.base
        self.var[pos] = var

    def random(self):
        self.bias = np.random.random(self.bias.shape)
        self.var = np.random.random(self.var.shape)

    @property
    def max_bias(self) -> float:
        return self.bias.max()

    @property
    def min_bias(self) -> float:
        return self.bias.min()

    @property
    def max_var(self) -> float:
        return self.var.max()

    @property
    def min_var(self) -> float:
        return self.var.min()


err_1qs = np.arange(0.0001, 0.001, 0.00005)
err_2qs = np.arange(0.001, 0.01, 0.0005)

tt_result = Result(len(err_1qs), len(err_2qs), ttsim.ground_energy)
jw_result = Result(len(err_1qs), len(err_2qs), jwsim.ground_energy)
bk_result = Result(len(err_1qs), len(err_2qs), bksim.ground_energy)
btt_result = Result(len(err_1qs), len(err_2qs), bttsim.ground_energy)
fh_result = Result(len(err_1qs), len(err_2qs), fhsim.ground_energy)

for i in tqdm(range(len(err_1qs))):
    for j in range(len(err_2qs)):

        tt_result[i, j] = ttsim.simulate(
            depolarerr_1q=err_1qs[i], depolarerr_2q=err_2qs[j], quiet=True
        )
        jw_result[i, j] = jwsim.simulate(
            depolarerr_1q=err_1qs[i], depolarerr_2q=err_2qs[j], quiet=True
        )
        bk_result[i, j] = bksim.simulate(
            depolarerr_1q=err_1qs[i], depolarerr_2q=err_2qs[j], quiet=True
        )
        btt_result[i, j] = bttsim.simulate(
            depolarerr_1q=err_1qs[i], depolarerr_2q=err_2qs[j], quiet=True
        )
        fh_result[i, j] = fhsim.simulate(
            depolarerr_1q=err_1qs[i], depolarerr_2q=err_2qs[j], quiet=True
        )

# Do the plotting


def plot(vorb: Literal["var", "bias"]):
    name = "Bias" if vorb == "bias" else "Variance"

    print(f"Plotting {name.lower()}:")

    if vorb == "bias":
        maxvalue = max(tt_result.max_bias, jw_result.max_bias, bk_result.max_bias)
        minvalue = min(tt_result.min_bias, jw_result.min_bias, bk_result.min_bias)
    else:
        maxvalue = max(tt_result.max_var, jw_result.max_var, bk_result.max_var)
        minvalue = min(tt_result.min_var, jw_result.min_var, bk_result.min_var)

    print(f"  Max {name} = {maxvalue}")
    print(f"  Min {name} = {minvalue}")

    plt.clf()
    fig, ax = plt.subplots(1, 5, figsize=(12, 4), layout="constrained")

    xticks = [0, 5, 10, 15]
    yticks = [0, 5, 10, 15]

    plt.subplot(1, 5, 1)
    plt.imshow(
        getattr(jw_result, vorb), interpolation="nearest", vmax=maxvalue, vmin=minvalue
    )
    plt.xticks([])
    plt.yticks([])
    # plt.xticks(xticks, [f"{i*1000:.2f}" for i in err_1qs[xticks]], fontsize=16)
    # plt.xlabel("1Q Gate Error Rate ($\\times10^{-4}$)", fontsize=16)
    # plt.yticks(yticks, [f"{i*1000:.2f}" for i in err_2qs[yticks]], fontsize=16)
    # plt.ylabel("2Q Gate Error Rate ($\\times10^{-4}$)", fontsize=16)

    plt.subplot(1, 5, 2)
    plt.imshow(
        getattr(bk_result, vorb), interpolation="nearest", vmax=maxvalue, vmin=minvalue
    )
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 5, 3)
    im = plt.imshow(
        getattr(btt_result, vorb), interpolation="nearest", vmax=maxvalue, vmin=minvalue
    )
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 5, 4)
    im = plt.imshow(
        getattr(fh_result, vorb), interpolation="nearest", vmax=maxvalue, vmin=minvalue
    )
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 5, 5)
    im = plt.imshow(
        getattr(tt_result, vorb), interpolation="nearest", vmax=maxvalue, vmin=minvalue
    )
    plt.xticks([])
    plt.yticks([])

    # cax = fig.add_axes([ax[-1].get_position().x1-0.25,ax[-1].get_position().y0,0.02,ax[-1].get_position().y1-ax[-1].get_position().y0])
    # divider = make_axes_locatable(ax[-1])
    # cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, fraction=0.05, pad=0.04, aspect=20)

    # for t in cbar.ax.get_yticklabels():
    #     t.set_fontsize(16)

    plt.savefig(f"tests/simulation/{molecule}-{name.lower()}.pdf")


plot("bias")
plot("var")


# tt_result.plot("$H_2$", "Tree", "tests/sim-H2-tree.pdf")

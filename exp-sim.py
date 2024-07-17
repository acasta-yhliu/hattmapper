from typing import Literal
from utility import load_molecule, pauli_weight, Simulation
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from ternary_bonsai_mapper import HamiltonianTernaryBonsaiMapper
from ternary_tree_mapper import TernaryTreeMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("classic")

from tqdm import tqdm
import sys


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

ttmapper = HamiltonianTernaryBonsaiMapper(hamiltonian)

ttsim = Simulation(hamiltonian, ttmapper)
jwsim = Simulation(hamiltonian, JordanWignerMapper())
bksim = Simulation(hamiltonian, BravyiKitaevMapper())
bttsim = Simulation(hamiltonian, TernaryTreeMapper())


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
    fig, axes = plt.subplots(1, 4, figsize=(12, 4), layout="constrained")
    plt.suptitle("$H_2$" + f", {name}", fontsize=24)

    xticks = [0, 5, 10, 15]
    yticks = [0, 5, 10, 15]

    plt.subplot(1, 4, 1)
    plt.title("JW", fontsize=22, y=1.05)
    plt.imshow(getattr(jw_result, vorb), interpolation="nearest", vmax=maxvalue, vmin=minvalue)
    plt.xticks(xticks, [f"{i*1000:.2f}" for i in err_1qs[xticks]], fontsize=16)
    plt.xlabel("1Q Gate Error Rate ($\\times10^{-4}$)", fontsize=16)
    plt.yticks(yticks, [f"{i*1000:.2f}" for i in err_2qs[yticks]], fontsize=16)
    plt.ylabel("2Q Gate Error Rate ($\\times10^{-4}$)", fontsize=16)

    plt.subplot(1, 4, 2)
    plt.title("BK", fontsize=22, y=1.05)
    plt.imshow(getattr(bk_result, vorb), interpolation="nearest", vmax=maxvalue, vmin=minvalue)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 4, 3)
    plt.title("BTT", fontsize=22, y=1.05)
    im = plt.imshow(
        getattr(btt_result, vorb), interpolation="nearest", vmax=maxvalue, vmin=minvalue
    )
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 4, 4)
    plt.title("Tree", fontsize=22, y=1.05)
    im = plt.imshow(
        getattr(tt_result, vorb), interpolation="nearest", vmax=maxvalue, vmin=minvalue
    )
    plt.xticks([])
    plt.yticks([])

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.66, aspect=15*0.65)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(16)

    plt.savefig(f"tests/sim-{name.lower()}.pdf")

plot("bias")
plot("var")


# tt_result.plot("$H_2$", "Tree", "tests/sim-H2-tree.pdf")

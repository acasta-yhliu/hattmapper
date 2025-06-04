# HATT: Hamiltonian Aware Ternary Tree for Optimizing Fermion-to-Qubit Mapping

## Requirements

Before install dependencies, you need:

* Linux 
* Python 3.9+
* virtual environment (`venv`)
* We are using a pre-release version of `qiskit_nature`

Create the virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Then install necessary Python packages:

```bash
pip3 install -r requirements.txt
```

## Usage

We contain these mappers:

1. `HATTMapper`: The Fermion-to-qubit mapper with _vacuum state preservation_ and _optimization_.

2. `HATTPairingMapper`: The mapper with only _vacuum state preservation_ based on operator pairing but no travserse optimization.

3. `HATTNaiveMapper`: The most basic one. No _vacuum state preservation_.

You could compare these mappers according to our paper. However, the **`HATTMapper`** is the one you should use.

## Citation

To cite our paper:

```bibtex
@INPROCEEDINGS{liu2025hatt,
  author={Liu, Yuhao and Yao, Kevin and Hong, Jonathan and Froustey, Julien and Rrapaj, Ermal and Iancull, Costin and Li, Gushu and Shi, Yunong},
  booktitle={2025 IEEE International Symposium on High Performance Computer Architecture (HPCA)}, 
  title={HATT: Hamiltonian Adaptive Ternary Tree for Optimizing Fermion-to-Qubit Mapping}, 
  year={2025},
  volume={},
  number={},
  pages={143-157},
  keywords={Resistance;Quantum system;Scalability;Qubit;Noise;Polynomials;Complexity theory;Optimization;Vacuum arcs;Quantum simulation},
  doi={10.1109/HPCA61900.2025.00022}}

```

# HATT: Hamiltonian Aware Ternary Tree for Optimizing Fermion-to-Qubit Mapping

## Requirements

Before install dependencies, you need:

* Linux 
* Python 3.9+
* virtual environment
* We are using a pre-release version of `qiskit_nature`

Then install necessary Python packages:

```bash
pip3 install -r requirements.txt
```

## Usage

We contain tree mappers:

1. `HamiltonianTernaryTreeMapper`: The most basic tree construction algorithm, without _vacuum state preservation_ or any optimization.

2. `HamiltonianTernaryBonsaiMapper`: Tree construction algorithm with _vacuum state preservation_ but use traverse for operator pairing.

3. `HamiltonianTernaryUnionMapper`: The most optimized version, use caching for fast traversing and retain _vacuum state preservation_. **This is the version you should use**. Others are for legacy purpose.

## Citation

To cite our paper:

```bibtex

```
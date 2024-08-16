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

```

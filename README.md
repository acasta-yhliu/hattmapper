# HATT: Hamiltonian Adaptive Ternary Tree for Optimizing Fermion-to-Qubit Mapping

Artifact-evaluation guide and checklist.

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

## Experiments

To run the experiments, simply follow the checklist and run the corresponding notebook/code to gain the result.

* Table I, Figure 10, Figure 12, Table IV, Table V: `ae-molecule.ipynb`
* Table II, Table III, Table VI: `ae-lattice.ipynb`
* Figure 11: `ae-forte.ipynb`
  **Note**: Running on IonQ Forte-1 requires _reservation_ through _Amazon Braket_ and credits.

## Usage

To integrate `HATTMapper` into your code, use:

```python
from hattmapper import HATTMapper

mapper = HATTMapper(hamiltonian)
```
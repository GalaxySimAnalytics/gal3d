### Introduction

Gal3D is a Python library for constructing and analyzing three-dimensional galaxy morphology models. It is designed for building 3D models from simulated particle data and for studying the intrinsic structure of galaxies in simulations.

### Documentation

The documentation is at [readthedocs](https://gal3d.readthedocs.io)


### Installation

Clone the repository and install in editable (-e) mode:

```bash
git clone https://github.com/GalaxySimAnalytics/gal3d.git
cd gal3d
pip install -e .
```

gal3d depends on the following libraries:

- **numpy** (numerical computations)
- **scipy** (scientific computing)
- **cython** (performance acceleration)
- **matplotlib** (visualization)
- **tqdm** (progress bars)
- **h5py** (HDF5 file support)

Optional (for advanced optimization):

- **lmfit** (recommended)
- **nlopt**
- **optimagic**

### Usage

```python
from gal3d.analyzer import Gal3DAnalyzer

analyzer = Gal3DAnalyzer.analyze(pos,mass)
model = analyzer.fit()
```



### Development Setup

Prerequisites: [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

then run the following commands to set up the development environment:

**Linux / macOS**
```bash
git clone https://github.com/GalaxySimAnalytics/gal3d.git
cd gal3d
make setup
```
**Windows** (PowerShell, no `make`):
```powershell
git clone https://github.com/GalaxySimAnalytics/gal3d.git
cd gal3d
uv sync --extra dev --extra tests --extra optimizer
uv run pre-commit install
```

### License

[MIT License](./LICENSE)
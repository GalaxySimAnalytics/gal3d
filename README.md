## Introduction
Gal3D is a Python library for constructing and analyzing three-dimensional galaxy or halo morphology models. It is designed for building 3D models from simulated particle data, similar to ellipse fitting techniques used in observations.

## Installation

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

## Usage

```python
from gal3d.analyzer import Gal3DAnalyzer

analyzer = Gal3DAnalyzer.analyze(pos,mass)
model = analyzer.fit()
```

See [gal3d_example](https://github.com/GalaxySimAnalytics/gal3d_example) for more usage examples, or refer to the documentation for details.

## License

[MIT License](./LICENSE)
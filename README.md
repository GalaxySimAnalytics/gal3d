## Introduction
Gal3D is a Python library for constructing and analyzing three-dimensional galaxy morphology models. It is specifically designed for building 3D models from simulated particle data, analogous to ellipse fitting techniques used in observation.

## Installation

```
git clone https://github.com/GalaxySimAnalytics/gal3d.git
```
Install this in editable mode.
```
cd gal3d
pip install -e .
```
gal3d depends on the following libraries:

- **numpy** (numerical computations)
- **scipy** (scientific computing)
- **cython** (default numerical acceleration)
- **matplotlib** (visualization)
- **tqdm** (progress bars)

optional:

- **lmfit** (numerical optimization algorithms, recommended)
- **nlopt** (numerical optimization algorithms)
- **optimagic** (numerical optimization algorithms)

## Usage

```python
from gal3d.analyzer import Gal3DAnalyzer

analyzer = Gal3DAnalyzer.analyze(pos,mass)

model = analyzer.fit()
```

See [gal3d_example](https://github.com/GalaxySimAnalytics/gal3d_example) for usage examples，or refer to the documentation for more details.
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

- **numba** (JIT acceleration)
- **nlopt** (numerical optimization algorithms)
- **optimagic** (numerical optimization algorithms)
- **emcee** (MCMC optimization)



## Model Fitting and Optimization
This section describes the core functionality of the gal3d package, including the fitting process, shape functions, and error functions used for optimization.


#### Fitting Equal Density Surfaces

The package fits the equal density surface from the inside out using the given shape functions (currently implemented: **Ellipsoid** and **Ellipsoid_S**). These functions describe the geometry of the surface and are optimized to match the simulated galaxy data.

The fitting process accounts for changes in the coordinate system, including:
* **The center position**: $P_c=[x_c,y_c,z_c]$
* **The Euler angles**: $R_{zyx}(\gamma,\beta,\alpha)=R_z(\gamma)R_y(\beta)R_x(\alpha)$


#### Shape Functions

* Ellipsoid:

$$f = (\frac{x}{a})^2+(\frac{y}{b})^2+(\frac{z}{c})^2,\ \ \ \ \ \ \ \ c \leq b \leq a$$

* Ellipsoid_S:

$$[(\frac{x}{a})^2]^{S_a}+[(\frac{y}{b})^2]^{S_b}+[(\frac{z}{c})^2]^{S_c}=1$$

where
$$0.2\leq S_a,S_b,S_c \leq 2,\ \ \ c \leq b \leq a$$
The shape of the equal density surface is characterized by 
  * Ellipticity:
    *  $\epsilon_{ab} = 1-\frac{b}{a}$
    *  $\epsilon_{bc} = 1-\frac{c}{b}$
  * Shape Indices:
    *  $S_a,S_b,S_c$


#### Error Functions

1. Normal Error Funciton:
    The standard error function minimizes the average squared deviation of the fitted surface from the target surface:
    $$f_{error} = (\sum_{i=1}^{n}(f_i-1)^2)/n ,$$
    where:
    * $f_i$ is the value of the shape function for the $i$-th point.
    * $n$ is the total number of points.
2. Weighted Error Function
   A weighted error function allows for specific weights $w_i$ to be assigned to each point:
    $$f_{error} = (\sum_{i=1}^{n}(w_i\times(f_i-1)^2))/n$$

3. Scaled Error Function
   A scaled error function incorporates radial distance $r_i$ as a weighting factor, similar to area weighting:
    $$f_{error} = (\sum_{i=1}^{n}(r_i^2\times(f_i-1)^2))/n$$

#### Optimization Process
The fitting process involves minimizing the chosen error function using optimization algorithms provided by **nlopt**, **scipy.optimize**, or **optimagic**. The optimization adjusts the parameters of the shape function (e.g., $a,\epsilon_{ab},\epsilon_{bc},S_a,S_b,S_c$, center position, and Euler angles) to achieve the best fit.

## Code Structure and Usage Examples

### Core Components
The package is organized into several modules:

1. **point**: Handles particle data and density estimation
2. **field**: Manages spherical field generation and interpolation 
3. **shape**: Defines geometric models and structure calculation
4. **optimization**: Provides parameter and optimization utilities
5. **analyzer**: Coordinates the fitting process

### Basic Usage Example

```python
import numpy as np
from gal3d.point import Particles
from gal3d.field import SphField
from gal3d.shape import Structure3D
from gal3d.optimization.optimizer import Optimizer
from gal3d.analyzer import Gal3DAnalyzer

# Load particle data (positions and masses)
positions = np.load('galaxy_positions.npy')  # Shape: (N, 3)
masses = np.load('galaxy_masses.npy')        # Shape: (N,)

# Create particle container with density estimation
particles = Particles(positions, masses)

# Create spherical field
field = SphField(particles, num_ray=1024)
field.build_field_boundary(inner=0.6, inner_mode='dist', outer=5e4, outer_mode='value')
field.build_profile_sample()
field.build_profile_interpolator()
field.build_isodensity_profile()

# Create structure model - Ellipsoid with EulerShift coordinates
structure = Structure3D(
    coordinate='EulerShift',
    geometry='Ellipsoid',
    error_func='sums_dev',
    error_method='isodensity_fcall'
)

# Create optimizer
optimizer = Optimizer.get_plugin('OptimizerScipy')(algorithm='Powell')

# Create analyzer and perform fitting
analyzer = Gal3DAnalyzer(particles, field, structure, optimizer)

# Fit at specific radius
result_single = analyzer.fit(5.0)

# Fit over a range of radii
radii = np.geomspace(1.0, 10.0, 20)  # 20 log-spaced radii from 1 to 10
result_multiple = analyzer.fit(radii)

# Extract results
print(f"Semi-major axes: {result_multiple['a']}")
print(f"Axis ratios b/a: {result_multiple['b']/result_multiple['a']}")
print(f"Axis ratios c/a: {result_multiple['c']/result_multiple['a']}")
```

### Configuration-based Setup

```python
import numpy as np
from gal3d.analyzer import Gal3DAnalyzer

# Load particle data
positions = np.load('galaxy_positions.npy')
masses = np.load('galaxy_masses.npy')

# Create analyzer from config file
analyzer = Gal3DAnalyzer.from_config(positions, masses, 'config.ini')

# Fit multiple radii
result = analyzer.fit(np.geomspace(1.0, 10.0, 20))
```

Example config.ini:
```ini
[Point]
r_max = 100.0
k_nearest = 32
r_cut = 10.0

[Field]
n_ray = 1024
ray_method = fibonacci
inner = 0.6
inner_mode = dist
outer = 50000
outer_mode = value
n_step = 500
step_mode = log
interpolator = SG
isodensity_method = both
from_rays_func = True
res_b = 0.2
res_c = 0.1
iso_step = 300
isodensity_interpolator = SG

[Shape]
coordinate = EulerShift
geometry = Ellipsoid
error_func = sums_dev
error_method = isodensity_fcall

[Optimizer]
optimizer = OptimizerScipy
algorithm = Powell
```

## Advanced Features

### Custom Error Functions

You can define and register custom error functions:

```python
from gal3d.shape.minimize_func import MinimizeFunc
import numpy as np

@MinimizeFunc.minimize_fn_registry
def custom_weighted_error(f_call, weights):
    """
    Custom weighted error function.
    
    Parameters
    ----------
    f_call : numpy.ndarray
        Function values at sample points
    weights : numpy.ndarray
        Weights for each point
        
    Returns
    -------
    float
        Weighted error value
    """
    return np.sum(weights * f_call**2) / len(f_call)
```

### Visualization Examples

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate 2D projection for visualization
X, Y = result_single[0].generate_edge2D()

# Plot the shape
plt.figure(figsize=(8, 8))
plt.plot(X, Y, 'b-')
plt.axis('equal')
plt.title(f'Isodensity Contour at a={result_single["a"][0]:.2f}')
plt.xlabel('x (kpc)')
plt.ylabel('y (kpc)')
plt.grid(True, alpha=0.3)
plt.show()

# Generate 3D visualization
from mpl_toolkits.mplot3d import Axes3D
X, Y, Z = result_single[0].generate_edge3D()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.3, color='b')
ax.set_title(f'3D Isodensity Surface at a={result_single["a"][0]:.2f}')
plt.tight_layout()
plt.show()
```
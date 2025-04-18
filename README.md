## Introduction
This Python package is designed to build an isodense ellipsoid model for simulated galaxies from the inside out. Similar to the ellipse fitting used in observations.

## Installation

```
git clone https://github.com/wx-ys/gal3d.git
```
Install this in editable mode.
```
cd gal3d
pip install -e .
```

The gal3d packages relies on the following Python packages:

* numpy
* scipy
* numba
* nlopt
* optimagic
* tqdm



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
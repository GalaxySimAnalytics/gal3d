## Introduction
This python package is aimed to build an isodense ellipsoid model for the simulated galaxy from the inside out. Similar to the ellipse fitting used in observations.

## Installation

```
git clone https://github.com/wx-ys/gal3d.git
```
Install this in editable mode.
```
cd gal3d
pip install -e .
```

gal3d uses the following python packages:

* numba
* numpy
* scipy
* tqdm
* nlopt
* optimagic

## Fitting
Fit an equal density surface from the inside out, by

$$[(\frac{x}{a})^2]^{S_a}+[(\frac{y}{b})^2]^{S_b}+[(\frac{z}{c})^2]^{S_c}=1,\ \ \ \ \  0.2\leq S_a,S_b,S_c \leq 2,\ \ \ c \leq b \leq a$$

and also changes in the coordinate system, the center position and Euler angle:

$$P_c=[x_c,y_c,z_c]$$
$$R_{zyx}(\gamma,\beta,\alpha)=R_z(\gamma)R_y(\beta)R_x(\alpha)$$

The shape of the equal density surface is measured by ellipticity ($\epsilon_{ab},\epsilon_{bc}$) and shape index ($S_a,S_b,S_c$), where $\epsilon_{ab} = 1-\frac{b}{a}$ and $\epsilon_{bc} = 1-\frac{c}{b}$




## Shape func

* Ellipsoid:
$$f = (\frac{x}{a})^2+(\frac{y}{b})^2+(\frac{z}{c})^2,\ \ \ \ \ \ \ \ c \leq b \leq a$$

* Ellipsoid_S:

$$f = [(\frac{x}{a})^2]^{S_a}+[(\frac{y}{b})^2]^{S_b}+[(\frac{z}{c})^2]^{S_c},\ \ \ \ \  0.2\leq S_a,S_b,S_c \leq 2,\ \ \ c \leq b \leq a$$


## Error func

$f_{error} = (\sum_{i=1}^{n}(f_i-1)^2)/n$

$f_{error} = (\sum_{i=1}^{n}(w_i\times(f_i-1)^2))/n$

$f_{error} = (\sum_{i=1}^{n}(r_i^2\times(f_i-1)^2))/n$


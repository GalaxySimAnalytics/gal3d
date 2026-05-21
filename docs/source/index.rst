Welcome to the Gal3D documentation! `Gal3D <https://github.com/GalaxySimAnalytics/gal3d>`_ is a Python library for three-dimensional morphological modeling and analysis of simulated galaxies.
It is designed to quantify the 3D morphological properties of mass distributions from particle data and to help reveal how galaxy structure and morphology evolve in three dimensions.



Installation
============

Clone the repository and install the package:

.. code-block:: bash

   git clone https://github.com/GalaxySimAnalytics/gal3d.git
   cd gal3d
   pip install .

For an editable installation, use:

.. code-block:: bash

   pip install -e .



Scientific Motivation
=====================

The intrinsic three-dimensional shapes of galaxies encode important information about their formation and evolution. 
In simulations, particle data provide direct access to the full 3D mass distribution, but robustly measuring radial variations in shape remains challenging.

`Gal3D` is designed to address this problem. It combines density-field reconstruction, radial sampling, and superellipsoid fitting to model the 3D iso-density 
structure of galaxies and to quantify quantities such as axis ratios, orientations, center offsets, and higher-order shape features.


Citation
========

If you use Gal3D in your research or project, please cite the following reference.
Formal publication information will be added once the corresponding paper is published.

::

    @software{gal3d_2026,
      author = {Shuai Lu and Min Du},
      title = {Gal3D},
      year = {2026},
      url = {https://github.com/GalaxySimAnalytics/gal3d}
    }



Documentation Contents
======================

This documentation is organized into three main parts:

- :ref:`Tutorials <tutorials>`: basic usage, key features, and advanced examples.
- :ref:`Implementation Details <details>`: technical descriptions of the underlying methods and modules.
- :ref:`Reference <reference>`: the full API reference.

.. toctree::
   :maxdepth: 2

   Tutorials <tutorials/index>
   Implementation Details <details/index>
   Reference <reference/index>
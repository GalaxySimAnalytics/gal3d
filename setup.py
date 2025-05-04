
from os import path
import sys

from setuptools import setup, Extension, find_packages
from Cython.Build import build_ext,cythonize
import numpy
from setuptools import Extension, setup




extensions = cythonize([
    Extension(
        name="gal3d.shape.geometry_plugins._ellipsoid_util_cy",
        sources=["src/gal3d/shape/geometry_plugins/_ellipsoid_util_cy.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        #extra_compile_args=['-fopenmp'],
        #extra_link_args=['-fopenmp'],
    )
],
    compiler_directives={'language_level': "3", "boundscheck": False, "wraparound": False},
)

install_requires=[
        "scipy>=1.15.2,<2.0.0",
        "numba>=0.61.0,<0.62.0",
        "optimagic>=0.5.1,<0.6.0",
        "nlopt>=2.9.1,<3.0.0",
        "tqdm>=4.67.1,<5.0.0",
        "matplotlib>=3.10.1,<4.0.0",
        "numpy>=2,<3",
        "h5py>=3.13.0,<4.0.0",
    ]

tests_require = [
    'pytest','pytest-cov'
]

extras_require = {
    'tests': tests_require,
}

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as snap:
    long_description = snap.read()



setup(
    name="gal3d",
    version="1.0.0",
    description="Build galaxy 3D morphology model",
    author="Shuai LU",
    author_email="lushuai@stu.xmu.edu.cn",
    
    package_dir={"": "src"},
    
    packages = ['gal3d',
                'gal3d.characterization',
                'gal3d.characterization.characterizer_plugins',
                'gal3d.field',
                'gal3d.field.grid',
                'gal3d.field.spherical_field',
                'gal3d.field.spherical_field.ray',
                'gal3d.optimization',
                'gal3d.optimization.optimizer_plugins',
                'gal3d.point',
                'gal3d.point.density_estimator_plugins',
                'gal3d.shape',
                'gal3d.shape.coordinate_plugins',
                'gal3d.shape.geometry_plugins',
                'gal3d.util',
                'gal3d.visualization',
                'gal3d.visualization.model_projector_plugins'],

    package_data={"": ["*.pxd", "*.pyx"],"src/gal3d":["plugins.json","default_config.ini","run_config.ini"]},
    include_package_data=True,
    install_requires=install_requires,
    
    ext_modules = extensions,
    
    cmdclass={'build_ext': build_ext},
    
    tests_require=tests_require,
    extras_require=extras_require,
    python_requires=">=3.10,<3.13",
    entry_points={
        "console_scripts": ["gal3d = gal3d.executor:main"]
    },
    long_description=long_description,
    long_description_content_type='text/markdown'
      )

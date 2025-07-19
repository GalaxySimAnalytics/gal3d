from os import path
from setuptools import setup, Extension
from Cython.Build import build_ext, cythonize
import numpy

openmp_args = ["-fopenmp"]
optimization_flags = ['-O3', '-march=native', '-ffast-math']
extra_compile_args = openmp_args + optimization_flags
extra_link_args = openmp_args + ['-std=c++14'] + optimization_flags

extensions = cythonize([
    Extension(
        name="gal3d.shape.geometry_plugins.ellipsoid_s_cy",
        sources=["src/gal3d/shape/geometry_plugins/ellipsoid_s_cy.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    Extension(
        name="gal3d.shape.geometry_plugins.ellipsoid_cy",
        sources=["src/gal3d/shape/geometry_plugins/ellipsoid_cy.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    Extension(
        name="gal3d.shape.fns_cy",
        sources=["src/gal3d/shape/fns_cy.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    Extension(
        name="gal3d.util.array_operate_cy",
        sources=["src/gal3d/util/array_operate_cy.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    Extension(
        "gal3d.field.spherical_field.util_cy",
        ["src/gal3d/field/spherical_field/util_cy.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    Extension(
        "gal3d.point.util_cy",
        ["src/gal3d/point/util_cy.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    Extension(
        name="gal3d.point.density_estimator_plugins.compute_pa_cy",
        sources=["src/gal3d/point/density_estimator_plugins/compute_pa_cy.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
],
    compiler_directives={'language_level': "3",
                         'boundscheck': False,
                         'wraparound': False,
                         'cdivision': True,
                         'initializedcheck': False,
                         'nonecheck': False,
    }
)

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
    packages=[
        'gal3d',
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
        'gal3d.visualization.model_projector_plugins'
    ],
    package_data={"": ["*.pxd", "*.pyx"], "src/gal3d": ["plugins.json", "default_config.ini", "run_config.ini"]},
    include_package_data=True,
    ext_modules=extensions,
    cmdclass={'build_ext': build_ext},
    long_description=long_description,
    long_description_content_type='text/markdown'
)
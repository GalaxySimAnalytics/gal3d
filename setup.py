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
        sources=["src/gal3d/shape/geometry_plugins/ellipsoid_s_cy.pyx", "src/gal3d/shape/geometry_plugins/ellipsoid_s.c"],
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
        include_dirs=[numpy.get_include(), "src/gal3d/point/density_estimator_plugins"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    Extension(
        name="gal3d.field.spherical_field.ray.lu_mono_cy",
        sources=[
            "src/gal3d/field/spherical_field/ray/lu_mono_cy.pyx",
            "src/gal3d/field/spherical_field/ray/pchip.cpp"
        ],
        include_dirs=[
            numpy.get_include(),
            "src/gal3d/field/spherical_field/ray"
        ],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        name="gal3d.visualization.render_wrapper",
        sources=[
            "src/gal3d/visualization/render_wrapper.pyx",
            "src/gal3d/visualization/render.cpp"
        ],
        include_dirs=[
            numpy.get_include(),
            "src/gal3d/visualization"
        ],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
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

setup(
    ext_modules=extensions,
    cmdclass={'build_ext': build_ext},
)
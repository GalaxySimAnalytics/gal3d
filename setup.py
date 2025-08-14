import platform
import subprocess

import numpy
from Cython.Build import build_ext, cythonize
from setuptools import Extension, setup


def is_macos():
    return platform.system() == 'Darwin'

def is_windows():
    return platform.system() == 'Windows'

def get_xcode_version():
    result = subprocess.run(['pkgutil', '--pkg-info=com.apple.pkg.CLTools_Executables'], capture_output=True, text=True)
    try:
        version_line = result.stdout.split('\n')[1]
        version = version_line.split(' ')[1]
    except IndexError:
        return 0 # looks like xcode-cltools not installed? try to proceed anyway
    return version

def xcode_fix_needed():
    if is_macos() and int(get_xcode_version().split('.')[0]) >= 15:
        return True
    else:
        return False


# Platform-specific compiler settings
if is_windows():
    # MSVC compiler flags
    openmp_args = ['/openmp']
    extra_compile_args = ['/O2', '/std:c++14']
    extra_link_args = ['/openmp']
else:
    # GCC/Clang compiler flags
    openmp_args = ['-fopenmp']
    
    extra_link_args = openmp_args + ['-std=c++14']


if xcode_fix_needed():
    # workaround for XCode bug FB13097713
    # https://developer.apple.com/documentation/xcode-release-notes/xcode-15-release-notes#Linking
    extra_link_args += ['-Wl,-ld_classic']


def with_cpp(compiler_flags):
    if is_windows():
        return compiler_flags + ['/std:c++14']
    else:
        return compiler_flags + ['-std=c++14']


#optimization_flags = ['-O3','-march=native', '-ffast-math'] # '-march=native'
#define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
#extra_link_args = extra_link_args + optimization_flags

incdir = [numpy.get_include()]


ellipsoid_s = Extension(
    name="gal3d.shape.geometry_plugins.ellipsoid_s_cy",
    sources=["src/gal3d/shape/geometry_plugins/ellipsoid_s_cy.pyx", "src/gal3d/shape/geometry_plugins/ellipsoid_s.c"],
    include_dirs=incdir,
    extra_compile_args=openmp_args,
    extra_link_args=extra_link_args,
)


ellipsoid = Extension(
    name="gal3d.shape.geometry_plugins.ellipsoid_cy",
    sources=["src/gal3d/shape/geometry_plugins/ellipsoid_cy.pyx"],
    include_dirs=incdir,
    extra_compile_args=openmp_args,
    extra_link_args=extra_link_args,
)


fns_error = Extension(
    name="gal3d.shape.fns_cy",
    sources=["src/gal3d/shape/fns_cy.pyx"],
    include_dirs=incdir,
    extra_compile_args=openmp_args,
    extra_link_args=extra_link_args,
)

array_operate = Extension(
    name="gal3d.util.array_operate_cy",
    sources=["src/gal3d/util/array_operate_cy.pyx"],
    include_dirs=incdir,
    extra_compile_args=openmp_args,
    extra_link_args=extra_link_args,
)


spherical_field_util = Extension(
    "gal3d.field.spherical_field.util_cy",
    ["src/gal3d/field/spherical_field/util_cy.pyx"],
    include_dirs=incdir,
    extra_compile_args=openmp_args,
    extra_link_args=extra_link_args,
)


point_util = Extension(
    "gal3d.point.util_cy",
    ["src/gal3d/point/util_cy.pyx"],
    include_dirs=incdir,
    extra_compile_args=openmp_args,
    extra_link_args=extra_link_args,
)

density_estimate = Extension(
    name="gal3d.point.density_estimator_plugins.compute_pa_cy",
    sources=["src/gal3d/point/density_estimator_plugins/compute_pa_cy.pyx"],
    include_dirs=incdir + ["src/gal3d/point/density_estimator_plugins"],
    extra_compile_args=openmp_args,
    extra_link_args=extra_link_args,
)


monotonic_interpolate = Extension(
    name="gal3d.field.spherical_field.ray.lu_mono_cy",
    sources=[
        "src/gal3d/field/spherical_field/ray/lu_mono_cy.pyx",
        "src/gal3d/field/spherical_field/ray/pchip.cpp"
    ],
    include_dirs=incdir + ["src/gal3d/field/spherical_field/ray"],
    extra_compile_args=openmp_args,
    extra_link_args=extra_link_args,
    language="c++",
)


sph_render = Extension(
    name="gal3d.visualization.render_wrapper",
    sources=[
        "src/gal3d/visualization/render_wrapper.pyx",
        "src/gal3d/visualization/render.cpp"
    ],
    include_dirs=incdir + ["src/gal3d/visualization"],
    extra_compile_args=openmp_args,
    extra_link_args=extra_link_args,
   # language="c++",
)

extensions =[
    ellipsoid_s,
    ellipsoid,
    fns_error,
    array_operate,
    spherical_field_util,
    point_util,
    density_estimate,
    monotonic_interpolate,
    sph_render,
]

setup(
    ext_modules=extensions,
    cmdclass={'build_ext': build_ext},
)
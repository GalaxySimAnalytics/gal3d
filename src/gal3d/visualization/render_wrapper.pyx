# distutils: language = c++
cimport numpy as np
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.string cimport memcpy

import numpy as np

from gal3d import config
from .model_projector import ImageData




cdef extern from "render.hpp":
    cdef cppclass Grid "Grid<double>":
        double xmin, ymin, xmax, ymax
        int nx, ny
        Grid(double, double, double, double, int, int)
        void add_qty(double, double, double)

    cdef cppclass CubicSplineSmoothingKernel "CubicSplineSmoothingKernel<double>":
        CubicSplineSmoothingKernel()
        double operator()(double) const
        double density(double) const
        double columnDensity(double) const

    cdef cppclass RenderImage "RenderImage<double>":
        Grid image_grid
        RenderImage(double, double, double, double, int, int,
                    CubicSplineSmoothingKernel&, int, int, int)
        void add_particle(double, double, double, double)
        void add_particle(vector[double]&, vector[double]&, vector[double]&, vector[double]&)
        int circle_vs_canvas(double, double, double) const

        vector[vector[double]] get_values() const
        const vector[double]& get_flat_values() const

cdef extern from "render.hpp":

    cdef cppclass GridFloat "Grid<float>":
        float xmin, ymin, xmax, ymax
        int nx, ny
        GridFloat(float, float, float, float, int, int)
        void add_qty(float, float, float)
    cdef cppclass CubicSplineSmoothingKernelFloat "CubicSplineSmoothingKernel<float>":
        CubicSplineSmoothingKernelFloat()
        float operator()(float) const
        float density(float) const
        float columnDensity(float) const

    cdef cppclass RenderImageFloat "RenderImage<float>":
        GridFloat image_grid
        RenderImageFloat(float, float, float, float, int, int,
                    CubicSplineSmoothingKernelFloat&, int, int, int)
        void add_particle(float, float, float, float)
        void add_particle(vector[float]&, vector[float]&, vector[float]&, vector[float]&)
        int circle_vs_canvas(float, float, float) const

        vector[vector[float]] get_values() const
        const vector[float]& get_flat_values() const



cdef class PyCubicSplineSmoothingKernel:
    cdef CubicSplineSmoothingKernel* cpp_kernel

    def __cinit__(self):
        self.cpp_kernel = new CubicSplineSmoothingKernel()

    def __dealloc__(self):
        del self.cpp_kernel

    def __call__(self, double r):
        return self.cpp_kernel[0](r)

    def density(self, double r):
        return self.cpp_kernel.density(r)

    def columnDensity(self, double R):
        return self.cpp_kernel.columnDensity(R)

cdef class PyCubicSplineSmoothingKernelFloat:
    cdef CubicSplineSmoothingKernelFloat* cpp_kernel

    def __cinit__(self):
        self.cpp_kernel = new CubicSplineSmoothingKernelFloat()

    def __dealloc__(self):
        del self.cpp_kernel

    def __call__(self, float r):
        return self.cpp_kernel[0](r)

    def density(self, float r):
        return self.cpp_kernel.density(r)

    def columnDensity(self, float R):
        return self.cpp_kernel.columnDensity(R)


# PERF P5: pointer-based copy — no Python-level loop
cdef vector[double] from_numpy_to_vector(np.ndarray[np.float64_t, ndim=1] arr):
    cdef Py_ssize_t n = arr.shape[0]
    cdef vector[double] v
    v.resize(n)
    if n > 0:
        memcpy(<void*>v.data(), &arr[0], n * sizeof(double))
    return v

cdef vector[float] from_numpy_to_vector_float(np.ndarray[np.float32_t, ndim=1] arr):
    cdef Py_ssize_t n = arr.shape[0]
    cdef vector[float] v
    v.resize(n)
    if n > 0:
        memcpy(<void*>v.data(), &arr[0], n * sizeof(float))
    return v

cdef class PyRenderImage:
    cdef RenderImage* cpp_image

    def __cinit__(self, double x_min, double x_max, double y_min, double y_max, int nx, int ny,
                  PyCubicSplineSmoothingKernel kernel, int subsample_nx, int subsample_ny, int numthreads = 1):
        self.cpp_image = new RenderImage(x_min, x_max, y_min, y_max, nx, ny,
                                         kernel.cpp_kernel[0], subsample_nx, subsample_ny, numthreads)

    def __dealloc__(self):
        del self.cpp_image

    def add_particle(self, x, y, mass, hsml):

        cdef vector[double] vx, vy, vm, vh

        if (isinstance(x, (float, int)) and isinstance(y, (float, int)) and
        isinstance(mass, (float, int)) and isinstance(hsml, (float, int))):
            self.cpp_image.add_particle(
                <double>x, <double>y, <double>mass, <double>hsml
            )
        else:
            x_arr = np.ascontiguousarray(np.asarray(x).ravel(), dtype=np.float64)
            y_arr = np.ascontiguousarray(np.asarray(y).ravel(), dtype=np.float64)
            mass_arr = np.ascontiguousarray(np.asarray(mass).ravel(), dtype=np.float64)
            hsml_arr = np.ascontiguousarray(np.asarray(hsml).ravel(), dtype=np.float64)

            vx = from_numpy_to_vector(x_arr)
            vy = from_numpy_to_vector(y_arr)
            vm = from_numpy_to_vector(mass_arr)
            vh = from_numpy_to_vector(hsml_arr)
            self.cpp_image.add_particle(vx, vy, vm, vh)


    def circle_vs_canvas(self, double x, double y, double hsml):
        return self.cpp_image.circle_vs_canvas(x, y, hsml)

    def get_image(self):
        """
        Return
        ------
        np.ndarray:
            image_grid.qty
        """
        cdef int ny = self.cpp_image.image_grid.ny
        cdef int nx = self.cpp_image.image_grid.nx
        cdef vector[double] flat = self.cpp_image.get_flat_values()
        cdef np.ndarray[np.float64_t, ndim=2] arr = np.empty((ny, nx), dtype=np.float64)
        memcpy(<void*>arr.data, flat.data(), ny * nx * sizeof(double))

        x_range = (self.cpp_image.image_grid.xmin, self.cpp_image.image_grid.xmax)
        y_range = (self.cpp_image.image_grid.ymin, self.cpp_image.image_grid.ymax)
        xs = np.linspace(x_range[0], x_range[1], nx + 1)
        ys = np.linspace(y_range[0], y_range[1], ny + 1)
        xs = 0.5 * (xs[:-1] + xs[1:])
        ys = 0.5 * (ys[:-1] + ys[1:])
        return ImageData(arr, xs, ys, tuple(x_range), tuple(y_range))


cdef class PyRenderImageFloat:
    cdef RenderImageFloat* cpp_image

    def __cinit__(self, float x_min, float x_max, float y_min, float y_max, int nx, int ny,
                  PyCubicSplineSmoothingKernelFloat kernel, int subsample_nx, int subsample_ny, int numthreads = 1):
        self.cpp_image = new RenderImageFloat(x_min, x_max, y_min, y_max, nx, ny,
                                              kernel.cpp_kernel[0], subsample_nx, subsample_ny, numthreads)

    def __dealloc__(self):
        del self.cpp_image

    def add_particle(self, x, y, mass, hsml):

        cdef vector[float] vx, vy, vm, vh

        if (isinstance(x, (float, int)) and isinstance(y, (float, int)) and
            isinstance(mass, (float, int)) and isinstance(hsml, (float, int))):
            self.cpp_image.add_particle(
                <float>x, <float>y, <float>mass, <float>hsml
            )
        else:
            x_arr    = np.ascontiguousarray(np.asarray(x).ravel(),    dtype=np.float32)
            y_arr    = np.ascontiguousarray(np.asarray(y).ravel(),    dtype=np.float32)
            mass_arr = np.ascontiguousarray(np.asarray(mass).ravel(), dtype=np.float32)
            hsml_arr = np.ascontiguousarray(np.asarray(hsml).ravel(), dtype=np.float32)

            vx = from_numpy_to_vector_float(x_arr)
            vy = from_numpy_to_vector_float(y_arr)
            vm = from_numpy_to_vector_float(mass_arr)
            vh = from_numpy_to_vector_float(hsml_arr)

            self.cpp_image.add_particle(vx, vy, vm, vh)

    def circle_vs_canvas(self, float x, float y, float hsml):
        return self.cpp_image.circle_vs_canvas(x, y, hsml)

    def get_image(self):
        cdef int ny = self.cpp_image.image_grid.ny
        cdef int nx = self.cpp_image.image_grid.nx
        cdef vector[float] flat = self.cpp_image.get_flat_values()
        cdef np.ndarray[np.float32_t, ndim=2] arr = np.empty((ny, nx), dtype=np.float32)
        memcpy(<void*>arr.data, flat.data(), ny * nx * sizeof(float))

        x_range = (self.cpp_image.image_grid.xmin, self.cpp_image.image_grid.xmax)
        y_range = (self.cpp_image.image_grid.ymin, self.cpp_image.image_grid.ymax)
        xs = np.linspace(x_range[0], x_range[1], nx + 1)
        ys = np.linspace(y_range[0], y_range[1], ny + 1)
        xs = 0.5 * (xs[:-1] + xs[1:])
        ys = 0.5 * (ys[:-1] + ys[1:])
        return ImageData(arr, xs, ys, tuple(x_range), tuple(y_range))


def get_kernel():
    if config.sph_render.render_double:
        return PyCubicSplineSmoothingKernel()
    else:
        return PyCubicSplineSmoothingKernelFloat()

def get_render_image(x_min, x_max, y_min, y_max, nx, ny, kernel, subsample_nx, subsample_ny, 
                    numthreads = config.general.number_of_threads):
    """
    Get a render image object for the specified parameters.

    Parameters
    ----------
    x_min, x_max, y_min, y_max : float
        Canvas extent.
    nx, ny : int
        Pixel resolution.
    kernel : PyCubicSplineSmoothingKernel or PyCubicSplineSmoothingKernelFloat
        Smoothing kernel.
    subsample_nx, subsample_ny : int
        Subsampling resolution.
    numthreads : int
        OpenMP thread count.
    """
    if config.sph_render.render_double:
        return PyRenderImage(x_min, x_max, y_min, y_max, nx, ny, kernel, subsample_nx, subsample_ny, numthreads)
    else:
        return PyRenderImageFloat(x_min, x_max, y_min, y_max, nx, ny, kernel, subsample_nx, subsample_ny, numthreads)
# distutils: language = c++
from libcpp.vector cimport vector
from libcpp cimport bool
cimport numpy as np
import numpy as np

cdef extern from "render.hpp":
    cdef cppclass Grid:
        Grid(double, double, double, double, int, int)
        void add_qty(double, double, double)

    cdef cppclass CubicSplineSmoothingKernel:
        CubicSplineSmoothingKernel()
        double operator()(double) const
        double density(double) const
        double columnDensity(double) const

    cdef cppclass RenderImage:
        RenderImage(double, double, double, double, int, int,
                    CubicSplineSmoothingKernel&, int, int)
        void add_particle(double, double, double, double)
        void add_particle(vector[double]&, vector[double]&, vector[double]&, vector[double]&)
        int circle_vs_canvas(double, double, double) const

        vector[vector[double]] get_values() const



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


cdef vector[double] from_numpy_to_vector(np.ndarray[np.float64_t] arr):
    cdef vector[double] v
    cdef Py_ssize_t i, n = arr.shape[0]
    v.reserve(n)
    for i in range(n):
        v.push_back(arr[i])
    return v

cdef np.ndarray matrix_to_numpy(vector[vector[double]] mat):
    cdef Py_ssize_t ny = mat.size()
    if ny == 0:
        return np.empty((0, 0), dtype=np.float64)
    cdef Py_ssize_t nx = mat[0].size()
    arr = np.empty((ny, nx), dtype=np.float64)
    for j in range(ny):
        for i in range(nx):
            arr[j, i] = mat[j][i]
    return arr

cdef class PyRenderImage:
    cdef RenderImage* cpp_image

    def __cinit__(self, double x_min, double x_max, double y_min, double y_max, int nx, int ny,
                  PyCubicSplineSmoothingKernel kernel, int subsample_nx, int subsample_ny):
        self.cpp_image = new RenderImage(x_min, x_max, y_min, y_max, nx, ny,
                                         kernel.cpp_kernel[0], subsample_nx, subsample_ny)

    def __dealloc__(self):
        del self.cpp_image

    def add_particle(self, x, y, mass, hsml):

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
        返回 image_grid.qty 作为 numpy 数组
        """
        return matrix_to_numpy(self.cpp_image.get_values())

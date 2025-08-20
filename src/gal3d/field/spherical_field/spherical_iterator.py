import numpy as np

from gal3d.shape import StructureCore

from .spherical_vector import SphVector


class SphericalDecIterator:

    def __init__(self, n_sample):
        self.sample_area = SphVector(n_sample=n_sample)
        self.structure = StructureCore("EulerShift","Ellipsoid_S")
        self.parameters = self.structure.parameters
        self.history = []



    def spherical_harmonics_dec(self, pos: np.ndarray, w: np.ndarray, lmax: int = 4) -> None:
        # Compute the spherical harmonics for the given positions and weights
        c0 = 0.5/np.sqrt(np.pi)
        coef: dict[int, np.ndarray] = {}
        self.coef = coef
        r = np.sqrt(np.sum(pos*pos,axis=1))
        x = pos[:,0]
        y = pos[:,1]
        z = pos[:,2]

        coef[0] = np.array(c0*np.sum(w))
        if lmax == 0:
            return

        coef[1] = np.sqrt(3)*c0*np.sum(pos.T/r*w,axis=1)   # x, y, z
        if lmax ==1:
            return
        coef[2] = np.sqrt(15)*c0*np.sum(w*np.array([(x*x-y*y), 2*z*z-x*x-y*y, x*y,y*z,z*x])/(r*r),axis=1) # eps_ab, eps_bc, xy, yz, zx
        coef[2][:2] = coef[2][:2]/2

        if lmax == 2:
            return


    def update_a(self, parameters, f_d):

        delta = self.coef[0]/f_d[0]
        if self.coef[2][0]>0:
            delta += self.coef[2][0]/f_d[1]
        if self.coef[2][1]<0:
            delta -= self.coef[2][1]/f_d[2]
        return delta

    def update_center_x(self, parameters, f_d):
        delta = self.coef[1][0]/f_d[0]
        return delta

    def update_center_y(self, parameters, f_d):
        delta = self.coef[1][1]/f_d[1]
        return delta

    def update_center_z(self, parameters, f_d):
        delta = self.coef[1][2]/f_d[2]
        return delta

    def update_center(self, parameters, f_d):
        delta = self.coef[1]/f_d
        return delta

    def update_eps_ab(self, parameters, f_d, fixed_f_d = False):
        if fixed_f_d:
            delta = self.coef[2][0]/f_d[1]/parameters["a"]
            return delta
        else:
            delta = 2*(np.exp(self.coef[2][0]/((1-parameters["eps_ab"])*parameters["a"])/f_d[1])-1) * (1-parameters["eps_ab"])
            return delta

    def update_eps_bc(self, parameters, f_d, fixed_f_d = False):
        if fixed_f_d:
            delta = -self.coef[2][1]/f_d[2]/(parameters["a"]*(1-parameters["eps_ab"]))
            return delta
        else:
            delta = 2*(np.exp(-self.coef[2][1]/((1-parameters["eps_bc"])*(parameters["a"]*(1-parameters["eps_ab"])))/f_d[2])-1) * (1-parameters["eps_bc"])
            return delta


    def update_sample(self, pos, assign_ind=True):
        error = self.structure.f_ray_d(pos)-1

        if assign_ind:
            aligned_pos = self.structure.transform_pos(pos)
            index = self.sample_area.assign_points(aligned_pos)

            ind: list[list[int]] = [[] for _ in range(self.sample_area.num)]
            for i, j in enumerate(index):
                ind[j].append(i)

            self._ind = ind


        sample_w = np.nan*np.ones(self.sample_area.num)
        for i in range(len(self._ind)):
            if self._ind[i]:
                sample_w[i] = np.mean(error[self._ind[i]])

        sample_pos = self.sample_area.pos[~np.isnan(sample_w)]
        sample_w = sample_w[~np.isnan(sample_w)]

        self.sample_pos = sample_pos
        self.sample_w = sample_w


    def update_gradient(self ):
        self.gradient = (self.structure.f_ray_d(500.*np.eye(3)) - self.structure.f_ray_d(499.*np.eye(3)))*len(self.sample_w)

    def update_spherical_harmonics(self, lmax=4):
        self.spherical_harmonics_dec(self.sample_pos,self.sample_w,lmax=lmax)


    def compute_error(self):
        return self.coef[0]+np.sum(np.abs(self.coef[1]))+np.sum(np.abs(self.coef[2]))

    def fit(self, pos, max_step = 500, per_iter = 20, dec_iter = 5, ftol = 0.01, **kwargs):
        rtol = kwargs.get("rtol", 0.1)
        atol = kwargs.get("atol", 0.1)
        epstol = kwargs.get("epstol", 0.001)

        self.update_sample(pos) # if center_pos, angle changed
        self.update_gradient()  # if eps , sa changed

        step = 0
        while step < max_step:

            for _ in range(per_iter):
                self.update_spherical_harmonics(lmax=2) # always
                delta = self.update_center(self.parameters,self.gradient)
                self.parameters["x"],self.parameters["y"],self.parameters["z"] = self.parameters["pos"] + delta
                self.update_sample(pos) # if center_pos, angle changed
                self.update_gradient()
                self.history.append(self.parameters.copy())
                step += 1
                if np.linalg.norm(delta) < rtol:
                    break

            for _ in range(per_iter):
                self.update_spherical_harmonics(lmax=2) # always
                delta = self.update_a(self.parameters,self.gradient)
                if (self.parameters["a"] + delta < 0):
                    break
                self.parameters["a"] = self.parameters["a"] + delta
                self.update_sample(pos, assign_ind=True) # if center_pos, angle changed
                self.update_gradient()
                self.history.append(self.parameters.copy())
                step += 1
                if abs(delta) < atol:
                    break

            for _ in range(per_iter):
                self.update_spherical_harmonics(lmax=2) # always

                delta_eps_ab = self.update_eps_ab(self.parameters,self.gradient)
                if (self.parameters["eps_ab"] + delta_eps_ab) > 1 or (self.parameters["eps_ab"] + delta_eps_ab) < 0:
                    break
                self.parameters["eps_ab"] = self.parameters["eps_ab"] + delta_eps_ab

                delta_eps_bc = self.update_eps_bc(self.parameters,self.gradient)
                if (self.parameters["eps_bc"] + delta_eps_bc) > 1 or (self.parameters["eps_bc"] + delta_eps_bc) < 0:
                    break
                self.parameters["eps_bc"] = self.parameters["eps_bc"] + delta_eps_bc

                self.update_sample(pos, assign_ind=False) # if center_pos, angle changed, assign_ind=True
                self.update_gradient()
                self.history.append(self.parameters.copy())
                step += 1
                if (delta_eps_ab < epstol) and (delta_eps_bc < epstol):
                    break

            if self.compute_error() < ftol:
                break

            per_iter = per_iter-dec_iter
            per_iter = max(per_iter,dec_iter)

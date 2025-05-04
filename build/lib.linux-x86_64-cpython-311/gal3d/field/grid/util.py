from numba import (
    int32,
    deferred_type,
    optional,
    float64,
    boolean,
    int64,
    njit,
    jit,
    prange,
    types,
)

import numpy as np


@njit(
    types.Tuple((float64[:, :], float64[:, :]))(float64[:], float64[:]),
    cache=True,
)
def split_byequalvolumn(posmin, posmax):
    # half_boxsize = (posmax - posmin)/2
    cent_pos = (posmin + posmax) / 2.0
    lower_pos = posmin * np.ones((8, 3))
    upper_pos = posmax * np.ones((8, 3))
    offest = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
    )
    for i in prange(8):
        for j in prange(3):
            if offest[i][j] > 0:
                lower_pos[i][j] = cent_pos[j]
            else:
                upper_pos[i][j] = cent_pos[j]
    return lower_pos, upper_pos


@njit(
    types.Tuple((float64[:, :], float64[:, :]))(float64[:], float64[:], float64[:, :]),
    parallel=True,
    fastmath=True,
    cache=True,
)
def split_by_median_point(posmin, posmax, pos):
    # np.median(gal.grid.base_pos[gal.grid.base_indice==2],axis=0)
    sele = np.sum(pos < posmax, axis=1) + np.sum(pos >= posmin, axis=1)
    samplepos = pos[sele == 6]
    cent_pos = np.zeros(3)
    for i in prange(3):
        cent_pos[i] = np.median(samplepos[:, i])
    lower_pos = posmin * np.ones((8, 3))
    upper_pos = posmax * np.ones((8, 3))
    offest = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
    )
    for i in prange(8):
        for j in prange(3):
            if offest[i][j] > 0:
                lower_pos[i][j] = cent_pos[j]
            else:
                upper_pos[i][j] = cent_pos[j]
    return lower_pos, upper_pos


@njit(
    int64(float64[:], float64[:], float64[:, :], int64[:], int64, int64),
    parallel=True,
    nogil=True,
    fastmath=True,
    cache=True,
)
def set_partical_num(posmin, posmax, pos, indices, parent_ind, child_ind):
    num = 0
    for i in prange(len(pos)):
        if indices[i] == parent_ind:
            if (pos[i] >= posmin).all() and (pos[i] < posmax).all():
                indices[i] = child_ind
                num += 1
    return num


@njit(
    int64(float64[:], float64[:], float64[:, :], int64[:], int64),
    parallel=True,
    nogil=True,
    fastmath=True,
    cache=True,
)
def test_partical_num(posmin, posmax, pos, indices, parent_ind):
    num = 0
    for i in prange(len(pos)):
        if indices[i] == parent_ind:
            if (pos[i] >= posmin).all() and (pos[i] < posmax).all():
                num += 1
    return num


@njit(int64(int64[:], int64), parallel=True, nogil=True, fastmath=True)
def read_partical_num(indices, tar_ind):
    num = 0
    for i in prange(len(indices)):
        if indices[i] == tar_ind:
            num += 1
    return num


@njit(
    types.Tuple((float64[:], float64[:], float64[:]))(
        float64[:, :], float64[:, :], float64[:], int64[:]
    ),
    parallel=True,
    nogil=True,
    fastmath=True,
    cache=True,
)
def cal_volumn_density(posmin, posmax, mass, indice):
    volumn = np.ones(len(posmax), dtype=np.float64)
    masses = np.zeros(len(posmax), dtype=np.float64)
    density = np.zeros(len(posmax), dtype=np.float64)

    for j in prange(len(indice)):
        masses[indice[j]] = masses[indice[j]] + mass[j]

    for i in prange(len(posmax)):
        for j in prange(3):
            volumn[i] = volumn[i] * (posmax[i][j] - posmin[i][j])
        density[i] = masses[i] / volumn[i]

    return volumn, masses, density


@njit(
    int64[:](float64[:, :], float64[:, :], float64[:, :], int64[:], int64, int64[:]),
    parallel=True,
    nogil=True,
    fastmath=True,
)
def set_partical_nums(posmin, posmax, pos, indices, parent_ind, child_ind):
    num = np.zeros(len(child_ind), dtype=np.int64)

    for i in prange(len(pos)):
        if indices[i] == parent_ind:
            for j in range(len(num)):
                if (pos[i] >= posmin[j]).all() and (pos[i] < posmax[j]).all():
                    indices[i] = child_ind[j]
                    num[j] += 1
                    break
    return num


@njit(
    types.Tuple((float64[:, :], float64[:, :], int32[:], int64[:], int64[:]))(
        float64[:, :], int32, int32, float64[:, :], float64[:, :]
    )
)
def make_grid_by_num(pos, maxdepth, cut_max_partnum, lower_pos, upper_pos):
    Nums = np.zeros(1, dtype=np.int64)

    Depth = np.zeros(1, dtype=np.int32)  # 8 ** 0 = 1

    Indice = np.zeros(len(pos), dtype=np.int64)

    i = 0

    while True:
        if i > (len(Depth) - 1):
            break
        Nums[i] = read_partical_num(Indice, i)

        # if Depth[i] >= maxdepth:
        #    i = i + 1
        #    continue
        while (Nums[i] > cut_max_partnum) and (Depth[i] < maxdepth):

            add_lower, add_upper = split_byequalvolumn(
                lower_pos[i], upper_pos[i]
            )  # 7 new

            Nums = np.concatenate((Nums, np.zeros(7, dtype=np.int64)))
            child_inds = np.zeros(8, dtype=np.int64)
            for grid_ind in range(len(add_lower)):
                if grid_ind == 0:
                    child_inds[grid_ind] = i
                else:
                    child_inds[grid_ind] = len(Depth) + grid_ind - 1

            nums_add = set_partical_nums(
                posmin=add_lower,
                posmax=add_upper,
                pos=pos,
                indices=Indice,
                parent_ind=i,
                child_ind=child_inds,
            )

            for grid_ind in range(len(add_lower)):
                if grid_ind == 0:
                    Nums[i] = nums_add[grid_ind]
                    continue
                else:
                    Nums[len(Depth) + grid_ind - 1] = nums_add[grid_ind]

            lower_pos = np.concatenate((lower_pos, add_lower[1:]))
            upper_pos = np.concatenate((upper_pos, add_upper[1:]))

            Depth[i] = Depth[i] + 1
            Depth = np.concatenate((Depth, np.ones(7, dtype=np.int32) * Depth[i]))

            lower_pos[i] = add_lower[0]
            upper_pos[i] = add_upper[0]

            Nums[i] = read_partical_num(Indice, i)

        i = 1 + i

    return lower_pos, upper_pos, Depth, Nums, Indice


@njit(
    types.Tuple((float64[:, :], float64[:, :], int32[:], int64[:], int64[:]))(
        float64[:, :], int32, int32, float64[:, :], float64[:, :]
    ),
    cache=True,
)
def make_grid_by_diff(pos, maxdepth, cut_diff_partnum, lower_pos, upper_pos):
    Nums = np.zeros(1, dtype=np.int64)
    Depth = np.zeros(1, dtype=np.int32)  # 8 ** 0 = 1
    Indice = np.zeros(len(pos), dtype=np.int64)

    i = 0
    while i <= (len(Depth) - 1):

        Nums[i] = read_partical_num(Indice, i)

        flag = True
        Size = upper_pos[i] - lower_pos[i]
        Volu = Size[0] * Size[1] * Size[2]
        if (Depth[i] >= maxdepth) or Nums[i] < cut_diff_partnum or Volu < 1e-3:

            flag = False
            i = i + 1
            continue
        while (Depth[i] < maxdepth) and flag:
            # print(lower_pos[i],upper_pos[i])
            add_lower, add_upper = split_by_median_point(
                lower_pos[i], upper_pos[i], pos
            )  # 7 new #TODO median ?

            ThisNums = np.zeros(8, dtype=np.int64)
            for grid_ind in range(len(add_lower)):
                ThisNums[grid_ind] = test_partical_num(
                    posmin=add_lower[grid_ind],
                    posmax=add_upper[grid_ind],
                    pos=pos,
                    indices=Indice,
                    parent_ind=i,
                )
            Ls = add_upper - add_lower
            Vs = Ls[:, 0] * Ls[:, 1] * Ls[:, 2]

            # sortNum = np.sort(ThisNums/Vs)

            if (np.max(ThisNums) < cut_diff_partnum) or np.min(Vs) < 1e-3:
                # print(((np.sum(sortNum[4:])-np.sum(sortNum[:4]))/4),cut_diff_partnum)
                flag = False
            else:
                for grid_ind in range(len(add_lower)):
                    if grid_ind == 0:
                        continue
                    ThisNums[grid_ind] = set_partical_num(
                        posmin=add_lower[grid_ind],
                        posmax=add_upper[grid_ind],
                        pos=pos,
                        indices=Indice,
                        parent_ind=i,
                        child_ind=len(Depth) + grid_ind - 1,
                    )

                Nums[i] = ThisNums[0]
                Nums = np.concatenate((Nums, ThisNums[1:]))

                lower_pos = np.concatenate((lower_pos, add_lower[1:]))
                upper_pos = np.concatenate((upper_pos, add_upper[1:]))

                Depth[i] = Depth[i] + 1
                Depth = np.concatenate((Depth, np.ones(7, dtype=np.int32) * Depth[i]))

                lower_pos[i] = add_lower[0]
                upper_pos[i] = add_upper[0]
        i = i + 1

    return lower_pos, upper_pos, Depth, Nums, Indice

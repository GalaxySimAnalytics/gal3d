import os
from functools import wraps
import logging

import h5py

logger = logging.getLogger('gal3d.optimization.util')


def truncate(num, n):
    return float(int(num * (10**n)) / 10**n)


def provide_save_keys(shape_name, error_name):
    save_keys = ['pos', 'angle', 'a', 'eps_ab', 'eps_bc', 'parameter']
    if shape_name == 'Ellipsoid_S':
        save_keys = save_keys + ['sa', 'sb', 'sc', 'parent_fun']

    res_keys = ['fun']

    return save_keys, res_keys


def save_dict_to_hdf5(file: h5py.File | h5py.Group | str, data_dict=dict()):
    if isinstance(file, str):
        if not os.path.exists(file):
            f = h5py.File(file, 'w')
            logger.info(f"Create {file}")
            f.close()
        with h5py.File(file, 'r+') as f:
            for i in data_dict:
                f.create_dataset(i, data=data_dict[i])
        return

    if isinstance(file, h5py.File) or isinstance(file, h5py.Group):
        for i in data_dict:
            file.create_dataset(i, data=data_dict[i])
        return file


def save_model_hdf5(
    model,
    hdf5_file_name: str,
    shape_name: str,
    error_name: str,
    all_header='/',
    other_info=dict(),
):
    import os

    if all_header[-1] == '/':
        save_header = all_header + shape_name + '/' + error_name + '/'
    else:
        save_header = all_header + '/' + shape_name + '/' + error_name + '/'

    save_keys, res_keys = provide_save_keys(shape_name, error_name)

    if not os.path.exists(hdf5_file_name):
        f = h5py.File(hdf5_file_name, 'w')
        f.close()
    with h5py.File(hdf5_file_name, 'r+') as f:
        for i in save_keys:
            f.create_dataset(save_header + i, data=model[i])
        for i in res_keys:
            f.create_dataset(save_header + i, data=model.res[i])
        if other_info:
            save_dict_to_hdf5(f[all_header], other_info)
    return

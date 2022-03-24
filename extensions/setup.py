#!/usr/bin/env python3
import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
]

cxx_args = ['-std=c++14']

nvcc_args = [
    '-gencode', 'arch=compute_50,code=sm_50',
    '-gencode', 'arch=compute_52,code=sm_52',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_70,code=compute_70',
]

setup(
    name='extensions',
    ext_modules=[
        CUDAExtension(*ext_module, extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
        for ext_module in ext_modules
    ],
    cmdclass={'build_ext': BuildExtension}
)

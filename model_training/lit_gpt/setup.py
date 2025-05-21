from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='xentropy_cuda_lib',
    ext_modules=[
        CUDAExtension('xentropy_cuda_lib', [
            'csrc/xentropy_cuda.cpp',
            'csrc/xentropy_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

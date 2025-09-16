import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

VERSION = 0.1

if __name__ == '__main__':
    setup(
        name = 'pyro_slam',
        version = VERSION,
        description = 'PyTorch implementation of SLAM',
        install_requires=[
            'torch',
            'torchvision',
            'warp-lang',
        ],
        packages=find_packages(exclude=['./ba_example.py', 
                                        './setup.py', 
                                        './README.md',
                                        './data',
                                        './1dsfm_bal',
                                        './bal_data',
                                        './colmap_helpers',
                                        './datapipes',
                                        './examples',
                                        './tests',]),
        ext_modules=[
            CppExtension('pyro_slam.sparse.bsr', ['pyro_slam/sparse/sparse_op_cpp.cpp']),
            CUDAExtension('pyro_slam.sparse.bsr_cuda', ['pyro_slam/sparse/sparse_op_cuda.cpp', 'pyro_slam/sparse/sparse_op_cuda_kernel.cu']),
            CUDAExtension('pyro_slam.sparse.solve', ['pyro_slam/sparse/sparse_cusolve.cu'],
                        libraries=['cusolver', 'cusparse', 'cudss'],
                        extra_compile_args={'nvcc': ['-lcusolver',
                                                     '-lcusparse',
                                                     '-lcudss',
                                                    #  f'-I{CUDSS_DIR}/include',
                                                    #  f'-L{CUDSS_DIR}/lib',
                                                    # f'-Xlinker={CUDSS_DIR}/lib/libcudss_static.a',
                                                    ]
                                                     }
                        ),
            CUDAExtension('pyro_slam.sparse.spgemm', ['pyro_slam/sparse/cusparse_wrapper.cpp'],),
            CUDAExtension('pyro_slam.sparse.conversion', ['pyro_slam/sparse/sparse_conversion.cu'],),
        ],
        cmdclass={'build_ext': BuildExtension}
    )

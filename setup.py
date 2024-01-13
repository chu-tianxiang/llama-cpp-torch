from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='llamacpp_cuda',
      ext_modules=[cpp_extension.CUDAExtension(
          'llamacpp_cuda',
          ['py_bind.cpp', 'llamacpp_kernel.cu'],
          extra_compile_args={'cxx': ['-g', '-lineinfo', '-fno-strict-aliasing'],
                              'nvcc': ['-O3', '-g', '-Xcompiler', '-rdynamic', '-lineinfo']})],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

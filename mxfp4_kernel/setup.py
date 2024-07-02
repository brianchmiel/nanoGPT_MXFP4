from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os



this_dir = os.path.dirname(os.path.abspath(__file__))
source_cuda = [os.path.join(this_dir,  filename)
               for filename in ['mxfp4_quant.cpp',
                                'mxfp4_quant_kernel.cu',
                                ]
               ]

setup(name='mxfp4Quant',
      packages=find_packages(),
      cmdclass={'build_ext': BuildExtension},
      ext_modules=[CUDAExtension('mxfp4Quant._C', source_cuda ,extra_compile_args={'cxx': ['-g'],
                                          'nvcc': ['-O2']})],)



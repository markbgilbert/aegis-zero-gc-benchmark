# ==============================================================================
# Aegis Zero-GC Flat Arena PyTorch C10 Extension Build Script
# Reference: PyTorch RFC-0036 (pytorch/rfcs#110)
# ==============================================================================

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="aegis_c10_feeder",
    ext_modules=[
        CppExtension(
            name="aegis_c10_feeder",
            sources=["aegis_c10_feeder.cpp"],
            extra_compile_args=["-O3", "-mavx2"],
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)

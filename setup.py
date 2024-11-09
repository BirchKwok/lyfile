from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
from pathlib import Path
import platform


# 获取当前文件所在目录
utils_dir = Path(__file__).parent / 'lyfile/utils'
storage_dir = Path(__file__).parent / 'lyfile/storage'

# 基础编译参数
extra_compile_args = ["-O3"]
machine = platform.machine().lower()
system = platform.system()

# 根据不同平台设置编译参数
if machine in ['arm64', 'aarch64']:
    extra_compile_args.extend([
        "-march=armv8-a+simd",
        "-D__ARM_NEON",
        "-DHAVE_NEON=1",
        "-DHAVE_SSE2=0"
    ])
elif system == "Windows":
    extra_compile_args.extend([
        "/arch:AVX2",
        "/D__SSE2__",
        "-DHAVE_SSE2=1",
        "-DHAVE_NEON=0"
    ])
elif system in ["Linux", "Darwin"]:
    if machine in ['x86_64', 'amd64', 'i386', 'i686']:
        extra_compile_args.extend([
            "-msse2",
            "-march=native",
            "-D__SSE2__",
            "-DHAVE_SSE2=1",
            "-DHAVE_NEON=0"
        ])

# C++ 特定的编译参数
cpp_extra_compile_args = extra_compile_args.copy()
if system != "Windows":
    cpp_extra_compile_args.append("-std=c++11")


def create_extension(name, source_file, extra_includes=None):
    """创建扩展模块的辅助函数
    
    Args:
        name: 模块名称 (如 'lyfile.utils.fsst')
        source_file: 源文件路径
        extra_includes: 额外的include目录列表
    """
    includes = [np.get_include()]
    if extra_includes:
        includes.extend(extra_includes)
    
    return Extension(
        name,
        [str(source_file)],
        include_dirs=includes,
        language="c++",
        extra_compile_args=cpp_extra_compile_args,
    )

# 定义扩展模块
extensions = [
    create_extension("lyfile.storage.lyfile", storage_dir / "lyfile.pyx", [str(utils_dir)]),
    create_extension("lyfile.storage.reader", storage_dir / "reader.pyx", [str(utils_dir)]),
    create_extension("lyfile.utils.array", utils_dir / "array.pyx"),
    create_extension("lyfile.utils.fsst", utils_dir / "fsst.pyx"),
    create_extension("lyfile.utils.nnp", utils_dir / "nnp.pyx"),
    create_extension("lyfile.storage.vec_storage", storage_dir / "vec_storage.pyx", [str(utils_dir)]),
    create_extension("lyfile.storage.old_reader", storage_dir / "old_reader.pyx", [str(utils_dir)]),
    create_extension("lyfile.storage.old_lyfile", storage_dir / "old_lyfile.pyx", [str(utils_dir)]),
]


# Cython 编译指令
compiler_directives = {
    'language_level': 3,
    'boundscheck': True,
    'wraparound': True,
    'initializedcheck': True,
    'cdivision': True,
}

setup(
    name='lyfile',
    version='0.1.0',
    description='A high-performance file format library',
    author='Birch Kowk',
    author_email='birchkowk@example.com',
    packages=find_packages(),
    ext_modules=cythonize(
        extensions,
        compiler_directives=compiler_directives
    ),
    install_requires=[
        'numpy>=1.19.0',
        'cython>=0.29.0',
        'pyarrow>=0.17.0',
        'pandas>=1.0.0',
        'psutil>=5.7.0',
        'lz4>=3.0.0',
    ],
    python_requires='>=3.7',
    package_data={
        'lyfile': ['utils/*.pxd', 'storage/*.pxd'],
    },
    include_package_data=True,
)
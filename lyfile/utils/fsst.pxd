# fsst.pxd
from libc.stdint cimport uint8_t, uint32_t, int32_t
from libcpp.vector cimport vector
import cython

cdef class FSST:
    cdef:
        public int thread_count  # public 属性
        object _executor
        
    # 数字压缩相关方法
    cdef bytes _compress_numeric_py(self, const unsigned char[:] data) except *
    cdef bytes _decompress_numeric_py(self, const unsigned char[:] data) except *
    cdef cython.str _analyze_data(self, const unsigned char[:] data, int sample_size=*) except *
    cdef bytes _parallel_compress_numeric(self, const unsigned char[:] data) except *
    cdef bytes _parallel_decompress_numeric(self, const unsigned char[:] data) except *
    
    # LZ4压缩相关方法
    cdef bytes _parallel_compress_lz4(self, const unsigned char[:] data) except *
    cdef bytes _parallel_decompress_lz4(self, const unsigned char[:] data) except *
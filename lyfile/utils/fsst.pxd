from libc.stdint cimport uint8_t, uint32_t, int32_t
from libcpp.vector cimport vector
import cython

cdef class FSST:
    cdef:
        public int thread_count  # public attribute
        object _executor
        
    # Numeric compression methods
    cdef bytes _compress_numeric_py(self, const unsigned char[:] data) except *
    cdef bytes _decompress_numeric_py(self, const unsigned char[:] data) except *
    cdef cython.str _analyze_data(self, const unsigned char[:] data, int sample_size=*) except *
    cdef bytes _parallel_compress_numeric(self, const unsigned char[:] data) except *
    cdef bytes _parallel_decompress_numeric(self, const unsigned char[:] data) except *
    
    # LZ4 compression methods
    cdef bytes _parallel_compress_lz4(self, const unsigned char[:] data) except *
    cdef bytes _parallel_decompress_lz4(self, const unsigned char[:] data) except *
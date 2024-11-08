# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=True
# cython: wraparound=True
# cython: cdivision=True

import os
import time
import psutil
from typing import List
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import struct
import lz4.frame
from cython.parallel import prange
import cython
from libc.string cimport memcpy, memset
from libc.stdlib cimport malloc, free
from libc.stdio cimport sprintf
from libc.stdint cimport uint8_t, uint32_t, int32_t
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libcpp.vector cimport vector

cdef class FSST:
    """Cython implementation of FSST compression class"""
    
    def __cinit__(self, int thread_count=1):
        self.thread_count = max(1, thread_count)

    def __init__(self, int thread_count=1):
        self._executor = ThreadPoolExecutor(max_workers=thread_count)

    cdef cython.str _analyze_data(self, const unsigned char[:] data, int sample_size=1024) except *:
        """Optimized data analysis"""
        cdef:
            int data_len = data.shape[0]
            int sample_len = min(sample_size, data_len)
            int unique_count = 0
            int digit_count = 0
            int i, j
            bint is_unique
            vector[uint8_t] seen
            
        if data_len == 0:
            return 'empty'
                
        for i in range(sample_len):
            if 48 <= data[i] <= 57:
                digit_count += 1
                
            is_unique = True
            for j in range(seen.size()):
                if seen[j] == data[i]:
                    is_unique = False
                    break
            if is_unique:
                unique_count += 1
                seen.push_back(data[i])
                
        cdef float unique_ratio = <float>unique_count / sample_len
        cdef float digit_ratio = <float>digit_count / sample_len
        
        if unique_ratio > 0.9:
            return 'random'
        elif digit_ratio > 0.5:
            return 'numeric'
        elif unique_ratio < 0.1:
            return 'repetitive'
        else:
            return 'normal'

    cdef bytes _compress_numeric_py(self, const unsigned char[:] data) except *:
        """Optimized numeric compression implementation"""
        cdef:
            int data_len = data.shape[0]
            vector[uint8_t] compressed
            int i = 0
            int prev_num = 0
            int num, diff
            uint8_t marker
            
        if data_len == 0:
            return b''
                
        compressed.resize(4)
        memcpy(&compressed[0], &data_len, 4)
        
        while i < data_len:
            if i + 4 <= data_len and (
                48 <= data[i] <= 57 and
                48 <= data[i+1] <= 57 and
                48 <= data[i+2] <= 57 and
                48 <= data[i+3] <= 57
            ):
                num = (
                    (data[i] - 48) * 1000 +
                    (data[i+1] - 48) * 100 +
                    (data[i+2] - 48) * 10 +
                    (data[i+3] - 48)
                )
                
                diff = num - prev_num
                
                if -64 <= diff <= 63:
                    marker = (diff & 0x7F) | 0x80
                    compressed.push_back(marker)
                elif -8192 <= diff <= 8191:
                    marker = (diff >> 8) & 0x3F
                    compressed.push_back(marker)
                    compressed.push_back(diff & 0xFF)
                else:
                    compressed.push_back(0x40)
                    compressed.resize(compressed.size() + 4)
                    memcpy(&compressed[compressed.size()-4], &diff, 4)
                    
                prev_num = num
                i += 4
                
            else:
                compressed.push_back(0xFF)
                compressed.push_back(data[i])
                i += 1
                
        return bytes(compressed)

    cdef bytes _decompress_numeric_py(self, const unsigned char[:] data) except *:
        """Optimized numeric decompression implementation"""
        cdef:
            int data_len = data.shape[0]
            int original_length
            int pos = 4
            vector[uint8_t] result
            int prev_num = 0
            int diff, num
            uint8_t marker
            char[16] num_str
            int num_len, i
            
        if data_len < 4:
            return bytes(data)
                
        memcpy(&original_length, &data[0], 4)
        
        while pos < data_len:
            if data[pos] == 0xFF:
                pos += 1
                if pos < data_len:
                    result.push_back(data[pos])
                pos += 1
                continue
                
            marker = data[pos]
            pos += 1
            
            if marker & 0x80:
                diff = ((marker & 0x7F) ^ 0x40) - 0x40
            elif marker == 0x40:
                if pos + 4 > data_len:
                    break
                memcpy(&diff, &data[pos], 4)
                pos += 4
            else:
                if pos + 1 > data_len:
                    break
                diff = ((marker & 0x3F) << 8) | data[pos]
                if diff & 0x2000:
                    diff |= -0x4000
                pos += 1
                
            num = prev_num + diff
            num_len = sprintf(num_str, "%04d", num)
            
            for i in range(num_len):
                result.push_back(num_str[i])
                
            prev_num = num
            
        return bytes(result[:original_length])

    cdef bytes _parallel_compress_numeric(self, const unsigned char[:] data) except *:
        """Parallel numeric compression"""
        cdef:
            int chunk_size
            list chunks = []
            int i
            bytes compressed_chunk
            list compressed_chunks
            vector[uint32_t] chunk_sizes
            bytes header, sizes_bytes, result
            
        if len(data) < 1024 * self.thread_count:
            return b'\x02' + self._compress_numeric_py(data)
            
        chunk_size = (len(data) // (self.thread_count * 4)) * 4
        
        for i in range(0, len(data), chunk_size):
            chunk_view = data[i:i+chunk_size]
            chunks.append(chunk_view)
            
        compressed_chunks = list(self._executor.map(
            lambda x: self._compress_numeric_py(x), chunks))
                
        chunk_sizes.resize(len(compressed_chunks))
        for i in range(len(compressed_chunks)):
            chunk_sizes[i] = len(compressed_chunks[i])
            
        header = struct.pack('<I', len(compressed_chunks))
        sizes_bytes = struct.pack('<' + 'I' * len(compressed_chunks),
                                *[size for size in chunk_sizes])
        result = b'\x02' + header + sizes_bytes + b''.join(compressed_chunks)
            
        return result

    cdef bytes _parallel_decompress_numeric(self, const unsigned char[:] data) except *:
        """Parallel numeric decompression"""
        cdef:
            uint32_t chunk_count
            list chunks = []
            vector[uint32_t] chunk_sizes
            int pos = 4
            int i
            bytes decompressed_chunk
            list decompressed_chunks
            
        try:
            memcpy(&chunk_count, &data[0], 4)
            
            if chunk_count > 100 or chunk_count < 1:
                return self._decompress_numeric_py(data)
                
            chunk_sizes.resize(chunk_count)
            memcpy(&chunk_sizes[0], &data[pos], 4 * chunk_count)
            pos += 4 * chunk_count
            
            for i in range(chunk_count):
                if pos + chunk_sizes[i] > len(data):
                    return self._decompress_numeric_py(data)
                chunks.append(bytes(data[pos:pos + chunk_sizes[i]]))
                pos += chunk_sizes[i]
                
            decompressed_chunks = list(self._executor.map(
                lambda x: self._decompress_numeric_py(x), chunks))
                    
            return b''.join(decompressed_chunks)
            
        except Exception:
            return self._decompress_numeric_py(data)

    cdef bytes _parallel_compress_lz4(self, const unsigned char[:] data) except *:
        """Parallel LZ4 compression"""
        cdef:
            int chunk_size
            list chunks = []
            list compressed_chunks
            vector[uint32_t] chunk_sizes
            bytes header, sizes_bytes
            int i
            
        if len(data) < 1024 * self.thread_count:
            return b'\x01' + lz4.frame.compress(bytes(data))
            
        chunk_size = len(data) // self.thread_count
        
        for i in range(0, len(data), chunk_size):
            chunks.append(bytes(data[i:i+chunk_size]))
            
        compressed_chunks = list(self._executor.map(
            lz4.frame.compress, chunks))
            
        chunk_sizes.resize(len(compressed_chunks))
        for i in range(len(compressed_chunks)):
            chunk_sizes[i] = len(compressed_chunks[i])
            
        header = struct.pack('<I', len(compressed_chunks))
        sizes_bytes = struct.pack('<' + 'I' * len(compressed_chunks),
                                *[size for size in chunk_sizes])
                                
        return b'\x01' + header + sizes_bytes + b''.join(compressed_chunks)

    cdef bytes _parallel_decompress_lz4(self, const unsigned char[:] data) except *:
        """Parallel LZ4 decompression"""
        cdef:
            int n_chunks
            vector[uint32_t] chunk_sizes
            list chunks = []
            list decompressed_chunks
            int pos = 4
            int i
            
        try:
            n_chunks = struct.unpack('<I', data[:4])[0]
            chunk_sizes.resize(n_chunks)
            
            # Read each chunk size
            for i in range(n_chunks):
                chunk_sizes[i] = struct.unpack('<I', bytes(data[pos:pos+4]))[0]
                pos += 4
                
            # Split data blocks
            for i in range(n_chunks):
                if pos + chunk_sizes[i] > len(data):
                    return lz4.frame.decompress(bytes(data))
                chunks.append(bytes(data[pos:pos + chunk_sizes[i]]))
                pos += chunk_sizes[i]
                
            # Parallel decompression
            decompressed_chunks = list(self._executor.map(
                lz4.frame.decompress, chunks))
                
            return b''.join(decompressed_chunks)
            
        except Exception:
            return lz4.frame.decompress(bytes(data))

    def compress(self, data: bytes) -> bytes:
        """Compression entry method.
        
        Parameters:
            data (bytes): Data to compress.

        Returns:
            bytes: Compressed data.
        """
        if not data:
            return b''
            
        if len(data) < 1024:
            return b'\x00' + data
            
        cdef const unsigned char[:] data_view = data
        data_type = self._analyze_data(data_view, 1024)
        
        if data_type == 'random':
            return b'\x00' + data
        elif data_type == 'numeric':
            if self.thread_count == 1:
                return b'\x02' + self._compress_numeric_py(data_view)
            return self._parallel_compress_numeric(data_view)
        else:
            if self.thread_count == 1:
                return b'\x01' + lz4.frame.compress(data)
            return self._parallel_compress_lz4(data_view)

    def decompress(self, data: bytes) -> bytes:
        """Decompression entry method.
        
        Parameters:
            data (bytes): Data to decompress.

        Returns:
            bytes: Decompressed data.
        """
        if not data:
            return b''
            
        cdef:
            const unsigned char[:] data_view = data
            uint8_t method = data_view[0]
            const unsigned char[:] compressed_data = data_view[1:]
            
        if method == 0:
            return bytes(compressed_data)
        elif method == 1:
            if self.thread_count == 1:
                return lz4.frame.decompress(bytes(compressed_data))
            try:
                return self._parallel_decompress_lz4(compressed_data)
            except Exception:
                return lz4.frame.decompress(bytes(compressed_data))
        elif method == 2:
            if self.thread_count == 1:
                return self._decompress_numeric_py(compressed_data)
            try:
                return self._parallel_decompress_numeric(compressed_data)
            except Exception:
                return self._decompress_numeric_py(compressed_data)
        else:
            raise ValueError(f"Unknown compression method: {method}")

def compress(data: bytes, thread_count: int = 4) -> bytes:
    """Global compression function.
    
    Parameters:
        data (bytes): Data to compress.
        thread_count (int): Number of threads.

    Returns:
        bytes: Compressed data.
    """
    compressor = FSST(thread_count=thread_count)
    return compressor.compress(data)

def decompress(data: bytes, thread_count: int = 4) -> bytes:
    """Global decompression function.
    
    Parameters:
        data (bytes): Data to decompress.
        thread_count (int): Number of threads.

    Returns:
        bytes: Decompressed data.
    """
    compressor = FSST(thread_count=thread_count)
    return compressor.decompress(data)

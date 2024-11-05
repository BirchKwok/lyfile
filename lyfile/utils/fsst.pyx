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
    """FSST压缩类的Cython实现"""
    
    def __cinit__(self, int thread_count=1):
        self.thread_count = max(1, thread_count)

    def __init__(self, int thread_count=1):
        self._executor = ThreadPoolExecutor(max_workers=thread_count)

    cdef cython.str _analyze_data(self, const unsigned char[:] data, int sample_size=1024) except *:
        """优化的数据分析"""
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
        """优化的数字压缩实现"""
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
        """优化的数字解压实现"""
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
        """并行数字压缩"""
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
        """并行数字解压"""
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
        """并行LZ4压缩"""
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
        """并行LZ4解压缩"""
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
            
            # 读取每个块的大小
            for i in range(n_chunks):
                chunk_sizes[i] = struct.unpack('<I', bytes(data[pos:pos+4]))[0]
                pos += 4
                
            # 分割数据块
            for i in range(n_chunks):
                if pos + chunk_sizes[i] > len(data):
                    return lz4.frame.decompress(bytes(data))
                chunks.append(bytes(data[pos:pos + chunk_sizes[i]]))
                pos += chunk_sizes[i]
                
            # 并行解压缩
            decompressed_chunks = list(self._executor.map(
                lz4.frame.decompress, chunks))
                
            return b''.join(decompressed_chunks)
            
        except Exception:
            return lz4.frame.decompress(bytes(data))

    def compress(self, data: bytes) -> bytes:
        """压缩入口方法"""
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
        """解压入口方法"""
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

# 全局函数
def compress(data: bytes, thread_count: int = 4) -> bytes:
    """全局压缩函数"""
    compressor = FSST(thread_count=thread_count)
    return compressor.compress(data)

def decompress(data: bytes, thread_count: int = 4) -> bytes:
    """全局解压函数"""
    compressor = FSST(thread_count=thread_count)
    return compressor.decompress(data)

def get_memory_usage() -> float:
    """获取内存使用量"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def run_benchmark(data: bytes, name: str, thread_count: int = 4):
    """性能测试函数"""
    print(f"\n测试 {name}:")
    print(f"原始大小: {len(data):,} 字节")

    initial_memory = get_memory_usage()

    start_time = time.time()
    compressed = compress(data, thread_count)
    compress_time = time.time() - start_time
    compress_memory = get_memory_usage() - initial_memory

    print(f"压缩后大小: {len(compressed):,} 字节")
    print(f"压缩率: {len(compressed) / len(data):.2%}")
    print(f"压缩时间: {compress_time:.3f} 秒")
    print(f"压缩内存: {compress_memory:.2f} MB")

    initial_memory = get_memory_usage()

    start_time = time.time()
    decompressed = decompress(compressed, thread_count)
    decompress_time = time.time() - start_time
    decompress_memory = get_memory_usage() - initial_memory

    print(f"解压时间: {decompress_time:.3f} 秒")
    print(f"解压内存: {decompress_memory:.2f} MB")
    print(f"数据完整性: {decompressed == data}")

    return {
        'name': name,
        'original_size': len(data),
        'compressed_size': len(compressed),
        'compression_ratio': len(compressed) / len(data),
        'compress_time': compress_time,
        'decompress_time': decompress_time,
        'compress_memory': compress_memory,
        'decompress_memory': decompress_memory,
        'is_valid': decompressed == data
    }

def run_comprehensive_benchmark(thread_counts: List[int] = [1, 2, 4, 8]):
    """综合性能测试函数"""
    test_cases = [
        (b"Hello" * 100000, "重复文本"),
        (os.urandom(100000), "随机二进制"),
        (b"".join([str(i).encode() for i in range(10000)]), "数字序列"),
        (b"abcdefghijklmnopqrstuvwxyz" * 10000, "重复字母"),
        (open(__file__, 'rb').read() * 100, "Python源代码")
    ]

    all_results = []

    for thread_count in thread_counts:
        print(f"\n使用 {thread_count} 个线程进行测试")
        print("=" * 80)

        results = []
        for data, name in test_cases:
            result = run_benchmark(data, name, thread_count)
            results.append(result)

        all_results.extend(results)

        print(f"\n{thread_count} 线程测试汇总:")
        print("-" * 80)
        print(f"{'测试类型':<15} {'原始大小':>10} {'压缩大小':>10} {'压缩率':>8} "
              f"{'压缩时间':>8} {'解压时间':>8} {'压缩内存':>8} {'解压内存':>8}")
        print("-" * 80)

        for result in results:
            print(f"{result['name']:<15} {result['original_size']:>10,} "
                  f"{result['compressed_size']:>10,} {result['compression_ratio']:>7.2%} "
                  f"{result['compress_time']:>7.3f}s {result['decompress_time']:>7.3f}s "
                  f"{result['compress_memory']:>7.1f}M {result['decompress_memory']:>7.1f}M")

    return all_results

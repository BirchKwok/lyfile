# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import cython
import numpy as np
cimport numpy as cnp
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free
from libc.stdio cimport snprintf
from libc.stdint cimport uint8_t, uint32_t

# 初始化 NumPy C API
cnp.import_array()

# 定义类型
ctypedef uint8_t dtype_uint8_t

# 声明全局变量
cdef uint8_t* LOOKUP_TABLE_DATA = NULL
lookup_table_global = None  # 保持对 NumPy 数组的引用，防止被垃圾回收

def initialize_lookup_table():
    global LOOKUP_TABLE_DATA, lookup_table_global
    cdef cnp.ndarray[dtype_uint8_t, ndim=2] lookup_table
    cdef int i
    lookup_table = np.zeros((10000, 4), dtype=np.uint8)
    for i in range(10000):
        s = f"{i:04d}".encode('ascii')
        lookup_table[i, :] = np.frombuffer(s, dtype=np.uint8)
    # 获取数据指针
    LOOKUP_TABLE_DATA = <uint8_t*>lookup_table.data
    # 保持对数组的引用，防止被垃圾回收
    lookup_table_global = lookup_table

# 调用初始化函数
initialize_lookup_table()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint is_digit_chunk(const unsigned char* data) nogil:
    cdef uint32_t chunk = (<uint32_t*>data)[0]
    cdef uint32_t temp1, temp2, temp3
    cdef uint32_t mask = <uint32_t>0x80808080
    cdef uint32_t add1 = <uint32_t>0x46464646
    cdef uint32_t add2 = <uint32_t>0x56565656

    temp1 = chunk & mask
    temp2 = (chunk + add1) & mask
    temp3 = (chunk + add2) & mask

    # 使用 C 级别的逻辑操作符
    return (temp1 == 0) and (temp2 != 0) and (temp3 == 0)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int parse_int(const unsigned char* data) nogil:
    cdef uint32_t chunk = (<uint32_t*>data)[0]
    cdef uint32_t a = (chunk >> 24) & 0xFF
    cdef uint32_t b = (chunk >> 16) & 0xFF
    cdef uint32_t c = (chunk >> 8) & 0xFF
    cdef uint32_t d = chunk & 0xFF
    return (a - 48) * 1000 + (b - 48) * 100 + (c - 48) * 10 + (d - 48)

@cython.boundscheck(False)
@cython.wraparound(False)
def compress_numeric(const unsigned char[:] data):
    if not data.size:
        return b''

    cdef:
        Py_ssize_t input_size = data.size
        Py_ssize_t max_output_size = input_size + 8
        unsigned char* output = <unsigned char*>malloc(max_output_size)
        Py_ssize_t pos = 0
        Py_ssize_t output_pos = 4
        int current_num, prev_num, diff
        int group_size
        bint is_digit
        const unsigned char* data_ptr = &data[0]
        cdef int i

    if not output:
        raise MemoryError()

    try:
        # 写入原始长度
        memcpy(output, &input_size, 4)

        while pos <= input_size - 32:
            group_size = 0
            is_digit = True
            # 使用 SIMD 加速
            for i in range(8):
                if not is_digit_chunk(data_ptr + pos + i * 4):
                    is_digit = False
                    break
            group_size = 8 if is_digit else 0

            if is_digit:
                output[output_pos] = 0
                output_pos += 1

                prev_num = 0
                for i in range(group_size):
                    current_num = parse_int(data_ptr + pos + i * 4)
                    diff = current_num - prev_num

                    if -64 <= diff <= 63:
                        output[output_pos] = (diff & 0x7F) | 0x80
                        output_pos += 1
                    elif -8192 <= diff <= 8191:
                        output[output_pos] = ((diff >> 8) & 0x3F)
                        output[output_pos + 1] = diff & 0xFF
                        output_pos += 2
                    else:
                        output[output_pos] = 0x40
                        memcpy(&output[output_pos + 1], &current_num, 4)
                        output_pos += 5

                    prev_num = current_num

                pos += group_size * 4
            else:
                output[output_pos] = 0xFF
                memcpy(&output[output_pos + 1], data_ptr + pos, 4)
                output_pos += 5
                pos += 4

        while pos < input_size:
            chunk_size = min(4, input_size - pos)
            output[output_pos] = 0xFF
            memcpy(&output[output_pos + 1], data_ptr + pos, chunk_size)
            output_pos += 1 + chunk_size
            pos += chunk_size

        return PyBytes_FromStringAndSize(<char*>output, output_pos)
    finally:
        free(output)

@cython.boundscheck(False)
@cython.wraparound(False)
def decompress_numeric(const unsigned char[:] data):
    if data.size < 4:
        return bytes(data)

    cdef:
        Py_ssize_t original_length
        unsigned char* result
        Py_ssize_t pos = 0
        Py_ssize_t result_pos = 0
        int current_num, prev_num, diff
        unsigned char marker
        const unsigned char* data_ptr = &data[0]
        cdef char[16] temp

    memcpy(&original_length, data_ptr, 4)
    pos = 4

    result = <unsigned char*>malloc(original_length)
    if not result:
        raise MemoryError()

    try:
        while pos < data.size and result_pos < original_length:
            marker = data_ptr[pos]
            pos += 1

            if marker == 0xFF:
                chunk_size = min(4, data.size - pos)
                memcpy(&result[result_pos], data_ptr + pos, chunk_size)
                result_pos += chunk_size
                pos += chunk_size
            else:
                prev_num = 0
                while pos < data.size and result_pos < original_length:
                    if data_ptr[pos] & 0x80:
                        diff = ((data_ptr[pos] & 0x7F) ^ 0x40) - 0x40
                        current_num = prev_num + diff
                        pos += 1
                    elif data_ptr[pos] == 0x40:
                        memcpy(&current_num, data_ptr + pos + 1, 4)
                        pos += 5
                    else:
                        diff = ((data_ptr[pos] & 0x3F) << 8) | data_ptr[pos + 1]
                        if diff & 0x2000:
                            diff |= -0x4000
                        current_num = prev_num + diff
                        pos += 2

                    if 0 <= current_num < 10000:
                        # 直接使用数据指针访问 LOOKUP_TABLE_DATA
                        memcpy(&result[result_pos], &LOOKUP_TABLE_DATA[current_num * 4], 4)
                    else:
                        snprintf(temp, 16, "%04d", current_num)
                        memcpy(&result[result_pos], temp, 4)

                    result_pos += 4
                    prev_num = current_num

        return PyBytes_FromStringAndSize(<char*>result, min(result_pos, original_length))
    finally:
        free(result)

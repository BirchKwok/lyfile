# cimport 必要的库
cimport numpy as np
import numpy as np
from libc.stdio cimport (
    FILE, fopen, fclose, fread, fwrite, fseek, SEEK_SET
)
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, strncpy
from libc.stdint cimport uint32_t
import os

# 定义C结构体来处理文件头
cdef struct NnpHeader:
    uint32_t rows
    char dtype[30]
    uint32_t shape

# 优化后的save_nnp函数
def save_nnp(str filename, np.ndarray data, bint append=True):
    cdef:
        FILE* f
        NnpHeader header
        char* raw_data
        Py_ssize_t data_size
        uint32_t current_rows, new_rows
        const char* mode
        bytes dtype_bytes
        np.ndarray data_2d

    if data.ndim == 1:
        data_2d = data.reshape(1, -1)
    else:
        data_2d = data

    data_size = data_2d.nbytes
    file_exists = os.path.exists(filename)
    if append and not file_exists:
        append = False  # 如果文件不存在，强制设置为非追加模式
    
    mode = 'rb+' if (append and file_exists) else 'wb+'
    
    f = fopen(filename.encode('utf-8'), mode)
    if f == NULL:
        raise IOError(f"无法打开文件 {filename}")

    try:
        if append and file_exists:  # 只在文件存在且确实要追加时才读取头部
            # 读取现有头部
            if fread(&header, sizeof(NnpHeader), 1, f) != 1:
                fclose(f)
                raise IOError("读取文件头失败")
            current_rows = header.rows
            new_rows = current_rows + data_2d.shape[0]

            # 更新行数
            fseek(f, 0, SEEK_SET)
            header.rows = new_rows
            if fwrite(&header, sizeof(NnpHeader), 1, f) != 1:
                fclose(f)
                raise IOError("写入文件头失败")

            # 移动到数据的末尾
            fseek(f, sizeof(NnpHeader) + current_rows * data_2d.shape[1] * data_2d.itemsize, SEEK_SET)
        else:
            # 创建新文件头
            header.rows = data_2d.shape[0]
            dtype_bytes = str(data_2d.dtype).encode('utf-8')
            strncpy(header.dtype, dtype_bytes, 29)
            header.dtype[29] = 0
            header.shape = data_2d.shape[1]

            if fwrite(&header, sizeof(NnpHeader), 1, f) != 1:
                fclose(f)
                raise IOError("写入文件头失败")

        # 一次性写入数据
        raw_data = <char*>data_2d.data
        if fwrite(raw_data, data_size, 1, f) != 1:
            fclose(f)
            raise IOError("写入数据失败")

    finally:
        fclose(f)

# 优化后的load_nnp函数
def load_nnp(str filename, bint mmap_mode=False):
    cdef:
        FILE* f
        NnpHeader header
        np.ndarray result
        tuple shape
        np.dtype dtype_obj

    if mmap_mode:
        # 使用内存映射
        f = fopen(filename.encode('utf-8'), 'rb')
        if f == NULL:
            raise IOError(f"无法打开文件 {filename}")
        try:
            if fread(&header, sizeof(NnpHeader), 1, f) != 1:
                fclose(f)
                raise IOError("读取文件头失败")
            shape = (int(header.rows), int(header.shape))
            dtype_obj = np.dtype(header.dtype.decode('utf-8').strip())
            fclose(f)
            return np.memmap(filename, mode='r', dtype=dtype_obj,
                             shape=shape, offset=sizeof(NnpHeader))
        except:
            fclose(f)
            raise

    f = fopen(filename.encode('utf-8'), 'rb')
    if f == NULL:
        raise IOError(f"无法打开文件 {filename}")

    try:
        # 读取文件头
        if fread(&header, sizeof(NnpHeader), 1, f) != 1:
            fclose(f)
            raise IOError("读取文件头失败")

        # 创建numpy数组
        shape = (int(header.rows), int(header.shape))
        dtype_obj = np.dtype(header.dtype.decode('utf-8').strip())
        result = np.empty(shape, dtype=dtype_obj)

        # 一次性读取数据
        if fread(<char*>result.data, result.nbytes, 1, f) != 1:
            fclose(f)
            raise IOError("读取数据失败")

        return result

    finally:
        fclose(f)

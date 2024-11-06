# lyfile.pyx
from io import BytesIO
import struct
import threading
import numpy as np
import pandas as pd
import pyarrow as pa
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import OrderedDict
import io
import os
from concurrent.futures import ThreadPoolExecutor
import cython
from libc.stdint cimport int64_t

# Add necessary cimports
cimport numpy as np
from ..utils import fsst

# Define types
ctypedef np.int32_t INT32
ctypedef np.int64_t INT64
ctypedef np.float32_t FLOAT32
ctypedef np.float64_t FLOAT64
cdef int64_t file_offset


cdef class LyFile:
    """
    Optimized columnar storage file format implementation.
    """
    cdef:
        public object filepath
        public int thread_count
        public object _executor
        public object _thread_local
        public object _block_index
        public object _memory_pool
        public object _pool_lock
        public list _column_order  # 添加列顺序属性

    MAGIC = b'LYFILE01'  # Update version number
    HEADER_FORMAT = '<8sIIIQ'  # File header format
    COLUMN_HEADER_FORMAT = '<II256s'  # Column header format
    BLOCK_INDEX_ENTRY = '<QII'  # Index entry format
    COMPRESSION_THRESHOLD = 1024  # Compression threshold
    DEFAULT_CHUNK_SIZE = 250_000  # Default block size
    DEFAULT_BUFFER_SIZE = 8 * 1024 * 1024  # 8MB buffer

    # Data type definitions
    TYPE_INT32 = 1
    TYPE_INT64 = 2
    TYPE_FLOAT32 = 3
    TYPE_FLOAT64 = 4
    TYPE_STRING = 5
    TYPE_BLOB = 6
    TYPE_NUMPY = 7
    TYPE_VECTOR = 8

    def __init__(self, filepath: Union[str, Path], thread_count: int = None):
        """Initialize the LyFile.
        
        Parameters:
            filepath (Union[str, Path]): The path to the LyFile.
            thread_count (int): The number of threads to use for compression, defaults to None.
        """
        self.filepath = Path(filepath)
        self.thread_count = thread_count or min(8, os.cpu_count() * 2)
        self._executor = ThreadPoolExecutor(max_workers=self.thread_count)
        self._thread_local = threading.local()
        self._block_index = {}
        self._memory_pool = {}
        self._pool_lock = threading.Lock()
        self._column_order = []  # 初始化列顺序列表

    cdef object _get_memory_buffer(self, int size):
        """Get a buffer from the memory pool."""
        with self._pool_lock:
            for buffer_size, buffers in self._memory_pool.items():
                if buffer_size >= size and buffers:
                    return buffers.pop()
            return bytearray(size)

    cdef void _return_memory_buffer(self, object buffer):
        """Return a buffer to the memory pool."""
        with self._pool_lock:
            size = len(buffer)
            if size not in self._memory_pool:
                self._memory_pool[size] = []
            if len(self._memory_pool[size]) < self.thread_count * 2:  # Limit pool size
                self._memory_pool[size].append(buffer)

    cdef object _compress_column(self, cython.str name, object values, bint compress=True):
        """Optimized column compression method
        
        Parameters:
            name (str): The column name.
            values (np.ndarray): The column data.
            compress (bool): Whether to compress the data.

        Returns:
            Tuple[str, bytes, int, int]: The compressed data.
        """
        # Prepare data
        raw_data = self._prepare_column_data(values)
        type_id = self._get_type_id(values)
        
        # If compression is disabled or data is too small, don't compress
        if not compress or len(raw_data) < self.COMPRESSION_THRESHOLD:
            # Add uncompressed mark
            return name, b'\x00' + raw_data, type_id, len(values)
            
        # Use thread-local compressor
        if not hasattr(self._thread_local, 'compressor'):
            self._thread_local.compressor = fsst.FSST()
        
        try:
            compressed = self._thread_local.compressor.compress(raw_data)
            # Only use compressed data if compression ratio is good
            if len(compressed) < len(raw_data) * 0.9:
                # Add compression mark
                return name, b'\x01' + compressed, type_id, len(values)
            # If compression is not effective, use raw data
            return name, b'\x00' + raw_data, type_id, len(values)
        except Exception as e:
            print(f"Compression failed for column {name}: {e}")
            return name, b'\x00' + raw_data, type_id, len(values)

    cdef bytes _prepare_column_data(self, object values):
        """Prepare column data for storage"""
        type_id = self._get_type_id(values)
        
        if type_id == self.TYPE_VECTOR:
            # 优化向量类型的存储
            if isinstance(values[0], np.ndarray):
                vector_dim = values[0].shape[0]
                # 预分配内存并直接拷贝数据
                stacked_vectors = np.empty((len(values), vector_dim), dtype=np.float64)
                for i, v in enumerate(values):
                    stacked_vectors[i] = v
            else:
                vector_dim = len(values[0])
                stacked_vectors = np.array(values, dtype=np.float64)
            
            header = struct.pack('<II', vector_dim, len(values))
            return header + stacked_vectors.tobytes()
        elif type_id in (self.TYPE_INT32, self.TYPE_INT64,
                      self.TYPE_FLOAT32, self.TYPE_FLOAT64):
            return values.tobytes()
        elif type_id == self.TYPE_STRING:
            return '\0'.join(str(v) for v in values).encode('utf-8') + b'\0'
        elif type_id == self.TYPE_BLOB:
            return b'\0'.join(v if isinstance(v, bytes) else v.encode()
                            for v in values) + b'\0'
        else:  # TYPE_NUMPY
            buffer = BytesIO()
            np.save(buffer, values, allow_pickle=False)
            return buffer.getvalue()

    cdef int _get_type_id(self, object values):
        """Get data type ID.
        
        Parameters:
            values (np.ndarray): The column data.

        Returns:
            int: The data type ID.
        """
        if values.dtype == np.int32:
            return self.TYPE_INT32
        elif values.dtype == np.int64:
            return self.TYPE_INT64
        elif values.dtype == np.float32:
            return self.TYPE_FLOAT32
        elif values.dtype == np.float64:
            return self.TYPE_FLOAT64
        elif values.dtype.kind == 'O':
            if isinstance(values[0], (str, np.str_)):
                return self.TYPE_STRING
            elif isinstance(values[0], (bytes, np.bytes_)):
                return self.TYPE_BLOB
            elif isinstance(values[0], np.ndarray):
                return self.TYPE_VECTOR
        return self.TYPE_NUMPY

    cdef object _convert_input_data(self, object data):
        """Convert input data to columnar storage format."""
        if isinstance(data, pd.DataFrame):
            # 使用有序字典来保持列顺序
            result = OrderedDict()
            for name in data.columns:
                result[name] = self._convert_column(data[name])
            return result
        elif isinstance(data, pd.Series):
            return {data.name or 'value': self._convert_column(data)}
        elif isinstance(data, pa.Table):
            result = OrderedDict()
            for name in data.column_names:
                result[name] = self._convert_column(data.column(name).to_numpy())
            return result
        else:  # List[Dict]
            df = pd.DataFrame(data)
            return self._convert_input_data(df)

    cdef object _convert_column(self, object values):
        """Convert single column data to numpy array.
        
        Parameters:
            values (Union[pd.Series, np.ndarray]): The column data.

        Returns:
            np.ndarray: The converted column data.
        """
        if isinstance(values, pd.Series):
            values = values.to_numpy()

        if (values.dtype == 'object' and len(values) > 0 and
            isinstance(values[0], np.ndarray)):
            vector_dim = len(values[0])
            for i, v in enumerate(values):
                if not isinstance(v, np.ndarray):
                    raise ValueError(f"The {i}th element is not a numpy array")
                if len(v) != vector_dim:
                    raise ValueError(f"The vector dimension is inconsistent: position {i}")
            return values

        return values

    cdef void _write_to_file(self, dict blocks_info, dict compressed_data, int n_rows):
        """Optimized file write method"""
        # 计算总大小
        total_size = (28 +  # 文件头
                     sum(len(header) + len(data) 
                         for header, data in compressed_data.values()) +
                     sum(256 + 4 + len(blocks) * 16  # 块索引大小
                         for blocks in blocks_info.values()))
        
        with open(self.filepath, 'wb') as f:
            f.truncate(total_size)
            
            # 使用缓冲写入
            with io.BufferedWriter(f, buffer_size=self.DEFAULT_BUFFER_SIZE) as bf:
                # 写入文件头
                index_pos = sum(len(header) + len(data)
                              for header, data in compressed_data.values()) + 28
                bf.write(struct.pack(self.HEADER_FORMAT,
                                  self.MAGIC, 1, len(compressed_data),
                                  n_rows, index_pos))

                # 批量写入列数据
                for name in self._column_order:  # 使用有序的列名
                    header, data = compressed_data[name]
                    bf.write(header)
                    bf.write(data)
                
                # 批量写入块索引
                for name in self._column_order:  # 使用有序的列名
                    blocks = blocks_info[name]
                    bf.write(name.encode('utf-8').ljust(256, b'\0'))
                    bf.write(struct.pack('<I', len(blocks)))
                    blocks_packed = struct.pack(
                        f'<{len(blocks)}QII',
                        *[item for block in blocks for item in block]
                    )
                    bf.write(blocks_packed)

        # 更新内存中的块索引
        self._block_index = blocks_info

    cdef void _init_block_index(self):
        """Initialize block index."""
        with open(self.filepath, 'rb') as f:
            magic, version, n_cols, n_rows, index_pos = struct.unpack(
                self.HEADER_FORMAT, f.read(28))
            assert magic == self.MAGIC, "Invalid file format"

            if index_pos > 0:
                f.seek(index_pos)
                for _ in range(n_cols):
                    name_bytes = f.read(256)
                    name = name_bytes.decode('utf-8').rstrip('\0')
                    n_blocks = struct.unpack('<I', f.read(4))[0]

                    blocks = []
                    for _ in range(n_blocks):
                        offset, length, block_rows = struct.unpack(
                            self.BLOCK_INDEX_ENTRY, f.read(16))
                        blocks.append((offset, length, block_rows))
                    self._block_index[name] = blocks

    # 添加一个普通的Python方法作为包装器
    def _compress_column_wrapper(self, name: str, values: object, compress: bool = True) -> tuple:
        """Wrapper for _compress_column to be used with ThreadPoolExecutor"""
        return self._compress_column(name, values, compress)

    def write(self, data: Union[List[Dict], pd.DataFrame, pd.Series, pa.Table], compress: bool = True):
        """Optimized parallel write method"""
        # 转换输入数据
        columns = self._convert_input_data(data)
        
        # 记录列顺序并打印
        if isinstance(data, pd.DataFrame):
            self._column_order = list(data.columns)
        elif isinstance(data, pa.Table):
            self._column_order = data.column_names
        else:
            self._column_order = list(columns.keys())
        
        # 并行压缩所有列
        futures = {}
        for name in self._column_order:  # 使用有序的列名
            futures[name] = self._executor.submit(
                self._compress_column_wrapper, name, columns[name], compress)

        # 收集压缩结果
        blocks_info = {}
        compressed_data = {}
        current_pos = 28  # 文件头长度

        for name in self._column_order:  # 使用有序的列名
            name, compressed, type_id, n_rows = futures[name].result()
            name_bytes = name.encode('utf-8').ljust(256, b'\0')
            header = struct.pack(self.COLUMN_HEADER_FORMAT,
                               type_id, len(compressed), name_bytes)

            blocks_info[name] = [(current_pos + 264, len(compressed), n_rows)]
            compressed_data[name] = (header, compressed)
            current_pos += 264 + len(compressed)

        # 写入文件
        self._write_to_file(blocks_info, compressed_data, len(next(iter(columns.values()))))

    def read(self, columns: Union[Optional[List[str]], str] = None) -> pa.Table:
        """Optimized read method, directly using pyarrow to process.

        Parameters:
            columns (Optional[List[str]] or str): The column(s) to read, defaults to all columns.

        Returns:
            pa.Table: The resulting table.
        """
        with self.mmap_reader() as reader:
            result = reader.read(columns=columns)
            return result
    
    def read_vec(self, column_name: str):
        """Read vector data as a continuous ndarray.
        
        Parameters:
            column_name (str): The column name.

        Returns:
            np.ndarray: The vector data as a continuous ndarray.
        """
        with self.mmap_reader() as reader:
            result = reader.read_vec(column_name)
            
        return result

    def mmap_reader(self):
        """Create a memory-mapped reader.
        
        Returns:
            MMapReader: The memory-mapped reader.
        """
        from .mmap import MMapReader
        return MMapReader(self.filepath, thread_count=self.thread_count)

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the file (number of rows, number of columns).
        
        Returns:
            Tuple[int, int]: The shape of the file.
        """
        if not self._block_index:
            self._init_block_index()

        total_rows = 0
        if self._block_index:
            first_col = next(iter(self._block_index.values()))
            total_rows = sum(block[2] for block in first_col)

        return total_rows, len(self._block_index)

    def append(self, data: Union[List[Dict], pd.DataFrame, pd.Series, pa.Table]):
        """Append data to the file.
        
        Parameters:
            data (Union[List[Dict], pd.DataFrame, pd.Series, pa.Table]): The data to append.

        Returns:
            None
        """
        # Convert input data
        new_columns = self._convert_input_data(data)

        # Validate column structure
        if not self._block_index:
            self._init_block_index()
        existing_columns = set(self._block_index.keys())
        new_column_set = set(new_columns.keys())
        if existing_columns != new_column_set:
            raise ValueError(
                f"The appended data columns do not match. Expected columns: {existing_columns}, actual columns: {new_column_set}")

        # Parallel compression all columns
        futures = {}
        for name, values in new_columns.items():
            # 使用包装器方法而不是直接使用cdef方法
            futures[name] = self._executor.submit(
                self._compress_column_wrapper, name, values)

        # Collect compression results
        blocks_info = {}
        compressed_data = {}
        
        # Get current file size as the starting position for new data
        with open(self.filepath, 'rb') as f:
            f.seek(0, 2)  # Move to the end of the file
            current_pos = f.tell()

        for name in new_columns.keys():
            name, compressed, type_id, n_rows = futures[name].result()
            name_bytes = name.encode('utf-8').ljust(256, b'\0')
            header = struct.pack(self.COLUMN_HEADER_FORMAT,
                               type_id, len(compressed), name_bytes)

            blocks_info[name] = [(current_pos + 264, len(compressed), n_rows)]
            compressed_data[name] = (header, compressed)
            current_pos += 264 + len(compressed)

        # Append data to the file
        self._append_to_file(blocks_info, compressed_data)
        
    def _append_to_file(self, blocks_info: Dict[str, List[Tuple[int, int, int]]],
                       compressed_data: Dict[str, Tuple[bytes, bytes]]):
        """Append data to the file.
        
        Parameters:
            blocks_info (Dict[str, List[Tuple[int, int, int]]]): The block information.
            compressed_data (Dict[str, Tuple[bytes, bytes]]): The compressed data.

        Returns:
            None
        """
        # Read existing block index
        if not self._block_index:
            self._init_block_index()

        # Update block index in memory
        for name, blocks in blocks_info.items():
            if name in self._block_index:
                self._block_index[name].extend(blocks)
            else:
                self._block_index[name] = blocks

        # Open file for appending
        with open(self.filepath, 'r+b') as f:
            # Move to the end of the file
            f.seek(0, 2)
            data_end_pos = f.tell()

            # Write new data blocks
            for name, (header, data) in compressed_data.items():
                f.write(header)
                f.write(data)

            # Write updated block index
            index_start_pos = f.tell()
            for name, blocks in self._block_index.items():
                f.write(name.encode('utf-8').ljust(256, b'\0'))
                f.write(struct.pack('<I', len(blocks)))
                for offset, length, block_rows in blocks:
                    f.write(struct.pack(self.BLOCK_INDEX_ENTRY,
                                      offset, length, block_rows))

            # Update index position in file header
            f.seek(0)
            magic, version, n_cols, n_rows, _ = struct.unpack(
                self.HEADER_FORMAT, f.read(28))
            
            # Calculate new total number of rows
            total_rows = n_rows
            for blocks in blocks_info.values():
                for _, _, block_rows in blocks:
                    total_rows += block_rows

            # Write updated file header
            f.seek(0)
            f.write(struct.pack(self.HEADER_FORMAT,
                              self.MAGIC, version, len(self._block_index),
                              total_rows, index_start_pos))

# lyfile.pyx
from io import BytesIO
import mmap
import struct
import threading
import numpy as np
import pandas as pd
import pyarrow as pa
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from pathlib import Path
import io
import queue
import os
from concurrent.futures import ThreadPoolExecutor
import cython
from libc.stdint cimport int64_t

# Add necessary cimports
cimport numpy as np
from ..utils import fsst, array

# Define types
ctypedef np.int32_t INT32
ctypedef np.int64_t INT64
ctypedef np.float32_t FLOAT32
ctypedef np.float64_t FLOAT64
cdef int64_t file_offset



cdef class MMapReader:
    """Optimized memory mapping reader"""
    cdef:
        public object filepath
        public cython.int thread_count
        public object _mmap
        public object _column_info
        public object _executor
        public cython.int n_rows
        public object _page_size
        public object _cache
        public object _cache_lock
        public object _prefetch_queue
        public object _prefetch_threads
        public object _vector_cache
        public object _vector_cache_lock
        public dict _vector_metadata
        public dict _vector_maps
        public object _vector_file
        public object _vector_lock
        public cython.int _vector_cache_size
        public object _prefetch_pool
        public dict _mmap_arrays

    def __init__(self, filepath: Union[str, Path], thread_count: cython.int = 4):
        """Initialize the MMapReader"""
        self.filepath = Path(filepath)
        self.thread_count = thread_count
        self._mmap = None
        self._column_info = {}
        self._executor = ThreadPoolExecutor(max_workers=thread_count)
        self._prefetch_pool = ThreadPoolExecutor(max_workers=thread_count)
        self.n_rows = 0
        self._page_size = mmap.PAGESIZE
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._prefetch_queue = None
        self._prefetch_threads = []
        self._vector_cache = {}
        self._vector_cache_lock = threading.Lock()
        self._vector_metadata = {}
        self._vector_maps = {}
        self._mmap_arrays = {}
        self._vector_file = None
        self._vector_lock = threading.Lock()
        self._vector_cache_size = 1000

    def __enter__(self):
        self._init_mmap()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()

    cdef void _cleanup(self):
        """Clean up resources"""
        # 首先清理memmap数组
        with self._vector_lock:
            for arr in self._vector_maps.values():
                del arr
            self._vector_maps.clear()
            
            # 关闭向量文件句柄
            if self._vector_file is not None:
                self._vector_file.close()
                self._vector_file = None

        # 清理缓存
        self._vector_cache.clear()
        
        # 清理线程资源
        if self._prefetch_queue:
            for _ in self._prefetch_threads:
                self._prefetch_queue.put(None)
            for thread in self._prefetch_threads:
                thread.join()

        self._executor.shutdown(wait=True)
        if hasattr(self, '_prefetch_pool'):
            self._prefetch_pool.shutdown(wait=True)

        # 强制进行垃圾回收
        import gc
        gc.collect()

        # 最后关闭主mmap
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None

    cdef void _init_mmap(self):
        """Initialize the memory mapping"""
        with open(self.filepath, 'rb') as f:
            self._mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            # Read file header
            magic, version, n_cols, self.n_rows, index_pos = struct.unpack(
                LyFile.HEADER_FORMAT, self._mmap.read(28))
            assert magic == LyFile.MAGIC, "Invalid file format"

            if index_pos > 0:
                self._mmap.seek(index_pos)
                for _ in range(n_cols):
                    name_bytes = self._mmap.read(256)
                    name = name_bytes.decode('utf-8').rstrip('\0')
                    n_blocks = struct.unpack('<I', self._mmap.read(4))[0]

                    blocks = []
                    for _ in range(n_blocks):
                        offset, length, block_rows = struct.unpack(
                            LyFile.BLOCK_INDEX_ENTRY, self._mmap.read(16))

                        # Read type information
                        current_pos = self._mmap.tell()
                        self._mmap.seek(offset - 264)
                        type_id, _, _ = struct.unpack(
                            LyFile.COLUMN_HEADER_FORMAT, self._mmap.read(264))
                        self._mmap.seek(current_pos)

                        blocks.append((type_id, length, offset, block_rows))

                    self._column_info[name] = blocks

        # Initialize prefetch
        self._prefetch_queue = queue.Queue(maxsize=self.thread_count * 2)
        for _ in range(self.thread_count):
            thread = threading.Thread(target=self._prefetch_worker, daemon=True)
            thread.start()
            self._prefetch_threads.append(thread)

        # 初始化向量文件句柄
        self._vector_file = open(self.filepath, 'rb')
        
        # 预处理向量列的元数据
        for name, blocks in self._column_info.items():
            if blocks and blocks[0][0] == LyFile.TYPE_VECTOR:
                with threading.Lock():
                    self._mmap.seek(blocks[0][2])  # offset
                    header_data = self._mmap.read(9)  # compression flag + vector metadata
                    compression_flag = header_data[0]
                    if compression_flag == 0:
                        vector_dim, _ = struct.unpack('<II', header_data[1:9])
                        self._vector_metadata[name] = {
                            'dim': vector_dim,
                            'blocks': blocks
                        }

    cdef object _fast_read_vector(self, cython.str column_name, int block_idx):
        """Fast vector reading using memory mapping"""
        metadata = self._vector_metadata.get(column_name)
        if not metadata:
            return None

        block = metadata['blocks'][block_idx]
        type_id, length, offset, n_rows = block

        # 检查缓存
        cache_key = (column_name, block_idx)
        with self._vector_cache_lock:
            if cache_key in self._vector_cache:
                return self._vector_cache[cache_key]

        # 直接从文件创建memmap
        with threading.Lock():
            self._mmap.seek(offset)
            header_data = self._mmap.read(9)  # compression flag + vector metadata
            compression_flag = header_data[0]
            
            if compression_flag == 0:
                vector_dim = metadata['dim']
                
                # 使用共享文件句柄创建memmap
                with self._vector_lock:
                    if cache_key not in self._vector_maps:
                        arr = np.memmap(self._vector_file, dtype=np.float64, mode='r',
                                      offset=offset + 9,  # skip header
                                      shape=(n_rows, vector_dim))
                        self._vector_maps[cache_key] = arr
                    else:
                        arr = self._vector_maps[cache_key]
                
                # 创建FixedSizeListArray
                result = pa.FixedSizeListArray.from_arrays(
                    pa.array(arr.reshape(-1), type=pa.float64()),
                    vector_dim
                )
                
                # 缓存结果
                with self._vector_cache_lock:
                    if len(self._vector_cache) < self._vector_cache_size:
                        self._vector_cache[cache_key] = result
                
                return result
            else:
                # 压缩数据的处理（保持原有逻辑）
                data = self._mmap.read(length - 9)
                compressor = fsst.FSST()
                try:
                    raw_data = compressor.decompress(data)
                except Exception as e:
                    raise ValueError(f"Decompression failed: {e}")
                
                return self._parse_vector_data(raw_data)

    cdef object _parse_vector_data(self, bytes raw_data):
        """Parse vector data"""
        vector_dim, n_vectors = struct.unpack('<II', raw_data[:8])
        vectors_data = raw_data[8:]
        vectors = np.frombuffer(vectors_data, dtype=np.float64).reshape(n_vectors, vector_dim)
        return pa.FixedSizeListArray.from_arrays(
            pa.array(vectors.ravel(), type=pa.float64()),
            vector_dim
        )

    def read(self, columns=None):
        """Optimized read method"""
        if isinstance(columns, str):
            columns = [columns]
        elif columns is None:
            columns = list(self._column_info.keys())

        # 区分向量列和非向量列
        vector_columns = []
        regular_columns = []
        
        for col in columns:
            if col in self._vector_metadata:
                vector_columns.append(col)
            else:
                regular_columns.append(col)

        # 并行预读取向量数据
        vector_results = {}
        if vector_columns:
            futures = []
            for col in vector_columns:
                metadata = self._vector_metadata[col]
                for block_idx in range(len(metadata['blocks'])):
                    futures.append((col, block_idx, 
                        self._prefetch_pool.submit(
                            self._fast_read_vector,
                            col, block_idx
                        )))

            # 收集结果
            for col, block_idx, future in futures:
                if col not in vector_results:
                    vector_results[col] = []
                vector_results[col].append((block_idx, future.result()))

        # 读取常规列
        regular_results = {}
        if regular_columns:
            regular_results = self._read_multiple_columns(regular_columns)

        # 合并结果
        arrays = []
        names = []
        
        # 处理向量列
        for col in vector_columns:
            sorted_blocks = sorted(vector_results[col], key=lambda x: x[0])
            arrays.append(pa.concat_arrays([block[1] for block in sorted_blocks]))
            names.append(col)

        # 处理常规列
        if regular_results:
            for col in regular_columns:
                arrays.append(regular_results[col])
                names.append(col)

        return pa.Table.from_arrays(arrays, names=names)

    cdef object _fast_single_column_read(self, cython.str column):
        """Optimized single column read implementation"""
        if column not in self._column_info:
            raise KeyError(f"Column {column} not found")

        blocks = self._column_info[column]
        arrays = []

        for type_id, length, offset, block_rows in blocks:
            # Cache check and data read
            with self._cache_lock:
                if offset in self._cache:
                    block_data = self._cache[offset]
                else:
                    with threading.Lock():
                        self._mmap.seek(offset)
                        block_data = self._mmap.read(length)
                    if len(self._cache) < 1000:
                        self._cache[offset] = block_data

            compression_flag = block_data[0]
            data = block_data[1:]

            if compression_flag == 0:
                raw_data = data
            else:
                compressor = fsst.FSST()
                try:
                    raw_data = compressor.decompress(data)
                except Exception as e:
                    raise ValueError(f"Decompression failed: {e}")

            arrays.append(MMapReader._parse_block_data_arrow(type_id, raw_data))

        combined_array = pa.concat_arrays(arrays)
        return pa.Table.from_arrays([combined_array], names=[column])

    cdef object _read_multiple_columns(self, list columns):
        """Read multiple columns"""
        futures = {}
        for col in columns:
            if col not in self._column_info:
                raise KeyError(f"Column {col} not found")

            col_futures = []
            for i, (type_id, length, offset, _) in enumerate(self._column_info[col]):
                col_futures.append((i, self._executor.submit(
                    self._read_block_arrow, type_id, offset, length)))
            futures[col] = col_futures

        # Collect results
        arrays = {}
        for col in columns:
            sorted_futures = sorted(futures[col], key=lambda x: x[0])
            column_arrays = []
            for _, future in sorted_futures:
                try:
                    column_arrays.append(future.result())
                except Exception as e:
                    print(f"Error reading column {col}: {e}")
                    raise
            arrays[col] = pa.concat_arrays(column_arrays)

        return pa.Table.from_arrays([arrays[col] for col in columns], names=columns)

    @staticmethod
    cdef object _parse_block_data_arrow(int type_id, bytes raw_data):
        """Parse data block to pyarrow array"""
        if type_id == LyFile.TYPE_VECTOR:
            # 优化向量类型的解析
            vector_dim, n_vectors = struct.unpack('<II', raw_data[:8])
            vectors_data = raw_data[8:]
            # 直接使用numpy的reshape，避免Python层面的列表转换
            vectors = np.frombuffer(vectors_data, dtype=np.float64).reshape(n_vectors, vector_dim)
            # 使用pyarrow的固定长度列表类型
            return pa.FixedSizeListArray.from_arrays(
                pa.array(vectors.ravel(), type=pa.float64()),
                vector_dim
            )
        elif type_id == LyFile.TYPE_INT32:
            return pa.array(np.frombuffer(raw_data, dtype=np.int32))
        elif type_id == LyFile.TYPE_INT64:
            return pa.array(np.frombuffer(raw_data, dtype=np.int64))
        elif type_id == LyFile.TYPE_FLOAT32:
            return pa.array(np.frombuffer(raw_data, dtype=np.float32))
        elif type_id == LyFile.TYPE_FLOAT64:
            return pa.array(np.frombuffer(raw_data, dtype=np.float64))
        elif type_id == LyFile.TYPE_STRING:
            strings = raw_data.decode('utf-8').split('\0')[:-1]
            return pa.array(strings, type=pa.string())
        elif type_id == LyFile.TYPE_BLOB:
            blobs = raw_data.split(b'\0')[:-1]
            return pa.array(blobs, type=pa.binary())
        else:  # TYPE_NUMPY
            array_data = np.load(BytesIO(raw_data), allow_pickle=False)
            return pa.array(array_data)

    cdef object _read_block_arrow(self, int type_id, int offset, int length):
        """Read a single data block and return pyarrow array"""
        if type_id == LyFile.TYPE_VECTOR:
            # 检查向量缓存
            with self._vector_cache_lock:
                if offset in self._vector_cache:
                    return self._vector_cache[offset]
        
        with threading.Lock():
            self._mmap.seek(offset)
            data = self._mmap.read(length)
        
        compression_flag = data[0]
        data = data[1:]
        
        if compression_flag == 0:
            raw_data = data
        else:
            compressor = fsst.FSST()
            try:
                raw_data = compressor.decompress(data)
            except Exception as e:
                raise ValueError(f"Decompression failed: {e}")
        
        result = MMapReader._parse_block_data_arrow(type_id, raw_data)
        
        # 缓存向量数据
        if type_id == LyFile.TYPE_VECTOR:
            with self._vector_cache_lock:
                if len(self._vector_cache) < 100:  # 限制缓存大小
                    self._vector_cache[offset] = result
        
        return result

    cdef void _prefetch_worker(self):
        """Prefetch worker thread"""
        while True:
            try:
                block_info = self._prefetch_queue.get()
                if block_info is None:
                    break

                type_id, length, offset, _ = block_info
                with threading.Lock():
                    self._mmap.seek(offset)
                    data = self._mmap.read(length)

                with self._cache_lock:
                    self._cache[offset] = data
            except Exception as e:
                print(f"Prefetch error: {e}")

    cdef object _read_block(self, int type_id, int offset, int length):
        """Read a single data block"""
        compressor = fsst.FSST()

        with threading.Lock():
            self._mmap.seek(offset)
            data = self._mmap.read(length)
            
        compression_flag = data[0]
        data = data[1:]
        
        if compression_flag == 0:
            raw_data = data
        else:
            raw_data = compressor.decompress(data)
            
        return MMapReader._parse_block_data(type_id, raw_data)

    @staticmethod
    cdef object _parse_block_data(int type_id, bytes raw_data):
        """Parse data block"""
        with threading.Lock():
            if type_id in (LyFile.TYPE_INT32, LyFile.TYPE_INT64,
                        LyFile.TYPE_FLOAT32, LyFile.TYPE_FLOAT64):
                dtype = {
                    LyFile.TYPE_INT32: np.int32,
                    LyFile.TYPE_INT64: np.int64,
                    LyFile.TYPE_FLOAT32: np.float32,
                    LyFile.TYPE_FLOAT64: np.float64
                }[type_id]
                return np.frombuffer(raw_data, dtype=dtype)
            elif type_id == LyFile.TYPE_STRING:
                strings = raw_data.decode('utf-8').split('\0')[:-1]
                return np.array(strings)
            elif type_id == LyFile.TYPE_BLOB:
                blobs = raw_data.split(b'\0')[:-1]
                return np.array(blobs)
            elif type_id == LyFile.TYPE_VECTOR:
                vector_dim, n_vectors = struct.unpack('<II', raw_data[:8])
                vectors_data = raw_data[8:]
                flat_vectors = np.frombuffer(vectors_data, dtype=np.float64)
                vectors = flat_vectors.reshape(n_vectors, vector_dim)
                result = np.empty(n_vectors, dtype=object)
                for i in range(n_vectors):
                    result[i] = vectors[i].copy()
                return result
            else:  # TYPE_NUMPY
                return np.load(BytesIO(raw_data), allow_pickle=False)

    def execute_along_column(self, column: str, func: Callable, parallel: bool = True) -> Any:
        """Execute function on a column, supports parallel processing
        
        Parameters:
            column (str): The column name
            func (Callable): The function to execute
            parallel (bool): Whether to use parallel processing, defaults to True

        Returns:
            Any: The result of the function execution

        Raises:
            KeyError: If the column is not found
        """
        if column not in self._column_info:
            raise KeyError(f"Column {column} not found")

        blocks = self._column_info[column]
        
        if not parallel:
            # Serial processing
            return self._execute_serial(blocks, func)
        
        # Parallel processing
        futures = []
        for type_id, length, offset, block_rows in blocks:
            futures.append(self._executor.submit(
                self._process_block,
                type_id, offset, length
            ))

        # Collect all data blocks
        all_data = []
        total_size = 0
        for future in futures:
            try:
                block_data = future.result()
                all_data.append(block_data)
                total_size += len(block_data)
            except Exception as e:
                raise ValueError(f"Block processing failed: {e}")

        # Preallocate result array and merge data
        result = np.empty(total_size, dtype=all_data[0].dtype)
        pos = 0
        for block in all_data:
            block_size = len(block)
            result[pos:pos + block_size] = block
            pos += block_size

        # Execute function on the complete data
        try:
            return func(result)
        except Exception as e:
            raise ValueError(f"Function execution failed: {e}")

    def execute_along_column_aggregation(self, column: str, 
                                       block_func: Callable, 
                                       merge_func: Callable) -> Any:
        """Parallel processing for aggregation operations
        
        Parameters:
            column (str): The column name
            block_func (Callable): The function to execute on each data block
            merge_func (Callable): The function to merge intermediate results

        Returns:
            Any: The result of the aggregation operation

        Raises:
            KeyError: If the column is not found
        """
        if column not in self._column_info:
            raise KeyError(f"Column {column} not found")
        
        blocks = self._column_info[column]
        
        # Parallel processing each data block
        futures = []
        for type_id, length, offset, block_rows in blocks:
            futures.append(self._executor.submit(
                self._process_block_with_func,
                type_id, offset, length, block_func
            ))

        # Collect intermediate results
        intermediate_results = []
        for future in futures:
            try:
                result = future.result()
                intermediate_results.append(result)
            except Exception as e:
                raise ValueError(f"Block processing failed: {e}")

        # Merge results
        try:
            return merge_func(intermediate_results)
        except Exception as e:
            raise ValueError(f"Result merging failed: {e}")

    def _process_block(self, type_id: int, offset: int, length: int) -> np.ndarray:
        """Process a single data block and return parsed data"""
        with threading.Lock():
            self._mmap.seek(offset)
            block_data = self._mmap.read(length)

        # Check compression flag
        compression_flag = block_data[0]
        data = block_data[1:]
        
        if compression_flag == 0:
            raw_data = data
        else:
            compressor = fsst.FSST()
            try:
                raw_data = compressor.decompress(data)
            except Exception as e:
                raise ValueError(f"Decompression failed: {e}")

        return MMapReader._parse_block_data(type_id, raw_data)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def read_vec(self, cython.str column_name):
        """优化的向量数据读取方法，使用memmap和非复制视图

        Args:
            column_name: 列名

        Returns:
            np.ndarray: 连续的向量数组视图
        """
        cdef:
            cython.int vector_dim
            cython.int total_rows = 0
            tuple cache_key
            object mmap_obj
            list block_views = []
            dict mmap_objects = {}
            object metadata
            list blocks
            cython.int block_idx, num_blocks
            cython.int type_id, length, offset, n_rows
            unsigned char compression_flag
            int64_t file_offset
            cython.int i

        if column_name not in self._vector_metadata:
            raise KeyError(f"Column {column_name} not found or not a vector column")

        metadata = self._vector_metadata[column_name]
        blocks = metadata['blocks']
        vector_dim = metadata['dim']
        num_blocks = len(blocks)

        # 检查完整缓存
        cache_key = (column_name, 'full')
        if cache_key in self._vector_maps:
            return self._vector_maps[cache_key]

        # 计算总行数
        for i in range(num_blocks):
            total_rows += blocks[i][3]  # blocks[i][3] is n_rows

        # 为每个数据块创建memmap和视图
        for block_idx from 0 <= block_idx < num_blocks:
            type_id = blocks[block_idx][0]
            length = blocks[block_idx][1]
            offset = blocks[block_idx][2]
            n_rows = blocks[block_idx][3]

            # 读取压缩标志
            file_offset = offset
            # with gil:
            self._mmap.seek(file_offset)
            compression_flag = self._mmap.read_byte()
            if compression_flag != 0:
                raise ValueError(f"Block {block_idx} is compressed")

            # 创建memmap对象，直接指定正确的形状
            mmap_obj = np.memmap(self._vector_file, dtype=np.float64, mode='r',
                                 offset=file_offset + 9,  # 跳过header
                                 shape=(n_rows, vector_dim))

            # 保存memmap对象和视图
            mmap_objects[block_idx] = mmap_obj
            block_views.append(mmap_obj)

        # 创建自定义的数组对象，组合所有视图
        result = array.ArrayView(block_views, total_rows, vector_dim)

        # 缓存结果和memmap对象
        with self._vector_cache_lock:
            self._vector_maps[cache_key] = result
            self._mmap_arrays.update(mmap_objects)

        return result


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
        self._block_index: Dict[str, List[Tuple[int, int, int]]] = {}
        self._memory_pool = {}
        self._pool_lock = threading.Lock()

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
        """Convert input data to columnar storage format.
        
        Parameters:
            data (Union[List[Dict], pd.DataFrame, pd.Series, pa.Table]): The input data.

        Returns:
            Dict[str, np.ndarray]: The converted data.
        """
        if isinstance(data, pd.DataFrame):
            return {name: self._convert_column(values)
                   for name, values in data.items()}
        elif isinstance(data, pd.Series):
            return {data.name or 'value': self._convert_column(data)}
        elif isinstance(data, pa.Table):
            return {name: self._convert_column(data.column(name).to_numpy())
                   for name in data.column_names}
        else:  # List[Dict]
            df = pd.DataFrame(data)
            return {name: self._convert_column(values)
                   for name, values in df.items()}

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
        """Optimized file write method
        
        Parameters:
            blocks_info (Dict[str, List[Tuple[int, int, int]]]): The block information.
            compressed_data (Dict[str, Tuple[bytes, bytes]]): The compressed data.
            n_rows (int): The number of rows.
        """
        # Preallocate file size
        total_size = (28 +  # File header
                     sum(len(header) + len(data) 
                         for header, data in compressed_data.values()) +
                     sum(256 + 4 + len(blocks) * 16  # Block index size
                         for blocks in blocks_info.values()))
        
        with open(self.filepath, 'wb') as f:
            f.truncate(total_size)
            
            # Buffered write
            with io.BufferedWriter(f, buffer_size=self.DEFAULT_BUFFER_SIZE) as bf:
                # Write file header
                index_pos = sum(len(header) + len(data)
                              for header, data in compressed_data.values()) + 28
                bf.write(struct.pack(self.HEADER_FORMAT,
                                  self.MAGIC, 1, len(compressed_data),
                                  n_rows, index_pos))

                # Batch write column data
                for name, (header, data) in compressed_data.items():
                    bf.write(header)
                    bf.write(data)
                
                # Batch write block index
                for name, blocks in blocks_info.items():
                    bf.write(name.encode('utf-8').ljust(256, b'\0'))
                    bf.write(struct.pack('<I', len(blocks)))
                    blocks_packed = struct.pack(
                        f'<{len(blocks)}QII',
                        *[item for block in blocks for item in block]
                    )
                    bf.write(blocks_packed)

        # Update block index in memory
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
        """Optimized parallel write method
        
        Parameters:
            data (Union[List[Dict], pd.DataFrame, pd.Series, pa.Table]): The data to write.
            compress (bool): Whether to compress the data, defaults to True.
        """
        # Convert input data
        columns = self._convert_input_data(data)

        # Parallel compression all columns
        futures = {}
        for name, values in columns.items():
            # 使用包装器方法而不是直接使用cdef方法
            futures[name] = self._executor.submit(
                self._compress_column_wrapper, name, values, compress)

        # Collect compression results
        blocks_info = {}
        compressed_data = {}
        current_pos = 28  # File header length

        for name in columns.keys():
            name, compressed, type_id, n_rows = futures[name].result()
            name_bytes = name.encode('utf-8').ljust(256, b'\0')
            header = struct.pack(self.COLUMN_HEADER_FORMAT,
                               type_id, len(compressed), name_bytes)

            blocks_info[name] = [(current_pos + 264, len(compressed), n_rows)]
            compressed_data[name] = (header, compressed)
            current_pos += 264 + len(compressed)

        # Write to file
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

    def mmap_reader(self) -> 'MMapReader':
        """Create a memory-mapped reader.
        
        Returns:
            MMapReader: The memory-mapped reader.
        """
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

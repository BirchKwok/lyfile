from io import BytesIO
import mmap
import struct
import threading
import numpy as np
import pyarrow as pa
from typing import Union, Callable, Any
from pathlib import Path
import queue
from concurrent.futures import ThreadPoolExecutor
import cython
from libc.stdint cimport int64_t

# Add necessary cimports
cimport numpy as np
from lyfile.storage.lyfile import LyFile

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
        public list _column_order

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
        self._column_order = []

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
                    self._column_order.append(name)
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
            # 使用保存的列顺序
            columns = self._column_order
        else:
            # 确保用户指定的列按原始顺序排序
            columns = [col for col in self._column_order if col in columns]

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
        
        # 按照原始列顺序处理结果
        for col in columns:  # 使用有序的列名列表
            if col in vector_columns:
                sorted_blocks = sorted(vector_results[col], key=lambda x: x[0])
                arrays.append(pa.concat_arrays([block[1] for block in sorted_blocks]))
                names.append(col)
            elif col in regular_columns:
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

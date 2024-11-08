import pyarrow as pa
from typing import Union, Optional, List
from pathlib import Path
from libc.stdint cimport int64_t
import os
import pyarrow.parquet as pq
import pandas as pd
import pyarrow.dataset as ds
import numpy as np

# Add necessary cimports
cimport numpy as np

from .vec_storage import VecStorage
from ..utils.array import ArrayView

# Define types
ctypedef np.int32_t INT32
ctypedef np.int64_t INT64
ctypedef np.float32_t FLOAT32
ctypedef np.float64_t FLOAT64
cdef int64_t file_offset


class TableWrapper:
    """包装 PyArrow Table 的鸭子类型类"""
    def __init__(self, table: pa.Table):
        # 使用 object.__setattr__ 避免递归调用
        object.__setattr__(self, '_table', table)
        
        # 获取所有 table 的公共属性和方法
        table_attrs = set(dir(table))
        # 排除我们自己实现的方法和私有属性
        exclude_attrs = {'to_pandas', '__class__', '__dict__', '__weakref__', '__new__', '__init__'}
        proxy_attrs = table_attrs - exclude_attrs
        
        # 将所有属性代理到 self._table
        for attr in proxy_attrs:
            if not hasattr(self, attr):
                setattr(self.__class__, attr, property(lambda self, attr=attr: getattr(self._table, attr)))

    def __getattribute__(self, name):
        """处理所有属性访问"""
        try:
            # 首先尝试获取对象自己的属性
            return object.__getattribute__(self, name)
        except AttributeError:
            # 如果属性不存在，则尝试从 _table 获取
            return getattr(object.__getattribute__(self, '_table'), name)

    def to_pandas(self) -> pd.DataFrame:
        """转换为 Pandas DataFrame"""
        cols = self._table.column_names
        
        chunk_cols = []
        regular_cols = []
        
        # 检查每一列的第一个元素
        for col in cols:
            if len(self._table[col]) > 0 and isinstance(self._table[col].chunk(0), pa.FixedSizeListArray):
                chunk_cols.append(col)
            else:
                regular_cols.append(col)

        # 处理向量列
        chunk_data = {col: col_data.to_pandas() for col, col_data in zip(chunk_cols, self._table.select(chunk_cols))}
        
        # 创建包含向量的 DataFrame
        vec_df = pd.DataFrame(data=chunk_data, columns=chunk_cols)
        
        # 处理常规列
        if regular_cols:
            regular_df = self._table.select(regular_cols).to_pandas()
            result = pd.concat([vec_df, regular_df], axis="columns")
        else:
            result = vec_df

        # 确保列顺序与原表一致
        return result[cols]

    def __getitem__(self, item):
        return self._table[item]

    def __repr__(self):
        return self._table.__repr__()

    def __str__(self):
        return self.__repr__()


cdef class MMapReader:
    """Parquet-based reader implementation."""
    cdef:
        public object folder_path
        public object vec_path
        public int thread_count
        public list _parquet_files
        public dict _vector_file_paths
        public int n_rows
        public list _column_order

    def __init__(self, folder_path: Union[str, Path], thread_count: int = 4):
        """Initialize the reader."""
        self.folder_path = Path(folder_path)
        self.vec_path = self.folder_path.parent / "vec_data"
        self.thread_count = thread_count
        self._parquet_files = []
        self._vector_file_paths = {}
        self._column_order = []
        self.n_rows = 0
        self._init_reader()

    cdef void _init_reader(self):
        """Initialize the parquet reader and metadata."""
        # 获取所有分区文件
        self._parquet_files = sorted(self.folder_path.glob("part-*.ly"))
        
        if not self._parquet_files:
            return
            
        # 从第一个文件获取schema
        first_file = pq.ParquetFile(self._parquet_files[0])
        self._column_order = first_file.schema.names
        
        # 计算总行数
        for parquet_file in self._parquet_files:
            metadata = pq.read_metadata(parquet_file)
            self.n_rows += metadata.num_rows

        # 检查所有可能的向量文件
        for vec_file in self.vec_path.glob("*.vec"):
            col_name = vec_file.stem  # 获取文件名（不含扩展名）
            col_name = col_name.split("-")[0]
            self._vector_file_paths[col_name] = vec_file
            if col_name not in self._column_order:
                self._column_order.append(col_name)
                
    def read(self, columns: Union[Optional[List[str]], str] = None) -> pa.Table:
        """读取指定列的数据
        
        Args:
            columns: 要读取的列名。可以是单个列名、列名列表或 None（读取所有列）
            
        Returns:
            pa.Table: 包含指定列的 PyArrow 表格
        """
        if isinstance(columns, str):
            columns = [columns]
        elif columns is None:
            columns = self._column_order
            
        # 分离向量列和常规列
        vector_cols = [col for col in columns if col in self._vector_file_paths]
        regular_cols = [col for col in columns if col not in self._vector_file_paths]
        
        # 使用 pyarrow.dataset 读取分区数据
        if regular_cols:
            dataset = ds.dataset(
                self.folder_path,
                format='parquet',
                partitioning=None  # 如果有特定的分区方案，可以在这里指定
            )
            table = dataset.to_table(columns=regular_cols)
        else:
            table = pa.Table.from_pandas(pd.DataFrame(index=range(self.n_rows)))

        # 读取向量列
        for col in vector_cols:
            vec_storage = VecStorage(self.vec_path, col)
            _arrays = vec_storage.load_vec(mmap_mode=True)
            vectors_col_num = _arrays.shape[1]
            flat_data = _arrays.reshape(-1)
            arrow_array = pa.array(flat_data, type=pa.float64())
            vector_array = pa.FixedSizeListArray.from_arrays(arrow_array, list_size=vectors_col_num)
            table = table.append_column(col, vector_array)
            
        # 确保列的顺序与请求的顺序一致
        table = table.select(columns)
        return TableWrapper(table)

    def read_batch(self, batch_size: int = 1000, columns: Union[Optional[List[str]], str] = None):
        """批量读取数据的生成器"""
        if isinstance(columns, str):
            columns = [columns]
        elif columns is None:
            columns = self._column_order
            
        # 分离向量列和常规列
        vector_cols = [col for col in columns if col in self._vector_file_paths]
        regular_cols = [col for col in columns if col not in self._vector_file_paths]
        
        # 预加载所有向量数据
        vector_data = {}
        for col in vector_cols:
            vec_storage = VecStorage(self.vec_path, col)
            vector_data[col] = vec_storage.load_vec(mmap_mode=True)

        # 使用 pyarrow.dataset 批量读取
        if regular_cols:
            dataset = ds.dataset(
                self.folder_path,
                format='parquet',
                partitioning=None
            )
            scanner = dataset.scanner(
                columns=regular_cols,
                batch_size=batch_size
            )
            
            row_offset = 0
            for batch in scanner.to_batches():
                current_batch_size = len(batch)
                
                # initial an empty PyArrow table
                table = pa.table([])
                
                # 添加向量列
                for col in vector_cols:
                    start_idx = row_offset
                    end_idx = row_offset + current_batch_size

                    _arrays = vector_data[col][start_idx:end_idx]
                    vectors_col_num = _arrays.shape[1]
                    flat_data = _arrays.reshape(-1)
                    arrow_array = pa.array(flat_data, type=pa.float64())
                    vector_array = pa.FixedSizeListArray.from_arrays(arrow_array, list_size=vectors_col_num)
                    table = table.append_column(col, vector_array)

                yield TableWrapper(table)
                row_offset += current_batch_size
        else:
            # 如果只有向量列，按批次生成数据
            for start_idx in range(0, self.n_rows, batch_size):
                end_idx = min(start_idx + batch_size, self.n_rows)
                table = pa.Table.from_pandas(pd.DataFrame(index=range(start_idx, end_idx)))
                for col in vector_cols:
                    _arrays = vector_data[col][start_idx:end_idx]
                    vectors_col_num = _arrays.shape[1]
                    flat_data = _arrays.reshape(-1)
                    arrow_array = pa.array(flat_data, type=pa.float64())
                    vector_array = pa.FixedSizeListArray.from_arrays(arrow_array, list_size=vectors_col_num)
                    table = table.append_column(col, vector_array)
                
                yield TableWrapper(table)

    def __getitem__(self, key):
        """支持切片操作和花样索引"""
        # 处理列名索引
        if isinstance(key, (str, list)) and (isinstance(key, str) or isinstance(key[0], str)):
            return self.read(columns=key)
        
        # 计算实际的行索引范围
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop if key.stop is not None else self.n_rows
            step = key.step or 1
            row_indices = range(start, stop, step)
        elif isinstance(key, (int, np.integer)):
            row_indices = range(key, key + 1)
        elif isinstance(key, (list, np.ndarray)):
            row_indices = key
        else:
            raise TypeError(f"不支持的索引类型: {type(key)}")
        
        # 分别读取常规列和向量列
        regular_cols = [col for col in self._column_order if col not in self._vector_file_paths]
        if regular_cols:
            dataset = ds.dataset(
                self.folder_path,
                format='parquet',
                partitioning=None
            )
            # 使用 filter 和 take 只读取需要的行
            scanner = dataset.scanner(columns=regular_cols)
            table = scanner.take(row_indices)
        else:
            table = pa.table([])
        
        # 读取向量列并添加到table中
        for col in self._vector_file_paths:
            vec_storage = VecStorage(self.vec_path, col)
            # 使用 mmap_mode 实现高效切片
            # 只读取需要的行
            _arrays = vec_storage[row_indices]
            vectors_col_num = _arrays.shape[1]
            flat_data = _arrays.reshape(-1)
            arrow_array = pa.array(flat_data, type=pa.float64())
            vector_array = pa.FixedSizeListArray.from_arrays(arrow_array, list_size=vectors_col_num)
            table = table.append_column(col, vector_array)
        
        return TableWrapper(table)

    def __len__(self):
        """返回数据行数"""
        return self.n_rows
    def read_vec(self, column_name: str, mmap_mode: bool = False) -> Union[np.ndarray, ArrayView]:
        """读取向量数据
        
        Args:
            column_name: 向量列名
            mmap_mode: 是否使用内存映射读取文件
        Returns:
            numpy.ndarray: 向量数据数组
            
        Raises:
            KeyError: 如果列名不存在或不是向量列
        """
        if column_name not in self._vector_file_paths:
            raise KeyError(f"列 '{column_name}' 不是向量列或不存在")
            
        vec_storage = VecStorage(self.vec_path, column_name)
        return vec_storage.load_vec(mmap_mode=mmap_mode)


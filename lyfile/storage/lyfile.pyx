import shutil
import threading
import numpy as np
import pandas as pd
import pyarrow as pa
from typing import Dict, List, Optional, Union
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor
from libc.stdint cimport int64_t
import pyarrow.parquet as pq
import logging

# Add necessary cimports
cimport numpy as np
from ..utils import fsst
from .mmap import MMapReader
from .vec_storage import VecStorage
from ..utils.array import ArrayView

# Define types
ctypedef np.int32_t INT32
ctypedef np.int64_t INT64
ctypedef np.float32_t FLOAT32
ctypedef np.float64_t FLOAT64
cdef int64_t file_offset

ctypedef tuple ColumnInfo
ctypedef dict BlocksInfo

cdef class LyFile:
    """
    Parquet-based columnar storage file format implementation.
    """
    cdef:
        public object folder_path
        public object filepath
        public int thread_count
        public object _executor
        public object _thread_local
        public object _parquet_folder
        public dict _vector_columns
        public int n_rows
        public list _column_order
        public int _partition_size
        public int _max_partitions
        public int _min_partition_size
        public object _merge_executor
        public object _merge_future
        public object _merge_lock

    def __init__(self, filepath: Union[str, Path], thread_count: int = None, 
                 partition_size: int = 100000, max_partitions: int = 10, 
                 min_partition_size: int = 10000, overwrite: bool = False):
        """Initialize the LyFile."""
        self.folder_path = Path(filepath)

        if self.folder_path.exists() and overwrite:
            shutil.rmtree(self.folder_path)
            
        self.folder_path.mkdir(parents=True, exist_ok=True)
        # 创建 vec_data 文件夹
        (self.folder_path / "vec_data").mkdir(parents=True, exist_ok=True)
        
        # 修改: 使用文件夹存储分区文件
        self._parquet_folder = self.folder_path / "data"
        self._parquet_folder.mkdir(exist_ok=True)
        self.filepath = self._parquet_folder
        
        self.thread_count = thread_count or min(8, os.cpu_count() * 2)
        self._executor = ThreadPoolExecutor(max_workers=self.thread_count)
        self._thread_local = threading.local()
        self._vector_columns = {}
        self._column_order = []
        self.n_rows = 0
        self._partition_size = partition_size
        self._max_partitions = max_partitions
        self._min_partition_size = min_partition_size
        
        self._merge_executor = ThreadPoolExecutor(max_workers=1)
        self._merge_future = None
        self._merge_lock = threading.Lock()
        
        if self._parquet_folder.exists():
            self._init_metadata()

    cdef void _init_metadata(self):
        """Initialize metadata from existing parquet files."""
        if not self._parquet_folder.exists():
            return
            
        # 读取所有分区文件的元数据
        self.n_rows = 0
        self._column_order = []
        
        # 获取所有分区文件
        parquet_files = sorted(self._parquet_folder.glob("part-*.ly"))
        if not parquet_files:
            return
            
        # 从第一个文件获取 schema
        first_file = pq.ParquetFile(parquet_files[0])
        self._column_order = first_file.schema.names
        
        # 计算总行数
        for parquet_file in parquet_files:
            metadata = pq.read_metadata(parquet_file)
            self.n_rows += metadata.num_rows
        
        # 处理向量列
        for col in self._column_order:
            vec_path = Path(self.folder_path).glob(f"{col}-*.vec")
            # 如果找不到对应的向量文件，忽略， 如果找到，使用相对路径，存储随机一个文件路径
            for vec_path in vec_path:
                relative_path = vec_path.relative_to(self.folder_path)
                self._vector_columns[col] = str(relative_path)
                break

    def write(self, data: Union[pd.DataFrame, dict]):
        """写入数据到文件"""
        # 转换输入数据为 DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
            
        # 分离向量列
        vector_columns = {}
        regular_columns = df.copy()
        
        for col in df.columns:
            if isinstance(df[col].iloc[0], np.ndarray):
                vector_columns[col] = df[col].values
                regular_columns.drop(col, axis=1, inplace=True)
                
                # 保存向量数据
                vec_storage = VecStorage(self.folder_path / "vec_data", col)
                vec_storage.save_vec(np.vstack(vector_columns[col]))
                
                # 存储相对路径
                self._vector_columns[col] = col  # 只存储列名

        # 将常规列写入 parquet 文件
        table = pa.Table.from_pandas(regular_columns)
        
        # 写入第一个分区文件
        parquet_path = self._parquet_folder / "part-00000.ly"
        pq.write_table(
            table, 
            parquet_path,
            compression='snappy',
            use_dictionary=True,
            version='2.6',
            data_page_version='2.0'
        )
        
        # 更新元数据
        self._column_order = list(df.columns)  # 包含所有列，包括向量列
        self.n_rows = len(df)

    def append(self, data: Union[List[Dict], pd.DataFrame, pd.Series, pa.Table]):
        """Append data to the file."""
        # 转换输入数据为 DataFrame
        if isinstance(data, (pd.Series, pa.Table)):
            df = pd.DataFrame(data)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
            
        # 检查列是否匹配
        if self._column_order and set(df.columns) != set(self._column_order):
            raise ValueError("新数据的列与现有列不匹配")
            
        # 分离并处理向量列
        vector_columns = {}
        regular_columns = df.copy()
        
        for col in df.columns:
            if isinstance(df[col].iloc[0], np.ndarray):
                vector_columns[col] = df[col].values
                regular_columns.drop(col, axis=1, inplace=True)
                
                # 追加向量数据
                vec_storage = VecStorage(self.folder_path / "vec_data", col)
                vec_storage.save_vec(np.vstack(vector_columns[col]))

        # 将常规列写入新的分区文件
        table = pa.Table.from_pandas(regular_columns)
        
        # 创建新的分区文件
        partition_files = list(self._parquet_folder.glob("part-*.ly"))
        partition_id = len(partition_files)
        parquet_path = self._parquet_folder / f"part-{partition_id:05d}.ly"
        
        # 写入数据
        pq.write_table(
            table, 
            parquet_path,
            compression='snappy',
            use_dictionary=True,
            version='2.6',
            data_page_version='2.0'
        )
        
        # 更新元数据
        self._init_metadata()
        
        # 异步触发合并操作
        self.trigger_merge()

    def read(self, columns: Union[Optional[List[str]], str] = None) -> pa.Table:
        """读取指定列的数据"""
        reader = self.mmap_reader()
        return reader.read(columns)

    def read_vec(self, column_name: str, mmap_mode: bool = True) -> Union[np.ndarray, ArrayView]:
        """读取向量数据"""
        reader = self.mmap_reader()
        return reader.read_vec(column_name, mmap_mode)

    def read_batch(self, batch_size: int = 1000, columns: Union[Optional[List[str]], str] = None):
        """批量读取数据的生成器"""
        reader = self.mmap_reader()
        yield from reader.read_batch(batch_size, columns)

    def __getitem__(self, key):
        """支持切片操作"""
        reader = self.mmap_reader()
        return reader[key]

    def __len__(self):
        """返回数据行数"""
        return self.n_rows

    def mmap_reader(self) -> MMapReader:
        """返回一个内存映射读取器"""
        return MMapReader(self._parquet_folder, thread_count=self.thread_count)

    def _merge_small_partitions(self):
        """合并小分区文件"""
        # 获取所有分区文件及其大小
        partitions = []
        for parquet_file in sorted(self._parquet_folder.glob("part-*.ly")):
            metadata = pq.read_metadata(parquet_file)
            partitions.append({
                'path': parquet_file,
                'rows': metadata.num_rows,
                'size': parquet_file.stat().st_size
            })
        
        if not partitions:
            return

        # 如果分区数量超过最大值或存在小分区，执行合并
        if (len(partitions) > self._max_partitions or 
            any(p['rows'] < self._min_partition_size for p in partitions)):
            
            # 读取所有需要合并的分区
            tables = []
            for partition in partitions:
                # 读取parquet文件
                table = pq.read_table(partition['path'])
                tables.append(table)
                
            # 合并所有表
            merged_table = pa.concat_tables(tables)
            
            # 创建临时文件夹用于存放新分区
            temp_folder = self._parquet_folder.parent / f"{self._parquet_folder.name}_temp"
            temp_folder.mkdir(exist_ok=True)
            
            try:
                # 重新分区并写入临时文件夹
                total_rows = len(merged_table)
                new_partitions = []
                for i in range(0, total_rows, self._partition_size):
                    end_idx = min(i + self._partition_size, total_rows)
                    partition_table = merged_table.slice(i, end_idx - i)
                    
                    # 写入新的分区文件
                    new_path = temp_folder / f"part-{i//self._partition_size:05d}.ly"
                    pq.write_table(
                        partition_table,
                        new_path,
                        compression='snappy',
                        use_dictionary=True,
                        version='2.6',
                        data_page_version='2.0'
                    )
                    new_partitions.append(new_path)
                
                # 获取锁以进行文件替换
                with self._merge_lock:
                    # 删除原有分区文件
                    for partition in partitions:
                        partition['path'].unlink()
                    
                    # 移动新分区文件到正式目录
                    for new_path in new_partitions:
                        target_path = self._parquet_folder / new_path.name
                        new_path.rename(target_path)
                    
                    # 更新元数据
                    self._init_metadata()
                    
            finally:
                # 清理临时文件夹
                if temp_folder.exists():
                    shutil.rmtree(temp_folder)
                    
            logging.info(f"Merged {len(partitions)} partitions into {len(new_partitions)} partitions")

    def _async_merge(self):
        """异步执行合并操作"""
        try:
            self._merge_small_partitions()
        except Exception as e:
            logging.error(f"Error during partition merge: {e}")
            raise

    def trigger_merge(self):
        """触发异步合并操作"""
        # 如果有正在进行的合并任务，检查其状态
        if self._merge_future and not self._merge_future.done():
            logging.info("Merge operation already in progress")
            return False
            
        # 启动新的合并任务
        self._merge_future = self._merge_executor.submit(self._async_merge)
        return True

    def optimize(self, wait: bool = True):
        """手动触发优化存储
        
        Args:
            wait: 是否等待优化完成
        """
        if self.trigger_merge() and wait:
            self._merge_future.result()  # 等待合并完成

    def __del__(self):
        """清理资源"""
        if self._merge_executor:
            self._merge_executor.shutdown(wait=True)

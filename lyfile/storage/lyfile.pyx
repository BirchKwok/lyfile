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

cimport numpy as np
from .reader import LyReader
from .vec_storage import VecStorage
from ..utils.array import ArrayView

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
        public object _reader
        
    def __init__(self, filepath: Union[str, Path], thread_count: int = None, 
                 partition_size: int = 100000, max_partitions: int = 10, 
                 min_partition_size: int = 10000, overwrite: bool = False):
        """Initialize the LyFile."""
        self.folder_path = Path(filepath)

        if self.folder_path.exists() and overwrite:
            shutil.rmtree(self.folder_path)
            
        self.folder_path.mkdir(parents=True, exist_ok=True)
        # Create vec_data folder
        (self.folder_path / "vec_data").mkdir(parents=True, exist_ok=True)
        
        # Use folder to store partition files
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
            
        # Read metadata from all partition files
        self.n_rows = 0
        self._column_order = []
        
        # Get all partition files
        parquet_files = sorted(self._parquet_folder.glob("part-*.ly"))
        if not parquet_files:
            return
            
        # Get schema from the first file
        first_file = pq.ParquetFile(parquet_files[0])
        regular_columns = first_file.schema.names
        
        # Process vector columns
        vector_columns = []
        for vec_file in (self.folder_path / "vec_data").glob("*.vec"):
            col_name = vec_file.stem.split('-')[0]  # 从文件名获取列名
            vector_columns.append(col_name)
            self._vector_columns[col_name] = col_name
        
        # Combine regular columns and vector columns
        self._column_order = regular_columns + vector_columns
        
        # Calculate total rows
        for parquet_file in parquet_files:
            metadata = pq.read_metadata(parquet_file)
            self.n_rows += metadata.num_rows

    def write(self, data: Union[pd.DataFrame, dict, pa.Table]):
        """Write data to file.
        
        Parameters:
            data (Union[pd.DataFrame, dict, pa.Table]): Data to write.
        """
        # Convert input data to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, pa.Table):
            df = data.to_pandas()
        else:
            df = data.copy()

        # overwrite
        for ly_file in self._parquet_folder.iterdir():
            ly_file.unlink()
        for vec_file in (self.folder_path / "vec_data").iterdir():
            vec_file.unlink()

        # Separate vector columns
        vector_columns = {}
        regular_columns = df.copy()
        
        for col in df.columns:
            if isinstance(df[col].iloc[0], np.ndarray):
                vector_columns[col] = df[col].values
                regular_columns.drop(col, axis=1, inplace=True)
                
                # Save vector data
                vec_storage = VecStorage(self.folder_path / "vec_data", col)
                vec_storage.save_vec(np.vstack(vector_columns[col]))
                
                # Store relative path
                self._vector_columns[col] = col  # Only store column name

        # Write regular columns to parquet file
        table = pa.Table.from_pandas(regular_columns)

        # Write to the first partition file
        parquet_path = self._parquet_folder / "part-00001.ly"
        pq.write_table(
            table, 
            parquet_path,
            compression='snappy',
            use_dictionary=True,
            version='2.6',
            data_page_version='2.0'
        )
        
        # Update metadata
        self._column_order = list(df.columns)  # Include all columns, including vector columns
        self.n_rows = len(df)

    def append(self, data: Union[List[Dict], pd.DataFrame, pd.Series, pa.Table]):
        """Append data to the file.
        
        Parameters:
            data (Union[List[Dict], pd.DataFrame, pd.Series, pa.Table]): Data to append.
        """
        # Convert input data to DataFrame
        if isinstance(data, (pd.Series, pa.Table)):
            df = pd.DataFrame(data)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
            
        # Check if columns match
        if self._column_order and set(df.columns) != set(self._column_order):
            print(set(df.columns), set(self._column_order))
            raise ValueError("New data columns do not match existing columns")
            
        # Separate and process vector columns
        vector_columns = {}
        regular_columns = df.copy()
        
        for col in df.columns:
            if isinstance(df[col].iloc[0], np.ndarray):
                vector_columns[col] = df[col].values
                regular_columns.drop(col, axis=1, inplace=True)
                
                # Append vector data
                vec_storage = VecStorage(self.folder_path / "vec_data", col)
                vec_storage.save_vec(np.vstack(vector_columns[col]))

        # Write regular columns to new partition file
        table = pa.Table.from_pandas(regular_columns)
        
        # Create new partition file
        partition_files = list(self._parquet_folder.glob("part-*.ly"))
        partition_id = len(partition_files) + 1
        parquet_path = self._parquet_folder / f"part-{partition_id:05d}.ly"
        
        # Write data
        pq.write_table(
            table, 
            parquet_path,
            compression='snappy',
            use_dictionary=True,
            version='2.6',
            data_page_version='2.0'
        )
        
        # Update metadata
        self._init_metadata()
        
        # Trigger merge operation asynchronously
        self.trigger_merge()

    def read(self, columns: Union[Optional[List[str]], str] = None, exclude_vec: bool = True) -> pa.Table:
        """Read data from specified columns.
        
        Parameters:
            columns (Union[Optional[List[str]], str]): Columns to read.
            exclude_vec (bool): Whether to not read vector columns.

        Returns:
            pa.Table: Data.
        """
        reader = LyReader(self._parquet_folder, thread_count=self.thread_count)
        return reader.read(columns, exclude_vec=exclude_vec)

    def read_vec(self, column_name: str, mmap_mode: bool = True) -> Union[np.ndarray, ArrayView]:
        """Read vector data.
        
        Parameters:
            column_name (str): Column name.
            mmap_mode (bool): Whether to use memory-mapped mode.

        Returns:
            Union[np.ndarray, ArrayView]: Vector data.
        """
        reader = LyReader(self._parquet_folder, thread_count=self.thread_count)
        return reader.read_vec(column_name, mmap_mode)

    def read_batch(self, batch_size: int = 1000, columns: Union[Optional[List[str]], str] = None, exclude_vec: bool = True):
        """Read data in batches.
        
        Parameters:
            batch_size (int): Batch size.
            columns (Union[Optional[List[str]], str]): Columns to read.
            exclude_vec (bool): Whether to not read vector columns.

        Yields:
            pa.Table: Batch of data.
        """
        reader = LyReader(self._parquet_folder, thread_count=self.thread_count)
        yield from reader.read_batch(batch_size, columns, exclude_vec=exclude_vec)

    def __getitem__(self, key):
        """Support slicing operation.
        
        Parameters:
            key: Slicing key.

        Returns:
            pa.Table: Sliced data.
        """
        reader = LyReader(self._parquet_folder, thread_count=self.thread_count)
        return reader[key]

    def __len__(self):
        """Return number of rows"""
        return self.n_rows

    def _merge_small_partitions(self):
        """Merge small partition files"""
        # Get all partition files and their sizes
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

        # If partition count exceeds maximum or there are small partitions, perform merge
        if (len(partitions) > self._max_partitions or 
            any(p['rows'] < self._min_partition_size for p in partitions)):
            
            # Read all partitions to be merged
            tables = []
            for partition in partitions:
                # Read parquet file
                table = pq.read_table(partition['path'])
                tables.append(table)
                
            # Merge all tables
            merged_table = pa.concat_tables(tables)
            
            # Create temporary folder to store new partitions
            temp_folder = self._parquet_folder.parent / f"{self._parquet_folder.name}_temp"
            temp_folder.mkdir(exist_ok=True)
            
            try:
                # Re-partition and write to temporary folder
                total_rows = len(merged_table)
                new_partitions = []
                for i in range(0, total_rows, self._partition_size):
                    end_idx = min(i + self._partition_size, total_rows)
                    partition_table = merged_table.slice(i, end_idx - i)
                    
                    # Write to new partition file
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
                
                # Get lock to replace files
                with self._merge_lock:
                    # Delete existing partition files
                    for partition in partitions:
                        partition['path'].unlink()
                    
                    # Move new partition files to official directory
                    for new_path in new_partitions:
                        target_path = self._parquet_folder / new_path.name
                        new_path.rename(target_path)
                    
                    # Update metadata
                    self._init_metadata()
                    
            finally:
                # Clean up temporary folder
                if temp_folder.exists():
                    shutil.rmtree(temp_folder)
                    
            logging.info(f"Merged {len(partitions)} partitions into {len(new_partitions)} partitions")

    def _async_merge(self):
        """Asynchronously execute merge operation"""
        try:
            self._merge_small_partitions()
        except Exception as e:
            logging.error(f"Error during partition merge: {e}")
            raise

    def trigger_merge(self):
        """Trigger asynchronous merge operation"""
        # If there is an ongoing merge task, check its status
        if self._merge_future and not self._merge_future.done():
            logging.info("Merge operation already in progress")
            return False
            
        # Start new merge task
        self._merge_future = self._merge_executor.submit(self._async_merge)
        return True

    def optimize(self, wait: bool = True):
        """Manually trigger optimization.
        
        Parameters:
            wait (bool): Whether to wait for optimization to complete
        """
        if self.trigger_merge() and wait:
            self._merge_future.result()  # Wait for merge to complete

    def __del__(self):
        """Clean up resources"""
        if self._merge_executor:
            self._merge_executor.shutdown(wait=True)

    def __repr__(self):
        """Return string representation"""
        return f"LyFile(\n    n_rows={self.n_rows},\n    columns={self._column_order}\n)"
    
    def __str__(self):
        """Return string representation"""
        return self.__repr__()

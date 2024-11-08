import pyarrow as pa
from typing import Union, Optional, List
from pathlib import Path
from libc.stdint cimport int64_t
import pyarrow.parquet as pq
import pandas as pd
import pyarrow.dataset as ds
import numpy as np

cimport numpy as np

from .vec_storage import VecStorage
from ..utils.array import ArrayView

ctypedef np.int32_t INT32
ctypedef np.int64_t INT64
ctypedef np.float32_t FLOAT32
ctypedef np.float64_t FLOAT64
cdef int64_t file_offset


class TableWrapper:
    """Duck-type class wrapping PyArrow Table."""
    def __init__(self, table: pa.Table):
        # Use object.__setattr__ to avoid recursive call
        object.__setattr__(self, '_table', table)
        
        # Get all public attributes and methods of the table
        table_attrs = set(dir(table))
        # Exclude methods and attributes we implemented ourselves and private attributes
        exclude_attrs = {'to_pandas', '__class__', '__dict__', '__weakref__', '__new__', '__init__'}
        proxy_attrs = table_attrs - exclude_attrs
        
        # Proxy all attributes to self._table
        for attr in proxy_attrs:
            if not hasattr(self, attr):
                setattr(self.__class__, attr, property(lambda self, attr=attr: getattr(self._table, attr)))

    def __getattribute__(self, name):
        """Handle all attribute access"""
        try:
            # First try to get the object's own attribute
            return object.__getattribute__(self, name)
        except AttributeError:
            # If the attribute does not exist, try to get it from _table
            return getattr(object.__getattribute__(self, '_table'), name)

    def to_pandas(self) -> pd.DataFrame:
        """Convert to Pandas DataFrame"""
        cols = self._table.column_names
        
        chunk_cols = []
        regular_cols = []
        
        # Check the first element of each column
        for col in cols:
            if len(self._table[col]) > 0 and isinstance(self._table[col].chunk(0), pa.FixedSizeListArray):
                chunk_cols.append(col)
            else:
                regular_cols.append(col)

        # Process vector columns
        chunk_data = {col: col_data.to_pandas() for col, col_data in zip(chunk_cols, self._table.select(chunk_cols))}
        
        # Create DataFrame with vectors
        vec_df = pd.DataFrame(data=chunk_data, columns=chunk_cols)
        
        # Process regular columns
        if regular_cols:
            regular_df = self._table.select(regular_cols).to_pandas()
            result = pd.concat([vec_df, regular_df], axis="columns")
        else:
            result = vec_df

        # Ensure column order matches the original table
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
        # Get all partition files
        self._parquet_files = sorted(self.folder_path.glob("part-*.ly"))
        
        if not self._parquet_files:
            return
            
        # Get schema from the first file
        first_file = pq.ParquetFile(self._parquet_files[0])
        self._column_order = first_file.schema.names
        
        # Calculate total row count
        for parquet_file in self._parquet_files:
            metadata = pq.read_metadata(parquet_file)
            self.n_rows += metadata.num_rows

        # Check all possible vector files
        for vec_file in self.vec_path.glob("*.vec"):
            col_name = vec_file.stem  # Get file name (without extension)
            col_name = col_name.split("-")[0]
            self._vector_file_paths[col_name] = vec_file
            if col_name not in self._column_order:
                self._column_order.append(col_name)
                
    def read(self, columns: Union[Optional[List[str]], str] = None) -> pa.Table:
        """Read data from specified columns.
        
        Parameters:
            columns (Union[Optional[List[str]], str]): Columns to read.

        Returns:
            pa.Table: Data.
        """
        if isinstance(columns, str):
            columns = [columns]
        elif columns is None:
            columns = self._column_order
            
        # Separate vector columns and regular columns
        vector_cols = [col for col in columns if col in self._vector_file_paths]
        regular_cols = [col for col in columns if col not in self._vector_file_paths]
        
        # Read partition data using pyarrow.dataset
        if regular_cols:
            dataset = ds.dataset(
                self.folder_path,
                format='parquet',
                partitioning=None  # If there is a specific partition scheme, specify it here
            )
            table = dataset.to_table(columns=regular_cols)
        else:
            table = pa.Table.from_pandas(pd.DataFrame(index=range(self.n_rows)))

        # Read vector columns
        for col in vector_cols:
            vec_storage = VecStorage(self.vec_path, col)
            _arrays = vec_storage.load_vec(mmap_mode=True)
            vectors_col_num = _arrays.shape[1]
            flat_data = _arrays.reshape(-1)
            arrow_array = pa.array(flat_data, type=pa.float64())
            vector_array = pa.FixedSizeListArray.from_arrays(arrow_array, list_size=vectors_col_num)
            table = table.append_column(col, vector_array)
            
        # Ensure column order matches the requested order
        table = table.select(columns)
        return TableWrapper(table)

    def read_batch(self, batch_size: int = 1000, columns: Union[Optional[List[str]], str] = None):
        """Generator for batch reading data.
        
        Parameters:
            batch_size (int): Batch size.
            columns (Union[Optional[List[str]], str]): Columns to read.

        Yields:
            pa.Table: Data.
        """
        if isinstance(columns, str):
            columns = [columns]
        elif columns is None:
            columns = self._column_order
            
        # Separate vector columns and regular columns
        vector_cols = [col for col in columns if col in self._vector_file_paths]
        regular_cols = [col for col in columns if col not in self._vector_file_paths]
        
        # Preload all vector data
        vector_data = {}
        for col in vector_cols:
            vec_storage = VecStorage(self.vec_path, col)
            vector_data[col] = vec_storage.load_vec(mmap_mode=True)

        # Batch read using pyarrow.dataset
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
                
                # Add vector columns
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
            # If only vector columns, generate data in batches
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
        """Support slicing and fancy indexing.
        
        Parameters:
            key: Index.

        Returns:
            pa.Table: Data.
        """
        # Process column name indexing
        if isinstance(key, (str, list)) and (isinstance(key, str) or isinstance(key[0], str)):
            return self.read(columns=key)
        
        # Calculate actual row index range
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
            raise TypeError(f"Unsupported index type: {type(key)}")
        
        # Read regular columns and vector columns separately
        regular_cols = [col for col in self._column_order if col not in self._vector_file_paths]
        if regular_cols:
            dataset = ds.dataset(
                self.folder_path,
                format='parquet',
                partitioning=None
            )
            # Use filter and take to read only the needed rows
            scanner = dataset.scanner(columns=regular_cols)
            table = scanner.take(row_indices)
        else:
            table = pa.table([])
        
        # Read vector columns and add to table
        for col in self._vector_file_paths:
            vec_storage = VecStorage(self.vec_path, col)
            # Use mmap_mode to implement efficient slicing
            # Read only the needed rows
            _arrays = vec_storage[row_indices]
            vectors_col_num = _arrays.shape[1]
            flat_data = _arrays.reshape(-1)
            arrow_array = pa.array(flat_data, type=pa.float64())
            vector_array = pa.FixedSizeListArray.from_arrays(arrow_array, list_size=vectors_col_num)
            table = table.append_column(col, vector_array)
        
        return TableWrapper(table)

    def __len__(self):
        """Return the number of rows."""
        return self.n_rows
    
    def read_vec(self, column_name: str, mmap_mode: bool = False) -> Union[np.ndarray, ArrayView]:
        """Read vector data.
        
        Parameters:
            column_name (str): Vector column name.
            mmap_mode (bool): Whether to use memory mapping to read the file.

        Returns:
            numpy.ndarray: Vector data array.
            
        Raises:
            KeyError: If the column name does not exist or is not a vector column.
        """
        if column_name not in self._vector_file_paths:
            raise KeyError(f"Column '{column_name}' is not a vector column or does not exist.")
            
        vec_storage = VecStorage(self.vec_path, column_name)
        return vec_storage.load_vec(mmap_mode=mmap_mode)


from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import pyarrow as pa

from .._lib_lyfile import _LyFile


class LyFile:
    """LyFile is a high-performance binary file format designed for storing and retrieving large-scale data.
    """
    def __init__(self, path: str):
        """
        Initializes a LyFile object.

        Args:
            path (str): Path to the LyFile.
        """
        self._inner = _LyFile(path)

    def write(self, tdata: Optional[Union[pd.DataFrame, dict, pa.Table]] = None, vdata: Optional[Dict[str, np.ndarray]] = None):
        """
        Writes data to the file.

        Args:
            tdata (Optional[Union[pandas.DataFrame, dict, pyarrow.Table]]):
                The table data to be written.
                Supported types include:
                - Pandas DataFrame
                - Python dictionary
                - PyArrow Table
                If None, only vector data will be written.
            vdata (Optional[dict]):
                Dictionary of vector data where keys are vector names and values are numpy arrays.
                If None, only table data will be written.

        Raises:
            ValueError: If both tdata and vdata are None, or if input types are not supported.
    """
        self._inner.write(tdata=tdata, vdata=vdata)

    def append(self, tdata: Optional[Union[pd.DataFrame, dict, pa.Table]] = None, vdata: Optional[Dict[str, np.ndarray]] = None):
        """
        Append data to the file.

        Args:
            tdata (Optional[Union[pandas.DataFrame, dict, pyarrow.Table]]):
                The table data to be written.
                Supported types include:
                - Pandas DataFrame
                - Python dictionary
                - PyArrow Table
                If None, only vector data will be written.
            vdata (Optional[dict]):
                Dictionary of vector data where keys are vector names and values are numpy arrays.
                If None, only table data will be written.

        Raises:
            ValueError: If both tdata and vdata are None, or if input types are not supported.
    """
        self._inner.append(tdata=tdata, vdata=vdata)

    def get_vec_shape(self, name: str):
        """Returns the shape of a stored vector.

        Args:
            name (str): Name of the vector

        Returns:
            tuple: Shape of the vector
        """
        return self._inner.get_vec_shape(name)
    
    def list_columns(self):
        """Returns a list of column names in the file.
        """
        return self._inner.list_columns()
    
    def list_vectors(self):
        """Returns a list of vector names in the file.
        """
        return self._inner.list_vectors()
    
    def read(self, columns: Optional[Union[str, List[str]]] = None, load_mmap_vec: bool = False):
        """Reads data from the file.

        Args:
            columns (Optional[Union[str, List[str]]]): 
                For table data: The names of columns to read. If None, all columns are read.
                For vector data: The name of the vector to read.
            load_mmap_vec (bool): Whether to use numpy's memmap to read vector data.

        Returns:
            LyDataView: 包含表格数据和向量数据的读取结果
                属性:
                    - all_entries: 包含表格数据和向量数据的读取结果
                    - columns_list: 列名列表
                    - table_data: 表格数据
                    - vector_data: 向量数据
        """
        return self._inner.read(columns=columns, load_mmap_vec=load_mmap_vec)

    def search_vector(self, emb_name: str, queries: np.ndarray, top_k: int = 10, metric: str = "l2"):
        """Searches for nearest neighbors in the specified vector column.

        Args:
            emb_name (str): The name of the vector column to search.
            queries (numpy.ndarray): Query vectors, shape is (n_queries, dim)
            top_k (int): Number of nearest neighbors to return
            metric (str): Distance metric, supported metrics are "l2", "ip" (inner product), "cosine"

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: (indices, distances)
            indices shape is (n_queries, top_k), distances shape is (n_queries, top_k)
        """
        return self._inner.search_vector(vector_name=emb_name, query_vectors=queries, top_k=top_k, metric=metric)
    
    @property
    def shape(self):
        """Returns the shape of the file.
        """
        return self._inner.shape
    
    def __getitem__(self, key):
        """
        Get item from the file.
        
        Args:
            key: 可以是以下类型:
                - int: 返回单行数据
                - str: 返回指定列的所有数据
                - slice: 返回指定范围的数据
                - list/ndarray: 包含整数索引或列名的列表

        Returns:
            tuple or dict or pyarrow.Table: 
                - 如果向量数据为空，则返回表格数据（pyarrow.Table）
                - 如果向量数据不为空，则返回包含两个元素的元组（tuple）
                    - 第一个元素: 表格数据（pyarrow.Table）
                    - 第二个元素: 向量数据（dict）
                - 如果表格数据为空，则返回向量数据（dict）
        """
        def _read_vectors(indices):
            """辅助函数：读取向量数据"""
            _ = self._inner.read(vector_columns, load_mmap_vec=True).vector_data
            if len(vector_columns) == 1:
                result[1][vector_columns[0]] = _[vector_columns[0]][indices]
            else:
                for vec_name in vector_columns:
                    result[1][vec_name] = _[vec_name][indices]

        vector_columns = [k for k, v in self._inner.list_columns().items() if v["lytype"] == "vector"]
        result = [None, {}]

        if isinstance(key, int):
            result[0] = self._inner.read_rows([key])
            _read_vectors(key)
            return tuple(result)
        
        elif isinstance(key, str):
            return self.read(columns=[key], load_mmap_vec=True).all_entries
        
        elif isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else len(self)
            step = key.step if key.step is not None else 1
            indices = list(range(start, stop, step))
            
            result[0] = self._inner.read_rows(indices)
            _read_vectors(indices)
            return tuple(result)
        
        elif isinstance(key, (list, np.ndarray)):
            if all(isinstance(k, int) for k in key):
                result[0] = self._inner.read_rows(key)
                _read_vectors(key)
                return tuple(result)
            elif all(isinstance(k, str) for k in key):
                return self.read(columns=key, load_mmap_vec=True).all_entries
            else:
                raise ValueError("Unsupported index type. Only int or str are supported.")
        
        else:
            raise ValueError(f"Unsupported index type: {type(key)}")

    def __len__(self):
        return self._inner.shape[0]

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
            Union[pyarrow.Table, numpy.ndarray]: 
                - If reading table columns: returns a pyarrow Table
                - If reading a vector: returns a numpy array
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
    


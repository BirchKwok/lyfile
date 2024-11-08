import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..utils.nnp import load_nnp, save_nnp
from ..utils.array import ArrayView


cdef class VecStorage:
    """Vector storage class."""
    cdef:
        public object root_path
        public object column_name
        public object max_rows
        public object suffix
        public dict vec_index_range

    def __init__(self, root_path: Path, column_name: str):
        """
        Initialize vector storage.

        Parameters:
            root_path (Pathlike): The root path of the vector storage.
            column_name (str): The column name of the vector storage.
        """
        self.root_path = root_path
        self.column_name = column_name
        self.max_rows = 1000000
        self.suffix = "vec"

        # initial vec index range
        # such as {0: (0, 1000000), 1: (1000000, 2000000), ...}
        self.vec_index_range = {}
        self._init_vec_index_range()

    def _init_vec_index_range(self):
        """Initial vec index range."""
        vec_files = sorted(self.root_path.glob(f"{self.column_name}*.{self.suffix}"), 
                           key=lambda x: int(x.stem.split("-")[-1]))
        for i, vec_file in enumerate(vec_files):
            if i == 0:
                self.vec_index_range[i] = (0, load_nnp(str(vec_file), mmap_mode=True).shape[0] - 1)
            else:
                self.vec_index_range[i] = (
                    self.vec_index_range[i - 1][1] + 1, 
                    self.vec_index_range[i - 1][1] + load_nnp(str(vec_file), mmap_mode=True).shape[0] - 1
                )

    def _get_partition_id(self, index: int):
        """Get partition ID for specified index.
        
        Parameters:
            index (int): Index.

        Returns:
            int: Partition ID.
        """
        for partition_id, vec_index_range in self.vec_index_range.items():
            if vec_index_range[0] <= index <= vec_index_range[1]:
                return partition_id
        raise ValueError(f"Index {index} out of range")

    def _get_vec_file_index(self, external_index: int):
        """Get vector file index for specified external index.
        
        Parameters:
            external_index (int): External index.

        Returns:
            int: Vector file index.
        """
        for partition_id, vec_index_range in self.vec_index_range.items():
            if vec_index_range[0] <= external_index <= vec_index_range[1]:
                return external_index - vec_index_range[0]

        raise ValueError(f"External index {external_index} out of range")

    def save_vec(self, vec: np.ndarray):
        """Save vector.
        
        Parameters:
            vec (np.ndarray): Vector.
        """
        vec_files = sorted(self.root_path.glob(f"{self.column_name}*.{self.suffix}"), 
                           key=lambda x: int(x.stem.split("-")[-1]))
        partition_id = len(vec_files)
        
        if vec_files:
            last_file_rows = load_nnp(str(vec_files[-1])).shape[0]
            last_file_path = vec_files[-1]
        else:
            last_file_rows = 0
            last_file_path = None
        
        if (last_file_path is not None) and (last_file_rows < self.max_rows):
            filling_rows = self.max_rows - last_file_rows
            to_fill_vec = vec[:filling_rows]
            save_nnp(str(last_file_path), to_fill_vec, append=True)
            vec = vec[filling_rows:]

        while vec.shape[0] > 0:
            if partition_id == 0:
                partition_id = 1
            else:
                partition_id += 1

            vec_path = self.root_path / f"{self.column_name}-{partition_id:05d}.{self.suffix}"
            save_nnp(str(vec_path), vec[:self.max_rows], append=False)
            vec = vec[self.max_rows:]
        
    
    def _load_vec_file(self, vec_file: Path, mmap_mode: bool = False):
        """Load vector file.
        
        Parameters:
            vec_file (Path): Vector file.
            mmap_mode (bool): Whether to use memory mapping to read the file.

        Returns:
            np.ndarray: Vector data.
            int: Partition ID.
        """
        return load_nnp(str(vec_file), mmap_mode=mmap_mode), int(vec_file.stem.split("-")[-1])
    
    def load_vec(self, mmap_mode: bool = False):
        """Load vector.
        
        Parameters:
            mmap_mode (bool): Whether to use memory mapping to read the file.

        Returns:
            np.ndarray: Vector data.
        """
        vec_files = sorted(self.root_path.glob(f"{self.column_name}*.{self.suffix}"), 
                           key=lambda x: int(x.stem.split("-")[-1]))
        if not vec_files:
            return np.array([])

        # load all vec files
        if len(vec_files) == 1:
            return self._load_vec_file(vec_files[0], mmap_mode)[0]

        results = []
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
            results = list(executor.map(lambda x: self._load_vec_file(x, mmap_mode), vec_files))
        
        # sort by partition_id
        results.sort(key=lambda x: x[1])

        total_rows = sum(vec.shape[0] for vec, _ in results)
        vector_dim = results[0][0].shape[1]
        return ArrayView([vec for vec, _ in results], total_rows, vector_dim)

    def __len__(self):
        """Return the number of vectors."""
        vec_files = sorted(self.root_path.glob(f"{self.column_name}*.{self.suffix}"), 
                           key=lambda x: int(x.stem.split("-")[-1]))
        return len(vec_files)

    def vector_shape(self):
        """Return the shape of vectors.
        
        Returns:
            tuple: Shape of vectors.
        """
        return self.load_vec(mmap_mode=True).shape

    def __getitem__(self, index):
        """Get vector for specified index.
        
        Supported index types:
        - int: Single index
        - slice: start:stop:step
        - list: [1, 2, 3]
        - numpy array: array([1, 2, 3])
        - boolean array: array([True, False, True])
        """
        total_len = len(self)
        
        # Process single integer index
        if isinstance(index, (int, np.integer)):
            if index < 0:
                index += total_len
            if not 0 <= index < total_len:
                raise IndexError("Index out of range")

            partition_id = self._get_partition_id(index)
            vec_file = self.root_path / f"{self.column_name}-{partition_id:05d}.{self.suffix}"
            vec_index = self._get_vec_file_index(index)
            return load_nnp(str(vec_file), mmap_mode=True)[vec_index]
        
        # Process slice
        elif isinstance(index, slice):
            start, stop, step = index.indices(total_len)
            
            # Optimize: Process slice by partition
            results = []
            current_idx = start
            
            while current_idx < stop:
                partition_id = self._get_partition_id(current_idx)
                vec_file = self.root_path / f"{self.column_name}-{partition_id:05d}.{self.suffix}"
                vec_data = load_nnp(str(vec_file), mmap_mode=True)
                
                # Calculate current partition range
                partition_start, partition_end = self.vec_index_range[partition_id]
                
                # Calculate slice range in current partition
                local_start = self._get_vec_file_index(current_idx)
                local_stop = min(
                    self._get_vec_file_index(min(stop - 1, partition_end)) + 1,
                    vec_data.shape[0]
                )
                
                # Extract data from current partition
                partition_indices = range(local_start, local_stop, step)
                if partition_indices:
                    results.append(vec_data[partition_indices])
                
                # Update index to next partition start position
                current_idx = partition_end + 1
                if step > 1:
                    # Adjust to next valid position of step
                    offset = (current_idx - start) % step
                    if offset:
                        current_idx += (step - offset)
            
            return np.vstack(results) if results else np.array([])
        
        # Process list, numpy array, etc. iterable objects
        elif isinstance(index, (list, np.ndarray)):
            if isinstance(index, np.ndarray) and index.dtype == bool:
                # Process boolean index
                if len(index) != total_len:
                    raise IndexError("Boolean index length must be the same as the number of vectors")
                indices = np.where(index)[0]
            else:
                # Process integer index array
                indices = np.asarray(index)
                # Process negative index
                indices = np.where(indices < 0, indices + total_len, indices)
                if np.any((indices < 0) | (indices >= total_len)):
                    raise IndexError("Index out of range")
            
            # Optimize: Batch read by partition
            partition_groups = {}
            for idx in indices:
                partition_id = self._get_partition_id(idx)
                if partition_id not in partition_groups:
                    partition_groups[partition_id] = []
                partition_groups[partition_id].append(idx)
            
            # Read by partition and assemble results
            results = []
            for partition_id, group_indices in sorted(partition_groups.items()):
                vec_file = self.root_path / f"{self.column_name}-{partition_id:05d}.{self.suffix}"
                vec_data = load_nnp(str(vec_file), mmap_mode=True)
                local_indices = [self._get_vec_file_index(idx) for idx in group_indices]
                results.append(vec_data[local_indices])
                    
            return np.vstack(results)
        
        else:
            raise TypeError(f"Unsupported index type: {type(index)}")
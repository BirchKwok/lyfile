import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..utils.nnp import load_nnp, save_nnp
from ..utils.array import ArrayView


cdef class VecStorage:
    """向量调度器"""
    cdef:
        public object root_path
        public object column_name
        public object max_rows
        public object suffix
        public dict vec_index_range

    def __init__(self, root_path: Path, column_name: str):
        self.root_path = root_path
        self.column_name = column_name
        self.max_rows = 1000000
        self.suffix = "vec"

        # initial vec index range
        # such as {0: (0, 1000000), 1: (1000000, 2000000), ...}
        self.vec_index_range = {}
        self._init_vec_index_range()

    def _init_vec_index_range(self):
        """初始化向量索引范围"""
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
        """获取指定索引的分区ID"""
        for partition_id, vec_index_range in self.vec_index_range.items():
            if vec_index_range[0] <= index <= vec_index_range[1]:
                return partition_id
        raise ValueError(f"索引 {index} 超出范围")

    def _get_vec_file_index(self, external_index: int):
        """获取指定外部索引的向量文件索引"""
        for partition_id, vec_index_range in self.vec_index_range.items():
            if vec_index_range[0] <= external_index <= vec_index_range[1]:
                return external_index - vec_index_range[0]

        raise ValueError(f"外部索引 {external_index} 超出范围")

    def save_vec(self, vec: np.ndarray):
        """保存向量"""
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
        """加载向量文件"""
        return load_nnp(str(vec_file), mmap_mode=mmap_mode), int(vec_file.stem.split("-")[-1])
    
    def load_vec(self, mmap_mode: bool = False):
        """加载向量"""
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
        """返回向量数量"""
        vec_files = sorted(self.root_path.glob(f"{self.column_name}*.{self.suffix}"), 
                           key=lambda x: int(x.stem.split("-")[-1]))
        return len(vec_files)

    def vector_shape(self):
        """返回向量形状"""
        return self.load_vec(mmap_mode=True).shape

    def __getitem__(self, index):
        """获取指定索引的向量，支持花样索引
        
        支持的索引类型：
        - 整数：单个索引
        - 切片：start:stop:step
        - 列表：[1, 2, 3]
        - numpy数组：array([1, 2, 3])
        - 布尔数组：array([True, False, True])
        """
        total_len = len(self)
        
        # 处理单个整数索引
        if isinstance(index, (int, np.integer)):
            if index < 0:
                index += total_len
            if not 0 <= index < total_len:
                raise IndexError("索引超出范围")

            partition_id = self._get_partition_id(index)
            vec_file = self.root_path / f"{self.column_name}-{partition_id:05d}.{self.suffix}"
            vec_index = self._get_vec_file_index(index)
            return load_nnp(str(vec_file), mmap_mode=True)[vec_index]
        
        # 处理切片
        elif isinstance(index, slice):
            start, stop, step = index.indices(total_len)
            
            # 优化：按分区分组处理切片
            results = []
            current_idx = start
            
            while current_idx < stop:
                partition_id = self._get_partition_id(current_idx)
                vec_file = self.root_path / f"{self.column_name}-{partition_id:05d}.{self.suffix}"
                vec_data = load_nnp(str(vec_file), mmap_mode=True)
                
                # 计算当前分区的范围
                partition_start, partition_end = self.vec_index_range[partition_id]
                
                # 计算在当前分区内的切片范围
                local_start = self._get_vec_file_index(current_idx)
                local_stop = min(
                    self._get_vec_file_index(min(stop - 1, partition_end)) + 1,
                    vec_data.shape[0]
                )
                
                # 提取当前分区的数据
                partition_indices = range(local_start, local_stop, step)
                if partition_indices:
                    results.append(vec_data[partition_indices])
                
                # 更新索引到下一个分区的起始位置
                current_idx = partition_end + 1
                if step > 1:
                    # 调整到step的下一个有效位置
                    offset = (current_idx - start) % step
                    if offset:
                        current_idx += (step - offset)
            
            return np.vstack(results) if results else np.array([])
        
        # 处理列表、numpy数组等可迭代对象
        elif isinstance(index, (list, np.ndarray)):
            if isinstance(index, np.ndarray) and index.dtype == bool:
                # 处理布尔索引
                if len(index) != total_len:
                    raise IndexError("布尔索引长度必须与向量数量相同")
                indices = np.where(index)[0]
            else:
                # 处理整数索引数组
                indices = np.asarray(index)
                # 处理负索引
                indices = np.where(indices < 0, indices + total_len, indices)
                if np.any((indices < 0) | (indices >= total_len)):
                    raise IndexError("索引超出范围")
            
            # 优化：按分区分组批量读取
            partition_groups = {}
            for idx in indices:
                partition_id = self._get_partition_id(idx)
                if partition_id not in partition_groups:
                    partition_groups[partition_id] = []
                partition_groups[partition_id].append(idx)
            
            # 按分区读取并组装结果
            results = []
            for partition_id, group_indices in sorted(partition_groups.items()):
                vec_file = self.root_path / f"{self.column_name}-{partition_id:05d}.{self.suffix}"
                vec_data = load_nnp(str(vec_file), mmap_mode=True)
                local_indices = [self._get_vec_file_index(idx) for idx in group_indices]
                results.append(vec_data[local_indices])
                    
            return np.vstack(results)
        
        else:
            raise TypeError(f"不支持的索引类型: {type(index)}")
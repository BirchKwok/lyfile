import os
import time
import shutil
import threading
import msgpack
import pandas as pd
import pyarrow as pa
import duckdb
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor

from ..utils.bitset import BitSet
from lyfile.storage.ly_file import LyFile, MMapReader


class MMapReaderContext:
    """MMapReader的上下文管理器"""
    
    def __init__(self, storage, columns: Optional[List[str]] = None):
        """
        Initialize the context manager.

        Parameters:
            storage (LyStorage): The LyStorage instance.
            columns (Optional[List[str]]): The columns to read.
        """
        self.storage = storage
        self.columns = columns
        self.readers = []

    def __enter__(self):
        """
        Enter the context and create and return all active regions' readers.

        Returns:
            pd.DataFrame: The merged dataframe
        """
        active_regions = [r for r in self.storage.regions if r['row_count'] > 0]
        dfs = []
        
        for region in active_regions:
            reader = MMapReader(self.storage.data_path / region['data_file'])
            self.readers.append(reader)
            
            df = reader.read(self.columns)
            if region['bitset'] is not None:
                active_indices = [i for i in range(len(df)) if region['bitset'].get_bit(i)]
                if active_indices:
                    df = df.iloc[active_indices]
                    dfs.append(df)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and close all active regions' readers.
        """
        for reader in self.readers:
            reader.close()


class LyStorage:
    """Storage class for LyFile, supporting multiple regions and lazy loading."""
    
    def __init__(self, data_path: str, 
                 schema: Optional[dict] = None,
                 lazy_load: bool = True, 
                 overwrite: bool = False,
                 max_rows_per_region: int = 1_000_000,
                 auto_optimize_interval: int = 3600,
                 auto_vacuum_threshold: float = 0.3):
        """
        初始化存储实例
        
        Args:
            data_path: 数据存储路径
            schema: 数据schema定义
            lazy_load: 是否延迟加载数据
            overwrite: 是否覆盖已存在的数据
            max_rows_per_region: 每个区域的最大行数
            auto_optimize_interval: 自动优化间隔（秒）
            auto_vacuum_threshold: 自动压缩阈值（已删除数据比例）
        """
        self.data_path = Path(data_path)
        self.schema = schema
        self.lazy_load = lazy_load
        self.overwrite = overwrite
        self.max_rows_per_region = max_rows_per_region
        
        if overwrite:
            self.delete()

        if not self.data_path.exists():
            self.data_path.mkdir(parents=True, exist_ok=True)

        # 初始化生命周期管理目录
        self._lifecycle_path = self.data_path / "__lifecycle"
        if not self._lifecycle_path.exists():
            self._lifecycle_path.mkdir(parents=True, exist_ok=True)

        # 初始化元数据
        self.region_meta_file = self.data_path / "regions_meta.lymp"
        if not self.region_meta_file.exists() or overwrite:
            self.regions_meta = {
                'version_counter': 0,
                'regions': []
            }
            self.regions = []
            self.version_counter = 0
            self._save_regions_meta()
        else:
            self._load_regions_meta()

        # 初始化当前区域
        self.current_region = self.regions[-1] if self.regions else None
        if not self.current_region or self.current_region['row_count'] >= self.max_rows_per_region:
            self._create_new_region()

        # 初始化DuckDB连接
        self._duckdb_conn = duckdb.connect(database=str(self.data_path / "__virtual.ddb"), read_only=False)
        self._register_table()

        # 初始化线程池和锁
        self._executor = ThreadPoolExecutor(max_workers=min(os.cpu_count(), 32))
        self.lock = threading.RLock()

        # 初始化后台维护任务
        self._last_optimize_time = time.time()
        self._auto_optimize_interval = auto_optimize_interval
        self._auto_vacuum_threshold = auto_vacuum_threshold
        self._background_thread = threading.Thread(target=self._background_maintenance, daemon=True)
        self._background_thread.start()

        if not lazy_load:
            self._load_all_regions()

    def mmap_reader(self, columns: Optional[List[str]] = None) -> MMapReaderContext:
        """
        获取内存映射读取器，用于高效读取指定列的数据
        
        Args:
            columns: 要读取的列名列表，为None时读取所有列
            
        Returns:
            MMapReaderContext: 内存映射读取器的上下文管理器
        """
        return MMapReaderContext(self, columns)

    def bulk_add(self, data: Union[List[Dict[str, Any]], pd.DataFrame], description: Optional[str] = None):
        """
        批量添加数据
        
        Args:
            data: 要添加的数据，可以是字典列表或DataFrame
            description: 数据描述
        """
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data

        if df.empty:
            return

        with self.lock:
            if self.current_region['row_count'] >= self.max_rows_per_region:
                self._create_new_region()

            ly_file = LyFile(self.data_path / self.current_region['data_file'])
            ly_file.append(df)
            
            self.current_region['row_count'] += len(df)
            
            # 创建新版本
            self.version_counter += 1
            version_info = {
                'version': self.version_counter,
                'description': description or f'Added {len(df)} rows at {time.strftime("%Y-%m-%d %H:%M:%S")}',
                'timestamp': time.time(),
                'row_count': len(df),
                'bitset_file': None
            }

            # 保存当前区域的BitSet状态
            bitset_file = f"v{self.version_counter}_r{self.current_region['region_id']}.bits"
            self.current_region['bitset'].save_to_file(self.data_path / bitset_file)
            version_info['bitset_file'] = bitset_file
            
            # 更新版本信息
            self.current_region['versions'].append(version_info)
            
            self._save_regions_meta()
            self._load_all_regions()
            
            return self.version_counter

    def delete_where(self, condition: str):
        """
        根据条件删除数据
        
        Args:
            condition: SQL WHERE子句条件
        """
        with self.lock:
            query = f"SELECT __rowid FROM storage WHERE {condition}"
            result = self._duckdb_conn.execute(query).fetchdf()
            
            if not result.empty:
                for rowid in result['__rowid']:
                    region_idx = self._find_region_by_rowid(rowid)
                    if region_idx is not None:
                        region = self.regions[region_idx]
                        local_rowid = rowid - region['start_row']
                        region['bitset'].clear_bit(local_rowid)
                
                self._save_regions_meta()
                self._load_all_regions()

    def update_where(self, condition: str, values: Dict[str, Any]):
        """
        根据条件更新数据
        
        Args:
            condition: SQL WHERE子句条件
            values: 要更新的列和值的字典
        """
        with self.lock:
            # 创建新版本
            self.version_counter += 1
            timestamp = time.time()
            
            # 获取要更新的行
            query = f"SELECT __rowid FROM storage WHERE {condition}"
            result = self._duckdb_conn.execute(query).fetchdf()
            
            if result.empty:
                return
            
            # 更新数据
            for rowid in result['__rowid']:
                region_idx = self._find_region_by_rowid(rowid)
                if region_idx is not None:
                    region = self.regions[region_idx]
                    local_rowid = rowid - region['start_row']
                    
                    # 读取原始数据
                    ly_file = LyFile(self.data_path / region['data_file'])
                    df = ly_file.read()
                    
                    # 更新数据
                    for col, value in values.items():
                        df.loc[local_rowid, col] = value
                    
                    # 保存更新后的数据
                    ly_file.write(df)
            
            self._save_regions_meta()
            self._load_all_regions()

    def vacuum(self):
        """压缩存储空间，移除已删除的数据"""
        with self.lock:
            new_regions = []
            new_start_row = 0
            
            for region in self.regions:
                if region['row_count'] == 0:
                    continue
                
                ly_file = LyFile(self.data_path / region['data_file'])
                df = ly_file.read()
                
                # 过滤活跃数据
                active_indices = [i for i in range(len(df)) if region['bitset'].get_bit(i)]
                if not active_indices:
                    continue
                
                df = df.iloc[active_indices]
                
                # 创建新区域
                new_region = {
                    "region_id": len(new_regions) + 1,
                    "start_row": new_start_row,
                    "row_count": len(df),
                    "data_file": f"region_{len(new_regions) + 1}.ly",
                    "versions": [],
                    "bitset": BitSet(size=len(df), fill=1)
                }
                
                # 写入新文件
                new_ly_file = LyFile(self.data_path / new_region['data_file'])
                new_ly_file.write(df)
                
                new_regions.append(new_region)
                new_start_row += len(df)
            
            # 更新状态
            self.regions = new_regions
            self.current_region = new_regions[-1] if new_regions else None
            self._save_regions_meta()
            self._load_all_regions()

    def create_version(self, version_name: str = None) -> int:
        """
        创建新版本
        
        Args:
            version_name: 版本名称
            
        Returns:
            int: 版本号
        """
        with self.lock:
            self.version_counter += 1
            version_info = {
                'version': self.version_counter,
                'name': version_name,
                'timestamp': time.time(),
                'bitset_file': None
            }

            for region in self.regions:
                if region['row_count'] > 0:
                    bitset_file = f"v{self.version_counter}_r{region['region_id']}.bits"
                    region['bitset'].save_to_file(self.data_path / bitset_file)
                    
                    version_info['bitset_file'] = bitset_file
                    region['versions'].append(version_info.copy())

            self._save_regions_meta()
            return self.version_counter

    def restore_version(self, target_version: int):
        """
        恢复到指定版本
        
        Args:
            target_version: 目标版本号
        """
        with self.lock:
            if target_version > self.version_counter:
                raise ValueError(f"Version {target_version} does not exist")

            current_start_row = 0
            for region in self.regions:
                valid_versions = [v for v in region['versions'] if v['version'] <= target_version]
                
                if valid_versions:
                    latest_valid_version = valid_versions[-1]
                    bitset_file = latest_valid_version.get('bitset_file')
                    
                    if bitset_file and (self.data_path / bitset_file).exists():
                        region['bitset'] = BitSet.load_from_file(self.data_path / bitset_file)
                    else:
                        region['bitset'] = BitSet(size=region['row_count'], fill=1)
                    
                    region['versions'] = valid_versions
                    region['start_row'] = current_start_row
                    current_start_row += region['row_count']
                else:
                    region['bitset'] = BitSet(size=region['row_count'], fill=0)
                    region['versions'] = []
                    region['start_row'] = current_start_row
                    current_start_row += region['row_count']

            self.version_counter = target_version
            self._save_regions_meta()
            self._load_all_regions()

    def list_versions(self) -> List[Dict[str, Any]]:
        """
        列出所有版本
        
        Returns:
            List[Dict[str, Any]]: 版本信息列表
        """
        with self.lock:
            all_versions = []
            for region in self.regions:
                for version in region['versions']:
                    version_info = {
                        'version': version['version'],
                        'name': version['name'],
                        'timestamp': version['timestamp'],
                        'region_id': region['region_id']
                    }
                    all_versions.append(version_info)
            
            all_versions.sort(key=lambda x: x['version'])
            return all_versions

    def get_stats(self) -> Dict[str, Any]:
        """
        获取存储统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        total_rows = sum(region['row_count'] for region in self.regions)
        active_rows = sum(
            sum(1 for i in range(region['row_count']) if region['bitset'].get_bit(i))
            for region in self.regions
        )
        
        return {
            'total_rows': total_rows,
            'active_rows': active_rows,
            'deleted_rows': total_rows - active_rows,
            'regions_count': len(self.regions),
            'current_region_id': self.current_region['region_id'] if self.current_region else None,
            'version_counter': self.version_counter
        }

    def _background_maintenance(self):
        """后台维护任务"""
        while True:
            try:
                current_time = time.time()
                
                if current_time - self._last_optimize_time > self._auto_optimize_interval:
                    stats = self.get_stats()
                    deleted_ratio = stats['deleted_rows'] / (stats['total_rows'] or 1)
                    
                    if deleted_ratio > self._auto_vacuum_threshold:
                        with self.lock:
                            self.vacuum()
                    
                    with self.lock:
                        self.optimize()
                    
                    self._last_optimize_time = current_time
                
                time.sleep(60)
                
            except Exception as e:
                print(f"Background maintenance error: {e}")
                time.sleep(60)

    def _create_new_region(self):
        """创建新的数据区域"""
        region_id = len(self.regions) + 1
        start_row = self.current_region['start_row'] + self.current_region['row_count'] if self.current_region else 0
        
        new_region = {
            "region_id": region_id,
            "start_row": start_row,
            "row_count": 0,
            "data_file": f"region_{region_id}.ly",
            "versions": [],
            "bitset": BitSet()
        }
        
        self.regions.append(new_region)
        self.current_region = new_region
        self._save_regions_meta()
        
        ly_file = LyFile(self.data_path / new_region['data_file'])
        return new_region

    def _load_regions_meta(self):
        """加载区域元数据"""
        with open(self.region_meta_file, 'rb') as f:
            self.regions_meta = msgpack.load(f)
            self.regions = self.regions_meta.get('regions', [])
            self.version_counter = self.regions_meta.get('version_counter', 0)
            
            for region in self.regions:
                if 'bitset' not in region or region['bitset'] is None:
                    region['bitset'] = BitSet(size=region['row_count'], fill=1)

    def _save_regions_meta(self):
        """保存区域元数据"""
        serializable_regions = []
        for region in self.regions:
            serializable_region = region.copy()
            if 'bitset' in serializable_region:
                serializable_region['bitset'] = None
            serializable_regions.append(serializable_region)

        self.regions_meta['regions'] = serializable_regions
        self.regions_meta['version_counter'] = self.version_counter

        with open(self.region_meta_file, 'wb') as f:
            msgpack.dump(self.regions_meta, f)

    def _load_all_regions(self):
        """加载所有区域的数据"""
        tables = []
        for region in self.regions:
            if region['row_count'] == 0:
                continue

            ly_file = LyFile(self.data_path / region['data_file'])
            df = ly_file.read()
            
            if region['bitset'] is not None:
                active_indices = [i for i in range(len(df)) if region['bitset'].get_bit(i)]
                if active_indices:
                    df = df.iloc[active_indices]
                    df['__rowid'] = [region['start_row'] + i for i in active_indices]
                    tables.append(df)

        if tables:
            self.table = pd.concat(tables, ignore_index=True)
        else:
            self.table = pd.DataFrame()

        self._register_table()

    def _register_table(self):
        """注册表到DuckDB"""
        try:
            self._duckdb_conn.unregister("storage")
        except:
            pass

        if hasattr(self, 'table') and not self.table.empty:
            self._duckdb_conn.register("storage", self.table)

    def _find_region_by_rowid(self, rowid: int) -> Optional[int]:
        """
        根据行ID找到对应的区域索引
        
        Args:
            rowid: 行ID
            
        Returns:
            Optional[int]: 区域索引
        """
        for i, region in enumerate(self.regions):
            if region['start_row'] <= rowid < region['start_row'] + region['row_count']:
                return i
        return None
    
    def delete(self):
        """删除存储"""
        shutil.rmtree(self.data_path)

    def to_pandas(self, include_vector: bool = True, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """转换为pandas DataFrame"""
        with self.lock:
            return self.table.to_pandas(include_vector, columns)
        
    def to_arrow(self, include_vector: bool = True, columns: Optional[List[str]] = None) -> pa.Table:
        """转换为Arrow Table"""
        with self.lock:
            return self.table.to_arrow(include_vector, columns)

    def __del__(self):
        """析构函数"""
        try:
            if hasattr(self, '_duckdb_conn'):
                self._duckdb_conn.close()
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=False)
        except:
            pass

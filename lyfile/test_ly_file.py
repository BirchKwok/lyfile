from typing import Optional
import pytest
import numpy as np
import pandas as pd
import pyarrow as pa
import tempfile
from pathlib import Path
from lyfile.storage.lyfile import LyFile
import struct

from lynse.computational_layer.engines import cosine


def verify_monotonic(table: pa.Table, column: str) -> bool:
    arr = table[column].combine_chunks()
    values = arr.to_numpy()
    return np.all(values[:-1] <= values[1:])


@pytest.fixture
def ly_file():
    """创建临时文件和LyFile实例"""
    temp_dir = tempfile.mkdtemp()
    file_path = Path(temp_dir) / "test.ly"
    file = LyFile(file_path)
    yield file
    # 清理临时文件
    if file_path.exists():
        file_path.unlink()
    Path(temp_dir).rmdir()

def test_basic_types(ly_file):
    """测试基本数据类型"""
    # 准备测试数据
    data = pd.DataFrame({
        'int32': np.array([1, 2, 3], dtype=np.int32),
        'int64': np.array([1, 2, 3], dtype=np.int64),
        'float32': np.array([1.1, 2.2, 3.3], dtype=np.float32),
        'float64': np.array([1.1, 2.2, 3.3], dtype=np.float64),
        'string': ['a', 'b', 'c'],
        'unicode': ['你好', '世界', '测试']
    })
    
    # 写入数据
    ly_file.write(data)
    
    # 普通读取测试
    read_data = ly_file.read()
    pd.testing.assert_frame_equal(data, read_data)
    
    # mmap读取测试
    with ly_file.mmap_reader() as reader:
        # 单列读取
        for col in data.columns:
            df = reader.read([col])
            pd.testing.assert_series_equal(data[col], df[col])
            
        # 多列读取
        df = reader.read(['int32', 'float64', 'string'])
        pd.testing.assert_frame_equal(data[['int32', 'float64', 'string']], df)

def test_special_types(ly_file):
    """测试特殊数据类型"""
    data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=3).astype(str),
        'category': ['A', 'B', 'C'],
        'bool': [True, False, True],
        'float': [1.5, 2.5, 3.5],
        'int': [1, 2, 3]
    })
    
    ly_file.write(data)
    read_data = ly_file.read()
    pd.testing.assert_frame_equal(data, read_data)

def test_numpy_arrays(ly_file):
    """测试numpy数组"""
    data = pd.DataFrame({
        'int_array': np.array([1, 2, 3], dtype=np.int64),
        'float_array': np.array([1.1, 2.2, 3.3], dtype=np.float64),
        'string_array': np.array(['a', 'b', 'c'])
    })
    
    ly_file.write(data)
    read_data = ly_file.read()
    pd.testing.assert_frame_equal(data, read_data)

def test_blob_types(ly_file):
    """测试Blob类型数据"""
    # 准备固定长度的测试数据
    data = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['file1', 'file2', 'file3'],
        'blob': [
            b'blob1' * 10,  # 50 bytes
            b'blob2' * 10,  # 50 bytes
            b'blob3' * 10   # 50 bytes
        ]
    })
    
    ly_file.write(data)
    read_data = ly_file.read()
    
    # 验证数据完整性
    pd.testing.assert_frame_equal(data[['id', 'name']], read_data[['id', 'name']])
    for i in range(len(data)):
        assert data['blob'].iloc[i] == read_data['blob'].iloc[i]

def test_large_blob(ly_file):
    """测试大型Blob数据"""
    # 使用简单的二进制数据而不是随机数据
    large_blob = b'large_blob' * 1024  # ~10KB
    
    data = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['file1', 'file2', 'file3'],
        'blob': [large_blob] * 3
    })
    
    ly_file.write(data)
    read_data = ly_file.read()
    
    # 验证数据完整性
    pd.testing.assert_frame_equal(data[['id', 'name']], read_data[['id', 'name']])
    for i in range(len(data)):
        assert data['blob'].iloc[i] == read_data['blob'].iloc[i]

def test_error_handling(ly_file):
    """测试错误处理"""
    # 测试空数据
    empty_df = pd.DataFrame({'column': []})
    ly_file.write(empty_df)
    assert ly_file.read().empty
    
    # 测试不存在的列
    with ly_file.mmap_reader() as reader:
        with pytest.raises(KeyError):
            reader.read(['non_existent_column'])
    
    # 测试文件损坏
    with open(ly_file.filepath, 'wb') as f:
        f.write(b'invalid data')
    with pytest.raises((AssertionError, struct.error)):
        ly_file.read()

def test_append_operations(ly_file):
    """测试追加操作"""
    # 准备测试数据
    data1 = pd.DataFrame({
        'id': [1, 2],
        'name': ['file1', 'file2'],
        'blob': [b'blob1' * 10, b'blob2' * 10]
    })
    
    data2 = pd.DataFrame({
        'id': [3],
        'name': ['file3'],
        'blob': [b'blob3' * 10]
    })
    
    # 写入和追加数据
    ly_file.write(data1)
    ly_file.append(data2)
    
    # 读取并验证
    read_data = ly_file.read()
    expected_data = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['file1', 'file2', 'file3'],
        'blob': [b'blob1' * 10, b'blob2' * 10, b'blob3' * 10]
    })
    
    pd.testing.assert_frame_equal(expected_data[['id', 'name']], read_data[['id', 'name']])
    for i in range(len(expected_data)):
        assert expected_data['blob'].iloc[i] == read_data['blob'].iloc[i]


if __name__ == '__main__':
    # 创建文件
    file = LyFile("data.ly", thread_count=8)

    # 写入数据
    data = pd.DataFrame({
        'id': np.arange(1000),
        'name': [f'name_{i}' for i in range(1000)],
        'value': np.random.random(1000)
    })
    file.write(data)
    print("写入数据完成")

    # 追加数据
    new_data = pd.DataFrame({
        'id': np.arange(1000, 2000),
        'name': [f'name_{i}' for i in range(1000, 2000)],
        'value': np.random.random(1000)
    })
    file.append(new_data)
    print("追加数据完成")

    # 读取数据
    df = file.read()
    print("全量读取: ")
    print(df)
    print("\n验证id列是否有序:")
    print(verify_monotonic(df, 'id'))

    # 使用mmap方式读特定列
    with file.mmap_reader() as reader:
        # 只读取id列
        df_id = reader.read(['id'])
        print("\n只读取id列: ")
        print(df_id)
        print("\n验证id列是否有序:")
        print(verify_monotonic(df_id, 'id'))

        # 读取多列
        df_partial = reader.read(['id', 'name'])
        print("\n读取id和name列: ")
        print(df_partial)
        print("\n验证id列是否有序:")
        print(verify_monotonic(df_partial, 'id'))

    # 显示文件信息
    print(f"\n文件形状: {file.shape}")

    file.append(new_data)
    print("追加数据完成")
    print("文件形状: ", file.shape)

    print("\n=======测试性能=======")

    import time
    import psutil
    import os

    def format_size(size):
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.2f}{unit}"
            size /= 1024
        return f"{size:.2f}TB"

    def get_memory_usage():
        """获取当前进程的内存使用"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss

    def run_benchmark(rows: int, thread_count: int):
        print(f"\n=== 性能测试: {rows:,} 行, {thread_count} 线程 ===")

        # 准备测试数据
        test_data = pd.DataFrame({
            'id': np.arange(rows),
            'name': [f'name_{i}' for i in range(rows)],
            'value': np.random.random(rows),
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], rows),
            'amount': np.random.normal(1000, 100, rows),
            'flag': np.random.choice([True, False], rows),
            'score': np.random.randint(0, 100, rows),
            'text': [f'This is a longer text for testing compression performance {i}' for i in range(rows)],
            'vector': [np.random.random(128) for _ in range(rows)]
        })

        file_path = f"benchmark_{rows}_{thread_count}.ly"
        file = LyFile(file_path, thread_count=thread_count)

        # 测试写入性能
        start_mem = get_memory_usage()
        start_time = time.time()
        file.write(test_data)
        write_time = time.time() - start_time
        file_size = os.path.getsize(file_path)
        mem_used = get_memory_usage() - start_mem

        print(f"\n写入性能:")
        print(f"耗时: {write_time:.2f}秒")
        print(f"速度: {rows/write_time:,.0f} 行/秒")
        print(f"文件大小: {format_size(file_size)}")
        print(f"压缩比: {file_size/(test_data.memory_usage(deep=True).sum()):.2%}")
        print(f"内存使用: {format_size(mem_used)}")

        # 测试全量读取性能
        start_mem = get_memory_usage()
        start_time = time.time()
        df = file.read()
        read_time = time.time() - start_time
        mem_used = get_memory_usage() - start_mem

        print(f"\n全量读取性能:")
        print(f"耗时: {read_time:.2f}秒")
        print(f"速: {rows/read_time:,.0f} 行/秒")
        print(f"内存使用: {format_size(mem_used)}")

        # 测试列式读取性能
        with file.mmap_reader() as reader:
            # 测试单列读取
            start_time = time.time()
            df_single = reader.read(['id'])
            single_col_time = time.time() - start_time

            print(f"\n单列读取性能:")
            print(f"耗时: {single_col_time:.2f}秒")
            print(f"速度: {rows/single_col_time:,.0f} 行/秒")

            # 测试多列读取
            start_time = time.time()
            df_multi = reader.read(['id', 'name', 'value'])
            multi_col_time = time.time() - start_time

            print(f"\n多列读取性能:")
            print(f"耗时: {multi_col_time:.2f}秒")
            print(f"速度: {rows/multi_col_time:,.0f} 行/秒")

            # 测试列处理性能
        print("\n=== 测试列处理性能 ===")
        with file.mmap_reader() as reader:
            # 测试1: 数值计算
            print("\n1. 数值列处理:")
            start_time = time.time()
            mean = float(reader.execute_along_column('value', np.mean))
            max_val = float(reader.execute_along_column('value', np.max))
            min_val = float(reader.execute_along_column('value', np.min))
            calc_time = time.time() - start_time

            print(f"数值统计耗时: {calc_time:.3f}秒")
            print(f"平均值: {mean:.3f}")
            print(f"最大值: {max_val:.3f}")
            print(f"最小值: {min_val:.3f}")

            # 验证结果
            np.testing.assert_almost_equal(
                mean,
                test_data['value'].mean(),
                decimal=3
            )

            # 测试2: 字符串处理
            print("\n2. 字符串列理:")
            start_time = time.time()
            # 计字符串平均长度
            avg_len = float(reader.execute_along_column(
                'text',
                lambda x: np.mean([len(s) for s in x])
            ))
            str_time = time.time() - start_time

            print(f"字符串处理耗时: {str_time:.3f}秒")
            print(f"本平均长度: {avg_len:.1f}")

            # 测试3: 向量处理
            print("\n3. 向量处理:")
            start_time = time.time()

            # 计算向量余弦相似度
            query = np.random.random((1, 128))
            def calc_cosine_similarity(vectors_array: np.ndarray, query_vector: Optional[np.ndarray] = None) -> np.ndarray:
                """计算余弦相似度"""
                if query_vector is None:
                    # 如果没有提供查询向量，使用第一个向量作为查询向量
                    query_vector = vectors_array[0]
                
                # 确保查询向量是二维的
                if len(query_vector.shape) == 1:
                    query_vector = query_vector.reshape(1, -1)
                
                # 确保向量数组是二维的
                if len(vectors_array.shape) == 1:
                    # 如果是对象数组，需要将向量堆叠成二维数组
                    vectors_array = np.vstack(vectors_array)
                
                # 确保维度匹配
                if query_vector.shape[1] != vectors_array.shape[1]:
                    raise ValueError(
                        f"Query vector dimension ({query_vector.shape[1]}) "
                        f"doesn't match vectors array dimension ({vectors_array.shape[1]})"
                    )
                
                return cosine(vectors_array, query_vector, 1)[1]

            norms = reader.execute_along_column('vector', calc_cosine_similarity)
            vector_time = time.time() - start_time

            print(f"向量处理耗时: {vector_time:.3f}秒")
            print(f"向量范数示例 (前5个):")
            for i in range(min(5, len(norms))):
                print(f"  向量 {i}: {norms[i]:.3f}")

            # 测试4: 复杂计算
            print("\n4. 复杂计算:")
            start_time = time.time()

            # 计算数值列的移动平均
            def moving_average(data, window=5):
                return np.convolve(data, np.ones(window)/window, mode='valid')

            ma = reader.execute_along_column('value',
                                        lambda x: moving_average(x))
            complex_time = time.time() - start_time

            print(f"复杂计算耗时: {complex_time:.3f}秒")
            print(f"移动平均示例 (前5个): {ma[:5]}")

            # 性能总结
            print("\n处理性能总结:")
            print(f"数值计算速度: {rows/calc_time:,.0f} 行/秒")
            print(f"字符串处理速度: {rows/str_time:,.0f} 行/秒")
            print(f"向量处理速度: {rows/vector_time:,.0f} 行/秒")
            print(f"复杂计算速度: {rows/complex_time:,.0f} 行/秒")

        # 测试追加性能
        append_data = test_data.copy()
        append_data['id'] += len(test_data)

        start_time = time.time()
        file.append(append_data)
        append_time = time.time() - start_time

        print(f"\n追加性能:")
        print(f"耗时: {append_time:.2f}秒")
        print(f"速度: {rows/append_time:,.0f} 行/秒")

        # 清理测试文件
        os.remove(file_path)

    # 运行不同规模的测试
    test_configs = [
        (10_000, 4),    # 小数据量
        (100_000, 4),   # 中等数据量
        (1_000_000, 4), # 大数据量
    ]

    # 测试不同线程数
    thread_configs = [
        (100_000, 1),   # 单线程
        (100_000, 2),   # 2线程
        (100_000, 4),   # 4线程
        (100_000, 8),   # 8线程
    ]

    # 运行数据量测试
    print("\n=== 测试不同数据量 ===")
    for rows, threads in test_configs:
        run_benchmark(rows, threads)

    # 运行线程数测试
    print("\n=== 测试不同线程数 ===")
    for rows, threads in thread_configs:
        run_benchmark(rows, threads)

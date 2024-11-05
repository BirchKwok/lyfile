from lynse.computational_layer.engines import cosine
from .storage.ly_file import LyFile
import pandas as pd
import numpy as np
import os
import psutil
import time
from typing import Optional

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


def run_all_benchmarks():
    """运行所有基准测试"""
    # 测试不同规模的数据量
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
    # print("\n=== 测试不同数据量 ===")
    for rows, threads in test_configs:
        run_benchmark(rows, threads)

    # 运行线程数测试
    print("\n=== 测试不同线程数 ===")
    for rows, threads in thread_configs:
        run_benchmark(rows, threads)


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

    # print(f"\n写入性能:")
    # print(f"耗时: {write_time:.2f}秒")
    # print(f"速度: {rows/write_time:,.0f} 行/秒")
    # print(f"文件大小: {format_size(file_size)}")
    # print(f"压缩比: {file_size/(test_data.memory_usage(deep=True).sum()):.2%}")
    # print(f"内存使用: {format_size(mem_used)}")

    # 测试全量读取性能
    start_mem = get_memory_usage()
    start_time = time.time()
    df = file.read()
    read_time = time.time() - start_time
    mem_used = get_memory_usage() - start_mem

    # print(f"\n全量读取性能:")
    # print(f"耗时: {read_time:.2f}秒")
    # print(f"速: {rows/read_time:,.0f} 行/秒")
    # print(f"内存使用: {format_size(mem_used)}")

    # 测试列式读取性能
    with file.mmap_reader() as reader:
        # 测试单列读取
        start_time = time.time()
        df_single = reader.read(['id'])
        single_col_time = time.time() - start_time

        # print(f"\n单列读取性能:")
        # print(f"耗时: {single_col_time:.2f}秒")
        # print(f"速度: {rows/single_col_time:,.0f} 行/秒")

        # 测试多列读取
        start_time = time.time()
        df_multi = reader.read(['id', 'name', 'value'])
        multi_col_time = time.time() - start_time

        # print(f"\n多列读取性能:")
        # print(f"耗时: {multi_col_time:.2f}秒")
        # print(f"速度: {rows/multi_col_time:,.0f} 行/秒")

        # 测试列处理性能
    # print("\n=== 测试列处理性能 ===")
    with file.mmap_reader() as reader:
        # 测试1: 数值计算
        # print("\n1. 数值列处理:")
        start_time = time.time()
        mean = float(reader.execute_along_column('value', np.mean))
        max_val = float(reader.execute_along_column('value', np.max))
        min_val = float(reader.execute_along_column('value', np.min))
        calc_time = time.time() - start_time

        # print(f"数值统计耗时: {calc_time:.3f}秒")
        # print(f"平均值: {mean:.3f}")
        # print(f"最大值: {max_val:.3f}")
        # print(f"最小值: {min_val:.3f}")

        # 验证结果
        np.testing.assert_almost_equal(
            mean,
            test_data['value'].mean(),
            decimal=3
        )

        # 测试2: 字符串处理
        # print("\n2. 字符串列理:")
        start_time = time.time()
        # 计字符串平均长度
        avg_len = float(reader.execute_along_column(
            'text',
            lambda x: np.mean([len(s) for s in x])
        ))
        str_time = time.time() - start_time

        # print(f"字符串处理耗时: {str_time:.3f}秒")
        # print(f"本平均长度: {avg_len:.1f}")

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
        # print(f"向量范数示例 (前5个):")
        # for i in range(min(5, len(norms))):
        #     print(f"  向量 {i}: {norms[i]:.3f}")

        # 测试4: 复杂计算
        # print("\n4. 复杂计算:")
        start_time = time.time()

        # 计算数值列的移动平均
        def moving_average(data, window=5):
            return np.convolve(data, np.ones(window)/window, mode='valid')

        ma = reader.execute_along_column('value',
                                    lambda x: moving_average(x))
        complex_time = time.time() - start_time

        # print(f"复杂计算耗时: {complex_time:.3f}秒")
        # print(f"移动平均示例 (前5个): {ma[:5]}")

        # 性能总结
        # print("\n处理性能总结:")
        # print(f"数值计算速度: {rows/calc_time:,.0f} 行/秒")
        # print(f"字符串处理速度: {rows/str_time:,.0f} 行/秒")
        # print(f"向量处理速度: {rows/vector_time:,.0f} 行/秒")
        # print(f"复杂计算速度: {rows/complex_time:,.0f} 行/秒")

    # 测试追加性能
    append_data = test_data.copy()
    append_data['id'] += len(test_data)

    start_time = time.time()
    file.append(append_data)
    append_time = time.time() - start_time

    # print(f"\n追加性能:")
    # print(f"耗时: {append_time:.2f}秒")
    # print(f"速度: {rows/append_time:,.0f} 行/秒")

    # 清理测试文件
    os.remove(file_path)


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


if __name__ == '__main__':
    run_all_benchmarks()

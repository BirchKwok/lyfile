import pytest
import numpy as np
import pandas as pd
import pyarrow as pa
import lyfile


@pytest.fixture
def test_data():
    rows = 100_0000
    
    # 创建测试数据
    df = pd.DataFrame({
        'id': np.arange(rows, dtype=np.int64),
        'name': [f'name_{i}' for i in range(rows)],
        'value': np.random.random(rows).astype(np.float64),
        'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], rows),
        'amount': np.random.normal(1000, 100, rows).astype(np.float64),
        'flag': np.random.choice([True, False], rows),
        'score': np.random.randint(0, 100, rows, dtype=np.int64),
        'text': [f'This is a longer text for testing compression performance {i}' for i in range(rows)],
        'vector': [np.array(np.random.random(128), dtype=np.float64) for _ in range(rows)]
    })
    
    data_dict = df.to_dict('list')
    table = pa.Table.from_pandas(df)
    
    return {'df': df, 'dict': data_dict, 'table': table}

def verify_data(filename):
    file = lyfile.LyFile(filename.as_posix())
    data = file.read()
    return {
        'num_rows': data.num_rows,
        'num_columns': data.num_columns,
        'schema': {field.name: str(field.type) for field in data.schema}
    }

def test_write_read_dataframe(test_data, tmp_path):
    filename = tmp_path / "test_df.lyf"
    file = lyfile.LyFile(filename.as_posix())
    file.write(test_data['df'])
    
    result = verify_data(filename)
    assert result['num_rows'] == len(test_data['df'])
    assert result['num_columns'] == len(test_data['df'].columns)

def test_write_read_dict(test_data, tmp_path):
    filename = tmp_path / "test_dict.lyf"
    file = lyfile.LyFile(filename.as_posix())
    file.write(test_data['dict'])
    
    result = verify_data(filename)
    assert result['num_rows'] == len(test_data['dict']['id'])
    assert result['num_columns'] == len(test_data['dict'])

def test_write_read_table(test_data, tmp_path):
    filename = tmp_path / "test_table.lyf"
    file = lyfile.LyFile(filename.as_posix())
    file.write(test_data['table'])
    
    result = verify_data(filename)
    assert result['num_rows'] == len(test_data['df'])
    assert result['num_columns'] == len(test_data['df'].columns)

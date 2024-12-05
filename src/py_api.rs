// py_api.rs
use std::fs::File;
use std::sync::{Arc, RwLock};
use std::path::Path;
use std::io::{Read, Seek, SeekFrom};
use byteorder::{LittleEndian, ReadBytesExt};

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use arrow::pyarrow::IntoPyArrow;
use arrow::record_batch::RecordBatch;
use arrow::datatypes::Schema;
use numpy::PyArray2;
use pyo3::types::PyDict;

use crate::structs::*;
use crate::distances::*;


#[pymethods]
impl LyFile {
    #[new]
    /// Initializes a new LyFile object.
    ///
    /// Args:
    ///     filepath (str): The path of the file to read or write.
    ///
    /// Returns:
    ///     LyFile: A new instance of LyFile.
    ///
    /// Examples:
    ///     >>> from lyfile import LyFile
    ///     >>> lyfile = LyFile("example.ly")
    #[pyo3(text_signature = "(self, filepath)")]
    fn new(filepath: String) -> Self {
        LyFile {
            filepath,
            schema: Arc::new(RwLock::new(None)),
            chunks: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Reads data from the file.
    ///
    /// Args:
    ///     columns (Optional[Union[str, List[str]]]): 
    ///         For table data: The names of columns to read. If None, all columns are read.
    ///         For vector data: The name of the vector to read.
    ///
    /// Returns:
    ///     Union[pyarrow.Table, numpy.ndarray]: 
    ///         - If reading table columns: returns a pyarrow Table
    ///         - If reading a vector: returns a numpy array
    #[pyo3(text_signature = "(self, columns=None)")]
    fn read(&self, columns: Option<&PyAny>, py: Python) -> PyResult<PyObject> {
        // 检查文件是否存在
        if !Path::new(&self.filepath).exists() {
            return Err(PyValueError::new_err("File does not exist"));
        }

        // 读取元数据
        let mut file = File::open(&self.filepath)?;
        let metadata = self.read_metadata(&mut file)?;

        // 初始化 schema
        let schema = Arc::new(metadata.schema.to_arrow_schema());
        {
            let mut schema_lock = self.schema.write().unwrap();
            *schema_lock = Some(schema.clone());
        }

        // 更新 chunks
        {
            let mut chunks_lock = self.chunks.write().unwrap();
            *chunks_lock = metadata.chunks.clone();
        }

        // 处理列选择
        let selected_columns = if let Some(cols) = columns {
            if cols.is_instance_of::<pyo3::types::PyString>() {
                vec![cols.extract::<String>()?]
            } else if cols.is_instance_of::<pyo3::types::PyList>() {
                cols.extract::<Vec<String>>()?
            } else {
                return Err(PyValueError::new_err(
                    "columns must be either a string or a list of strings",
                ));
            }
        } else {
            // 如果没有指定列，返回所有表格列
            schema.fields()
                .iter()
                .map(|f| f.name().clone())
                .collect()
        };

        // 分类列
        let mut table_columns = Vec::new();
        let mut vector_columns = Vec::new();
        
        for col in &selected_columns {
            if schema.fields().iter().any(|f| f.name() == col) {
                table_columns.push(col.clone());
            } else if metadata.vec_region.as_ref().map_or(false, |vr| vr.vectors.iter().any(|v| &v.name == col)) {
                vector_columns.push(col.clone());
            } else {
                return Err(PyValueError::new_err(format!("Column '{}' not found in schema or vector region", col)));
            }
        }

        // 根据请求的列类型返回相应的数据
        if table_columns.is_empty() && vector_columns.len() == 1 && 
           // 检查是否是单个字符串或只包含一个元素的列表
           (columns.map_or(false, |c| c.is_instance_of::<pyo3::types::PyString>()) || 
            (columns.map_or(false, |c| c.is_instance_of::<pyo3::types::PyList>()) && selected_columns.len() == 1)) {
            // 只读取单个向量列
            self.read_vec(vector_columns[0].clone(), py)
        } else if !table_columns.is_empty() && !vector_columns.is_empty() {
            // 同时读取表格列和向量列
            let table = self.read_table_data(&table_columns, py)?;
            let vector_data = PyDict::new(py);
            for vec_name in vector_columns {
                let vec_data = self.read_vec(vec_name.clone(), py)?;
                vector_data.set_item(vec_name, vec_data)?;
            }
            Ok((table, vector_data).into_py(py))
        } else if !table_columns.is_empty() {
            // 只读取表格列
            self.read_table_data(&table_columns, py)
        } else if !vector_columns.is_empty() {
            // 读取多个向量列
            let vector_data = PyDict::new(py);
            for vec_name in vector_columns {
                let vec_data = self.read_vec(vec_name.clone(), py)?;
                vector_data.set_item(vec_name, vec_data)?;
            }
            Ok(vector_data.into_py(py))
        } else {
            // 如果没有指定列，返回所有表格列
            self.read_table_data(&schema.fields().iter().map(|f| f.name().clone()).collect::<Vec<_>>(), py)
        }
    }

    /// Writes data to the file.
    ///
    /// Args:
    ///     tdata (Optional[Union[pandas.DataFrame, dict, pyarrow.Table]]):
    ///         The table data to be written.
    ///         Supported types include:
    ///         - Pandas DataFrame
    ///         - Python dictionary
    ///         - PyArrow Table
    ///         If None, only vector data will be written.
    ///     vdata (Optional[dict]):
    ///         Dictionary of vector data where keys are vector names and values are numpy arrays.
    ///         If None, only table data will be written.
    ///
    /// Raises:
    ///     ValueError: If both tdata and vdata are None, or if input types are not supported.
    #[pyo3(text_signature = "(self, tdata=None, vdata=None)")]
    fn write(&mut self, tdata: Option<&PyAny>, vdata: Option<&PyAny>, py: Python) -> PyResult<()> {
        if tdata.is_none() && vdata.is_none() {
            return Err(PyValueError::new_err("Both tdata and vdata cannot be None"));
        }

        // 写入表格数据
        if let Some(data) = tdata {
            let table = if data.hasattr("__class__")? {
                let class_name = data.getattr("__class__")?.getattr("__name__")?.extract::<String>()?;
                match class_name.as_str() {
                    "DataFrame" => {
                        let pa = PyModule::import(py, "pyarrow")?;
                        pa.getattr("Table")?.call_method1("from_pandas", (data,))?
                    },
                    "dict" => {
                        let pa = PyModule::import(py, "pyarrow")?;
                        pa.getattr("Table")?.call_method1("from_pydict", (data,))?
                    },
                    "Table" => {
                        // 直接使用 PyArrow Table，不需要转换
                        data
                    },
                    _ => return Err(PyValueError::new_err(format!("Unsupported input type: {}", class_name))),
                }
            } else {
                return Err(PyValueError::new_err("Invalid input type"));
            };

            self.write_table_data(table, py)?;
        }

        // 写入向量数据
        if let Some(data) = vdata {
            if !data.is_instance_of::<pyo3::types::PyDict>() {
                return Err(PyValueError::new_err("vdata must be a dictionary"));
            }
            let dict = data.downcast::<pyo3::types::PyDict>()?;
            for (name, array) in dict.iter() {
                let name = name.extract::<String>()?;
                self.write_vec(name, array, py)?;
            }
        }

        Ok(())
    }

    /// Appends data to the existing file.
    ///
    /// Args:
    ///     tdata (Optional[Union[pandas.DataFrame, dict, pyarrow.Table]]):
    ///         The table data to be appended.
    ///     vdata (Optional[dict]):
    ///         Dictionary of vector data to be appended.
    ///
    /// Raises:
    ///     ValueError: If both tdata and vdata are None, or if the file doesn't exist.
    #[pyo3(text_signature = "(self, tdata=None, vdata=None)")]
    fn append(&mut self, tdata: Option<&PyAny>, vdata: Option<&PyAny>, py: Python) -> PyResult<()> {
        if tdata.is_none() && vdata.is_none() {
            return Err(PyValueError::new_err("Both tdata and vdata cannot be None"));
        }

        // 如果文件不存在调用 write
        if !Path::new(&self.filepath).exists() {
            return self.write(tdata, vdata, py);
        }

        // 追加表格数据
        if let Some(data) = tdata {
            self.append_table_data(data, py)?;
        }

        // 追加向量数据
        if let Some(data) = vdata {
            if !data.is_instance_of::<pyo3::types::PyDict>() {
                return Err(PyValueError::new_err("vdata must be a dictionary"));
            }
            let dict = data.downcast::<pyo3::types::PyDict>()?;
            self._append_vec(dict, py)?;
        }

        Ok(())
    }

    /// Returns the shape of the data stored in the file.
    #[getter]
    pub fn shape(&self) -> PyResult<(usize, usize)> {
        let chunks_lock = self.chunks.read().unwrap();
        let total_rows: usize = chunks_lock.iter().map(|chunk| chunk.rows).sum();
        let columns = if let Some(schema_ref) = &*self.schema.read().unwrap() {
            schema_ref.fields().len()
        } else {
            0
        };
        Ok((total_rows, columns))
    }

    /// Returns the shape of a stored vector.
    ///
    /// Args:
    ///     name (str): Name of the vector
    ///
    /// Returns:
    ///     tuple: Shape of the vector
    #[pyo3(text_signature = "(self, name)")]
    fn get_vec_shape(&self, name: String) -> PyResult<Vec<usize>> {
        let mut file = File::open(&self.filepath)?;
        let metadata = self.read_metadata(&mut file)?;

        let vec_region = metadata.vec_region
            .ok_or_else(|| PyValueError::new_err("No vector data in file"))?;

        let vector_info = vec_region.vectors.iter()
            .find(|v| v.name == name)
            .ok_or_else(|| PyValueError::new_err(format!("Vector '{}' not found", name)))?;

        Ok(vector_info.shape.clone())
    }

    /// Lists all vector names stored in the file.
    ///
    /// Returns:
    ///     list: List of vector names
    #[pyo3(text_signature = "(self)")]
    fn list_vectors(&self) -> PyResult<Vec<String>> {
        let mut file = File::open(&self.filepath)?;
        let metadata = self.read_metadata(&mut file)?;

        Ok(metadata.vec_region
            .map(|vr| vr.vectors.into_iter().map(|v| v.name).collect())
            .unwrap_or_default())
    }

    /// Lists all columns in the file with their types and storage format.
    ///
    /// Returns:
    ///     dict: Dictionary where keys are column names and values are tuples of (data_type, storage_format)
    ///           storage_format is either 'ly_table' or 'ly_vec'
    ///
    /// Example:
    ///     >>> lyfile.list_columns()
    ///     {
    ///         'col1': ('Int32', 'ly_table'),
    ///         'vec1': ('Float32', 'ly_vec')
    ///     }
    #[pyo3(text_signature = "(self)")]
    fn list_columns(&self, py: Python) -> PyResult<PyObject> {
        let mut file = File::open(&self.filepath)?;
        let metadata = self.read_metadata(&mut file)?;
        
        // 创建一个 Python 字典
        let dict = pyo3::types::PyDict::new(py);
        
        // 添加常规列
        for field in metadata.schema.fields.iter() {
            dict.set_item(
                &field.name,
                (field.data_type.clone(), "ly_table")
            )?;
        }
        
        // 添加向量列
        if let Some(vec_region) = metadata.vec_region {
            for vector in vec_region.vectors {
                dict.set_item(
                    &vector.name,
                    (vector.dtype, "ly_vec")
                )?;
            }
        }
        
        Ok(dict.into())
    }

    /// 在指定的向量列中搜索最近邻
    ///
    /// Args:
    ///     vector_name (str): 要搜索的向量列名
    ///     query_vectors (numpy.ndarray): 查询向，shape为(n_queries, dim)
    ///     top_k (int): 返回的最近邻数量
    ///     metric (str): 距离度量方式，可选 "l2", "ip" (inner product), "cosine"
    ///
    /// Returns:
    ///     Tuple[numpy.ndarray, numpy.ndarray]: (indices, distances)
    ///     indices的shape为(n_queries, top_k)，distances的shape为(n_queries, top_k)
    #[pyo3(signature = (vector_name, query_vectors, top_k, metric="l2"))]
    fn search_vector(
        &self,
        vector_name: String,
        query_vectors: &PyAny,
        top_k: usize,
        metric: &str,
        py: Python<'_>,
    ) -> PyResult<(Py<PyArray2<i32>>, Py<PyArray2<f32>>)> {
        if metric != "l2" && metric != "ip" && metric != "cosine" {
            return Err(PyValueError::new_err("Invalid metric. Supported metrics are 'l2', 'ip', and 'cosine'."));
        }
        
        let binding = self.read_vec(vector_name, py)?;
        let base_array = binding.extract::<&PyAny>(py)?;
        
        // 获取数据类型字符串并转换为小写以便统一比较
        let dtype_str = base_array.getattr("dtype")?.str()?.to_string().to_lowercase();
        
        // 更新类型判断逻辑
        let use_f32 = if dtype_str.contains("float32") || dtype_str.contains("f4") || dtype_str.contains("<f4") {
            true
        } else if dtype_str.contains("float64") || dtype_str.contains("f8") || dtype_str.contains("<f8") {
            // 删除未使用的 np 变量
            let max_val: f64 = base_array.call_method0("max")?.extract()?;
            let min_val: f64 = base_array.call_method0("min")?.extract()?;
            max_val <= f32::MAX as f64 && min_val >= f32::MIN as f64
        } else {
            return Err(PyValueError::new_err(format!("Unsupported data type: {}", dtype_str)));
        };
    
        if use_f32 {
            // f32 计算路径
            let base_vectors = if dtype_str.contains("float32") || dtype_str.contains("f4") {
                base_array.extract::<&PyArray2<f32>>()?
            } else {
                let array_f32 = base_array.call_method1("astype", ("float32",))?;
                array_f32.extract::<&PyArray2<f32>>()?
            };
    
            let query_array = if query_vectors.getattr("dtype")?.str()?.to_string().to_lowercase().contains("float32") {
                query_vectors.extract::<&PyArray2<f32>>()?
            } else {
                let array_f32 = query_vectors.call_method1("astype", ("float32",))?;
                array_f32.extract::<&PyArray2<f32>>()?
            };
    
            compute_distances_f32(py, query_array.readonly(), base_vectors.readonly(), top_k, metric)
        } else {
            // f64 计算路径
            let base_vectors = base_array.extract::<&PyArray2<f64>>()?;
            let query_array = if query_vectors.getattr("dtype")?.str()?.to_string().to_lowercase().contains("float64") {
                query_vectors.extract::<&PyArray2<f64>>()?
            } else {
                let array_f64 = query_vectors.call_method1("astype", ("float64",))?;
                array_f64.extract::<&PyArray2<f64>>()?
            };
    
            // 计算结果后转换为 f32
            let (indices, distances) = compute_distances_f64(py, query_array.readonly(), base_vectors.readonly(), top_k, metric)?;
            let distances_f32 = distances.as_ref(py).call_method1("astype", ("float32",))?;
            Ok((indices, distances_f32.extract()?))
        }
    }

    pub fn read_columns(&self, columns: Vec<String>, py: Python) -> PyResult<PyObject> {
        // 检查文件是否存在
        if !Path::new(&self.filepath).exists() {
            return Err(PyValueError::new_err("File does not exist"));
        }

        // 打开文件并读取 schema
        let mut file = File::open(&self.filepath)?;
        let metadata = self.read_metadata(&mut file)?;

        // 初始化 schema（如果还没有初始化）
        {
            let mut schema_lock = self.schema.write().unwrap();
            if schema_lock.is_none() {
                *schema_lock = Some(Arc::new(metadata.schema.to_arrow_schema()));
            }
        }

        // 更 chunks（如果需要）
        {
            let mut chunks_lock = self.chunks.write().unwrap();
            if chunks_lock.is_empty() {
                *chunks_lock = metadata.chunks.clone();
            }
        }

        // 读取指定的列
        self.read_table_data(&columns, py)
    }

    pub fn get_schema(&self, py: Python) -> PyResult<PyObject> {
        // 检查文件是否存在
        if !Path::new(&self.filepath).exists() {
            return Err(PyValueError::new_err("File does not exist"));
        }

        // 如果 schema 已经加载，直接返回
        if let Some(schema) = self.schema.read().unwrap().as_ref() {
            // 从 Arc<Schema> 取引用，然后克隆内部的 Schema
            let schema_clone = (**schema).clone();
            return Ok(schema_clone.into_pyarrow(py)?.into_py(py));
        }

        // 否则从文件读取
        let mut file = File::open(&self.filepath)?;
        let metadata = self.read_metadata(&mut file)?;
        let schema = Arc::new(metadata.schema.to_arrow_schema());

        // 更新缓存的 schema
        {
            let mut schema_lock = self.schema.write().unwrap();
            *schema_lock = Some(schema.clone());
        }

        // 克隆内部的 Schema 并转换为 PyArrow
        let schema_clone = (*schema).clone();
        Ok(schema_clone.into_pyarrow(py)?.into_py(py))
    }
}


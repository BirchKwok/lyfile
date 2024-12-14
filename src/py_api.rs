// py_api.rs
use std::fs::File;
use std::sync::{Arc, RwLock};
use std::path::Path;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::PyArray2;
use pyo3::types::PyDict;

use crate::structs::*;
use arrow::pyarrow::IntoPyArrow;
use crate::neighbors::search_vector_internal;
use once_cell::sync::OnceCell;

#[pyclass]
struct LyDataView {
    #[pyo3(get)]
    columns_list: Vec<String>,
    cached_table: OnceCell<PyObject>,
    cached_vectors: OnceCell<PyObject>,
    file: Arc<_LyFile>,
    selected_columns: Option<Vec<String>>,
    load_mmap_vec: bool,
}

#[pymethods]
impl LyDataView {
    #[getter]
    fn table_data(&self, py: Python) -> PyResult<PyObject> {
        self.cached_table.get_or_try_init(|| {
            // 如果只选择了向量列，直接返回 None
            if let Some(cols) = &self.selected_columns {
                let schema = self.file.schema.read().unwrap();
                let schema = schema.as_ref()
                    .ok_or_else(|| PyValueError::new_err("Schema not initialized"))?;
                
                // 检查是否有表格列
                let has_table_columns = cols.iter()
                    .any(|col| schema.fields().iter().any(|f| f.name() == col));
                
                if !has_table_columns {
                    return Ok(py.None());
                }
            }

            // 原有的表格数据读取逻辑...
            let schema = self.file.schema.read().unwrap();
            let schema = schema.as_ref()
                .ok_or_else(|| PyValueError::new_err("Schema not initialized"))?;

            let table_columns: Vec<String> = match &self.selected_columns {
                Some(cols) => cols.iter()
                    .filter(|col| schema.fields().iter().any(|f| f.name() == *col))
                    .cloned()
                    .collect(),
                None => schema.fields()
                    .iter()
                    .map(|f| f.name().clone())
                    .collect()
            };

            if table_columns.is_empty() {
                Ok(py.None())
            } else {
                self.file.read_table_data(&table_columns, py)
            }
        }).map(|obj| obj.clone_ref(py))
    }

    #[getter]
    fn vector_data(&self, py: Python) -> PyResult<PyObject> {
        self.cached_vectors.get_or_try_init(|| {
            // 如果只选择了表格列，直接返回 None
            if let Some(cols) = &self.selected_columns {
                let mut file = File::open(&self.file.filepath)?;
                let metadata = self.file.read_metadata(&mut file)?;
                
                if let Some(vec_region) = &metadata.vec_region {
                    // 检查是否有向量列
                    let has_vector_columns = cols.iter()
                        .any(|col| vec_region.vectors.iter().any(|v| &v.name == col));
                    
                    if !has_vector_columns {
                        return Ok(py.None());
                    }
                } else {
                    return Ok(py.None());
                }
            }

            // 原有的向量数据读取逻辑...
            let mut file = File::open(&self.file.filepath)?;
            let metadata = self.file.read_metadata(&mut file)?;

            let vec_region = match metadata.vec_region {
                Some(vr) => vr,
                None => return Ok(py.None())
            };

            let vector_names: Vec<String> = match &self.selected_columns {
                Some(cols) => cols.iter()
                    .filter(|col| vec_region.vectors.iter().any(|v| &v.name == *col))
                    .cloned()
                    .collect(),
                None => vec_region.vectors.iter().map(|v| v.name.clone()).collect()
            };

            if vector_names.is_empty() {
                Ok(py.None())
            } else {
                let vector_data = PyDict::new(py);
                for name in vector_names {
                    let vec_data = self.file.read_vec_with_mmap(name.clone(), self.load_mmap_vec, py)?;
                    vector_data.set_item(name, vec_data)?;
                }
                Ok(vector_data.into_py(py))
            }
        }).map(|obj| obj.clone_ref(py))
    }

    #[getter]
    fn all_entries(&self, py: Python) -> PyResult<PyObject> {
        let table = self.table_data(py)?;
        let vectors = self.vector_data(py)?;
        
        // 如果两者都是 None，返回 None
        if table.is_none(py) && vectors.is_none(py) {
            return Ok(py.None());
        }
        
        if let Some(_cols) = &self.selected_columns {
            // 如果只有表格数据，直接返回表格数据
            if vectors.is_none(py) {
                return Ok(table);
            }
            
            // 如果只有向量数据（无论单列还是多列），直接返回向量数据
            if table.is_none(py) {
                return Ok(vectors);
            }
        }
        
        // 同时有表格和向量数据时，返回元组
        Ok((table, vectors).into_py(py))
    }
}

#[pymethods]
impl _LyFile {
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
        _LyFile {
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
    ///     load_mmap_vec (bool): Whether to use numpy's memmap to read vector data.
    ///
    /// Returns:
    ///     Union[pyarrow.Table, numpy.ndarray]: 
    ///         - If reading table columns: returns a pyarrow Table
    ///         - If reading a vector: returns a numpy array
    #[pyo3(signature = (columns=None, load_mmap_vec=true))]
    fn read(&self, columns: Option<&PyAny>, load_mmap_vec: bool, py: Python) -> PyResult<Py<LyDataView>> {
        // check if file exists
        if !Path::new(&self.filepath).exists() {
            return Err(PyValueError::new_err("File does not exist"));
        }

        // read metadata
        let mut file = File::open(&self.filepath)?;
        let metadata = self.read_metadata(&mut file)?;

        // initialize schema
        let schema = Arc::new(metadata.schema.to_arrow_schema());
        {
            let mut schema_lock = self.schema.write().unwrap();
            *schema_lock = Some(schema.clone());
        }

        // update chunks
        {
            let mut chunks_lock = self.chunks.write().unwrap();
            *chunks_lock = metadata.chunks.clone();
        }

        // 处理列名
        let selected_columns = if let Some(cols) = columns {
            let cols = if cols.is_instance_of::<pyo3::types::PyString>() {
                vec![cols.extract::<String>()?]
            } else if cols.is_instance_of::<pyo3::types::PyList>() {
                cols.extract::<Vec<String>>()?
            } else {
                return Err(PyValueError::new_err(
                    "columns must be either a string or a list of strings",
                ));
            };

            // 验证所有列是否存在
            for col in &cols {
                let is_table_col = schema.fields().iter().any(|f| f.name() == col);
                let is_vector_col = metadata.vec_region.as_ref()
                    .map_or(false, |vr| vr.vectors.iter().any(|v| &v.name == col));
                
                if !is_table_col && !is_vector_col {
                    return Err(PyValueError::new_err(format!("Column '{}' not found", col)));
                }
            }

            Some(cols)
        } else {
            None
        };

        // 获取选中的列名或所有列名
        let columns_list = if let Some(cols) = &selected_columns {
            cols.clone()
        } else {
            let mut all_columns = schema.fields()
                .iter()
                .map(|f| f.name().clone())
                .collect::<Vec<_>>();
            
            if let Some(vec_region) = metadata.vec_region.as_ref() {
                all_columns.extend(vec_region.vectors.iter().map(|v| v.name.clone()));
            }
            all_columns
        };

        // 创建 LyDataView 实例
        let view = LyDataView {
            columns_list,
            cached_table: OnceCell::new(),
            cached_vectors: OnceCell::new(),
            file: Arc::new((*self).clone()),  // 先解引用，再克隆，最后创建新的 Arc
            selected_columns,
            load_mmap_vec,
        };

        Py::new(py, view)
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

        // write table data
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
                        // directly use PyArrow Table, no conversion needed
                        data
                    },
                    _ => return Err(PyValueError::new_err(format!("Unsupported input type: {}", class_name))),
                }
            } else {
                return Err(PyValueError::new_err("Invalid input type"));
            };

            self.write_table_data(table, py)?;
        }

        // write vector data
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

        // if file doesn't exist, call write
        if !Path::new(&self.filepath).exists() {
            return self.write(tdata, vdata, py);
        }

        // append table data
        if let Some(data) = tdata {
            self.append_table_data(data, py)?;
        }

        // append vector data
        if let Some(data) = vdata {
            if !data.is_instance_of::<pyo3::types::PyDict>() {
                return Err(PyValueError::new_err("vdata must be a dictionary"));
            }
            let dict = data.downcast::<pyo3::types::PyDict>()?;
            self.append_vec(dict, py)?;
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
        
        let dict = pyo3::types::PyDict::new(py);

        // add regular columns
        for field in metadata.schema.fields.iter() {
            let subdict = pyo3::types::PyDict::new(py);
            subdict.set_item("dtype", field.data_type.clone())?;
            subdict.set_item("lytype", "table")?;
            dict.set_item(&field.name, subdict)?;
        }
        
        // add vector columns
        if let Some(vec_region) = metadata.vec_region {
            for vector in vec_region.vectors {
                let subdict = pyo3::types::PyDict::new(py);
                subdict.set_item("dtype", vector.dtype)?;
                subdict.set_item("lytype", "vector")?;
                dict.set_item(&vector.name, subdict)?;
            }
        }
        
        Ok(dict.into())
    }

    /// Searches for nearest neighbors in the specified vector column.
    ///
    /// Args:
    ///     vector_name (str): The name of the vector column to search.
    ///     query_vectors (numpy.ndarray): Query vectors, shape is (n_queries, dim)
    ///     top_k (int): Number of nearest neighbors to return
    ///     metric (str): Distance metric, supported metrics are "l2", "ip" (inner product), "cosine"
    ///
    /// Returns:
    ///     Tuple[numpy.ndarray, numpy.ndarray]: (indices, distances)
    ///     indices shape is (n_queries, top_k), distances shape is (n_queries, top_k)
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
        
        let binding = self.read_vec_native(vector_name.clone())?;
        
        match binding {
            VectorData::F32(base_vecs, _shape) => {
                search_vector_internal::<f32>(base_vecs, query_vectors, top_k, metric, py)
            },
            VectorData::F64(base_vecs, _shape) => {
                // for f64, we first convert to f32
                let base_vecs_f32: Vec<f32> = base_vecs.into_iter()
                    .map(|x| x as f32)
                    .collect();
                search_vector_internal::<f32>(base_vecs_f32, query_vectors, top_k, metric, py)
            },
            VectorData::F16(data, _shape) => {
                // 改进float16到float32的换
                let base_vecs_f32: Vec<f32> = data.into_iter()
                    .map(|x| {
                        let val = f32::from(x);
                        if val.is_nan() { 0.0 } else { val }
                    })
                    .collect();
                search_vector_internal::<f32>(base_vecs_f32, query_vectors, top_k, metric, py)
            },
            // 对于其他整数类型，先转换为 f32
            VectorData::I32(data, _shape) => {
                let base_vecs_f32: Vec<f32> = data.into_iter()
                    .map(|x| x as f32)
                    .collect();
                search_vector_internal::<f32>(base_vecs_f32, query_vectors, top_k, metric, py)
            },
            VectorData::I64(data, _shape) => {
                let base_vecs_f32: Vec<f32> = data.into_iter()
                    .map(|x| x as f32)
                    .collect();
                search_vector_internal::<f32>(base_vecs_f32, query_vectors, top_k, metric, py)
            },
            VectorData::U32(data, _shape) => {
                let base_vecs_f32: Vec<f32> = data.into_iter()
                    .map(|x| x as f32)
                    .collect();
                search_vector_internal::<f32>(base_vecs_f32, query_vectors, top_k, metric, py)
            },
            VectorData::U64(data, _shape) => {
                let base_vecs_f32: Vec<f32> = data.into_iter()
                    .map(|x| x as f32)
                    .collect();
                search_vector_internal::<f32>(base_vecs_f32, query_vectors, top_k, metric, py)
            },
            VectorData::I16(data, _shape) => {
                let base_vecs_f32: Vec<f32> = data.into_iter()
                    .map(|x| x as f32)
                    .collect();
                search_vector_internal::<f32>(base_vecs_f32, query_vectors, top_k, metric, py)
            },
            VectorData::U16(data, _shape) => {
                let base_vecs_f32: Vec<f32> = data.into_iter()
                    .map(|x| x as f32)
                    .collect();
                search_vector_internal::<f32>(base_vecs_f32, query_vectors, top_k, metric, py)
            },
            VectorData::I8(data, _shape) => {
                let base_vecs_f32: Vec<f32> = data.into_iter()
                    .map(|x| x as f32)
                    .collect();
                search_vector_internal::<f32>(base_vecs_f32, query_vectors, top_k, metric, py)
            },
            VectorData::U8(data, _shape) => {
                let base_vecs_f32: Vec<f32> = data.into_iter()
                    .map(|x| x as f32)
                    .collect();
                search_vector_internal::<f32>(base_vecs_f32, query_vectors, top_k, metric, py)
            },
            VectorData::Bool(data, _shape) => {
                let base_vecs_f32: Vec<f32> = data.into_iter()
                    .map(|x| if x { 1.0 } else { 0.0 })
                    .collect();
                search_vector_internal::<f32>(base_vecs_f32, query_vectors, top_k, metric, py)
            },
        }
    }

    /// 读取指定行号的数据
    ///
    /// Args:
    ///     row_indices (List[int]): 要读取的行号列表
    ///
    /// Returns:
    ///     pyarrow.Table: 包含指定行的数据表
    ///
    /// Example:
    ///     >>> lyfile = LyFile("example.ly")
    ///     >>> data = lyfile.read_rows([0, 5, 10, 15])  # 读取第0、5、10、15行
    #[pyo3(text_signature = "(self, row_indices)")]
    fn read_rows(&self, row_indices: Vec<usize>, py: Python) -> PyResult<PyObject> {
        // 验证输入
        if row_indices.is_empty() {
            return Err(PyValueError::new_err("row_indices cannot be empty"));
        }

        // 读取指定行的数据
        let batch = self.read_rows_by_indices(&row_indices)?;
        
        // 转换为PyArrow Table并返回
        let py_batch = batch.into_pyarrow(py)?;
        let pa = py.import("pyarrow")?;
        Ok(pa.getattr("Table")?.call_method1("from_batches", ([py_batch],))?.into_py(py))
    }
}


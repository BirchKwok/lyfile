// py_api.rs
use std::fs::File;
use std::sync::{Arc, RwLock};
use std::path::Path;
use std::cmp::Ordering;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::PyArray2;
use pyo3::types::PyDict;

use crate::structs::*;
use simsimd::SpatialSimilarity;
use num_traits::{Float, ToPrimitive, FromPrimitive};
use arrow::pyarrow::IntoPyArrow;


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
    fn read(&self, columns: Option<&PyAny>, load_mmap_vec: bool, py: Python) -> PyResult<PyObject> {
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

        // handle column selection
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
            // if no columns specified, return all table columns
            schema.fields()
                .iter()
                .map(|f| f.name().clone())
                .collect()
        };

        // classify columns
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

        // return data based on requested column type
        if table_columns.is_empty() && vector_columns.len() == 1 && 
           (columns.map_or(false, |c| c.is_instance_of::<pyo3::types::PyString>()) || 
            (columns.map_or(false, |c| c.is_instance_of::<pyo3::types::PyList>()) && selected_columns.len() == 1)) {
            // read single vector column
            self.read_vec_with_mmap(vector_columns[0].clone(), load_mmap_vec, py)
        } else if !table_columns.is_empty() && !vector_columns.is_empty() {
            // read both table and vector columns
            let table = self.read_table_data(&table_columns, py)?;
            let vector_data = PyDict::new(py);
            for vec_name in vector_columns {
                let vec_data = self.read_vec_with_mmap(vec_name.clone(), load_mmap_vec, py)?;
                vector_data.set_item(vec_name, vec_data)?;
            }
            Ok((table, vector_data).into_py(py))
        } else if !table_columns.is_empty() {
            // read table columns
            self.read_table_data(&table_columns, py)
        } else if !vector_columns.is_empty() {
            // read multiple vector columns
            let vector_data = PyDict::new(py);
            for vec_name in vector_columns {
                let vec_data = self.read_vec_with_mmap(vec_name.clone(), load_mmap_vec, py)?;
                vector_data.set_item(vec_name, vec_data)?;
            }
            Ok(vector_data.into_py(py))
        } else {
            // if no columns specified, return all table columns
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
            }
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

fn search_vector_internal<T>(
    base_vecs: Vec<T>, 
    query_vectors: &PyAny, 
    top_k: usize, 
    metric: &str,
    py: Python<'_>
) -> PyResult<(Py<PyArray2<i32>>, Py<PyArray2<f32>>)> 
where 
    T: SpatialSimilarity + Copy + 'static + PartialOrd + Float + numpy::Element + FromPrimitive,
    for<'py> &'py PyArray2<T>: FromPyObject<'py>,
{
    let query_array = if query_vectors.getattr("dtype")?.str()?.to_string().to_lowercase().contains(std::any::type_name::<T>()) {
        query_vectors.extract::<&PyArray2<T>>()?
    } else {
        let dtype = if std::any::type_name::<T>() == "f32" { "float32" } else { "float64" };
        let array = query_vectors.call_method1("astype", (dtype,))?;
        array.extract::<&PyArray2<T>>()?
    };
    
    let query_slice = unsafe { query_array.as_slice()? };
    let query_shape = query_array.shape();
    let n_queries = query_shape[0];
    let dim = query_shape[1];
    
    let mut indices = vec![vec![0i32; top_k]; n_queries];
    let mut distances = vec![vec![0f32; top_k]; n_queries];
    
    for (query_idx, query_vec) in query_slice.chunks(dim).enumerate() {
        let mut distances_with_indices: Vec<_> = base_vecs.chunks(dim)
            .enumerate()
            .map(|(i, base_vec)| {
                let dist = match metric {
                    "l2" => T::sqeuclidean(query_vec, base_vec)
                        .unwrap_or_else(|| num_traits::cast(1e30_f64).unwrap()),
                    "cosine" => T::cosine(query_vec, base_vec)
                        .unwrap_or_else(|| num_traits::cast(-1e30_f64).unwrap()),
                    "ip" => T::dot(query_vec, base_vec)
                        .unwrap_or_else(|| num_traits::cast(-1e30_f64).unwrap()),
                    _ => unreachable!(),
                };
                (dist, i as i32)
            })
            .collect();

        // 使用快速选择算法找到前k个元素
        match metric {
            "ip" => {
                // 对于内积，我们要找最大的k个
                quick_select_k_largest(&mut distances_with_indices, top_k);
                distances_with_indices[..top_k].sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            },
            _ => {
                // 对于l2和cosine，我们要找最小的k个
                quick_select_k_smallest(&mut distances_with_indices, top_k);
                distances_with_indices[..top_k].sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            }
        };
        
        // 复制结果到输出数组
        for k in 0..top_k {
            let (dist, idx) = distances_with_indices[k];
            indices[query_idx][k] = idx;
            distances[query_idx][k] = dist.to_f32().unwrap_or(0.0);
        }
    }
    
    let indices_array = PyArray2::from_vec2(py, &indices)?;
    let distances_array = PyArray2::from_vec2(py, &distances)?;
    
    Ok((indices_array.to_owned(), distances_array.to_owned()))
}

// 获取pivot的辅助函数
fn get_pivot<T: PartialOrd + Copy>(arr: &[(T, i32)], left: usize, right: usize) -> usize {
    let mid = left + (right - left) / 2;
    let (a, b, c) = (arr[left].0, arr[mid].0, arr[right].0);
    
    if a <= b {
        if b <= c { mid }
        else if a <= c { right }
        else { left }
    } else {
        if a <= c { left }
        else if b <= c { right }
        else { mid }
    }
}

// 插入排序 - 升序
fn insertion_sort<T: PartialOrd + Copy>(arr: &mut [(T, i32)], left: usize, right: usize) {
    for i in (left + 1)..=right {
        let mut j = i;
        while j > left && arr[j - 1].0 > arr[j].0 {
            arr.swap(j - 1, j);
            j -= 1;
        }
    }
}

fn block_sort<T: PartialOrd + Copy>(arr: &mut [(T, i32)], left: usize, right: usize) {
    const BLOCK_SIZE: usize = 32;  // 可以根据实际情况调整
    let len = right - left + 1;
    
    if len <= 1 {
        return;
    }
    
    if len <= BLOCK_SIZE {
        insertion_sort(arr, left, right);
        return;
    }
    
    // 将数组分成若干块并分别排序
    for i in (left..=right).step_by(BLOCK_SIZE) {
        let end = (i + BLOCK_SIZE - 1).min(right);
        insertion_sort(arr, i, end);
    }
}

// 2. 三路快速选择算法
fn partition3way<T: PartialOrd + Copy>(arr: &mut [(T, i32)], left: usize, right: usize) 
    -> (usize, usize) // 返回等于pivot的区间(lt, gt)
{
    let pivot_idx = get_pivot(arr, left, right);
    let pivot_val = arr[pivot_idx].0;
    
    // 将pivot移到开头
    arr.swap(pivot_idx, left);
    
    let mut lt = left;      // 小于pivot的右边界
    let mut i = left + 1;   // 当前处理的元素
    let mut gt = right;     // 大于pivot的左边界
    
    while i <= gt {
        match arr[i].0.partial_cmp(&pivot_val).unwrap_or(Ordering::Equal) {
            Ordering::Less => {
                arr.swap(lt, i);
                lt += 1;
                i += 1;
            },
            Ordering::Greater => {
                arr.swap(i, gt);
                if gt > 0 {
                    gt -= 1;
                }
            },
            Ordering::Equal => {
                i += 1;
            }
        }
    }
    
    (lt, gt)
}

// 3. 优化的快速选择算法
fn quick_select_k_smallest<T: PartialOrd + Copy>(arr: &mut [(T, i32)], k: usize) {
    if k >= arr.len() {
        return;
    }
    
    const BLOCK_SIZE: usize = 32;
    let mut left = 0;
    let mut right = arr.len() - 1;
    
    while right - left > BLOCK_SIZE {
        // 使用三路划分
        let (lt, gt) = partition3way(arr, left, right);
        
        if k < lt {
            right = lt - 1;
        } else if k > gt {
            left = gt + 1;
        } else {
            return; // k在[lt,gt]区间内，已找到
        }
        
        // 如果区间仍然很大，考虑采样
        if right - left > 1000 {
            sample_and_presort(arr, left, right);
        }
    }
    
    // 对小区间使用块排序
    block_sort(arr, left, right);
}

// 4. 采样预处理
fn sample_and_presort<T: PartialOrd + Copy>(arr: &mut [(T, i32)], left: usize, right: usize) {
    const SAMPLE_SIZE: usize = 5;
    let len = right - left + 1;
    
    if len <= SAMPLE_SIZE {
        return;
    }
    
    // 采样并排序
    let step = len / SAMPLE_SIZE;
    let mut samples = Vec::with_capacity(SAMPLE_SIZE);
    
    for i in 0..SAMPLE_SIZE {
        let idx = left + i * step;
        samples.push((arr[idx].0, idx));
    }
    
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    
    // 将采样点移动到合适的位置
    for (i, &(_, idx)) in samples.iter().enumerate() {
        let target = left + i * step;
        if target != idx {
            arr.swap(target, idx);
        }
    }
}

// 5. 优化的最大k个元素选择
fn quick_select_k_largest<T: PartialOrd + Copy>(arr: &mut [(T, i32)], k: usize) {
    if k >= arr.len() {
        return;
    }
    
    const BLOCK_SIZE: usize = 32;
    let mut left = 0;
    let mut right = arr.len() - 1;
    let target = arr.len() - k;
    
    while right - left > BLOCK_SIZE {
        let (lt, gt) = partition3way(arr, left, right);
        
        if target < lt {
            right = lt - 1;
        } else if target > gt {
            left = gt + 1;
        } else {
            return;
        }
        
        if right - left > 1000 {
            sample_and_presort(arr, left, right);
        }
    }
    
    block_sort(arr, left, right);
}


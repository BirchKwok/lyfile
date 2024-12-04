// py_api.rs
use std::fs::File;
use std::io::{Write, Seek, Read, Cursor};
use std::sync::{Arc, RwLock};
use std::fs::OpenOptions;
use std::path::Path;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use arrow::pyarrow::{PyArrowType, ToPyArrow};

use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};
use pyo3::exceptions::PyValueError;

use serde_json;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use memmap2::MmapOptions;
use rayon::prelude::*;  

use numpy::PyArray2;

use crate::structs::*;
use crate::distances::*;

// 在文件开头添加常量
const CHUNK_MAX_ROWS: usize = 10000; // 每个chunk最大行数为1万行

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

    // Writes data to the custom file format.
    // 
    // Args:
    //     data (Union[pandas.DataFrame, dict, pyarrow.Table]):
    //         The input data to be written.
    //         Supported types include:
    //         - Pandas DataFrame
    //         - Python dictionary
    //         - PyArrow Table
    // 
    // Raises:
    //     ValueError: If the input data type is not supported or is empty.
    //     IOError: If an error occurs while writing to the file.
    //     ArrowError: If there is an error with Arrow serialization.
    //     SerializationError: If an error occurs during metadata serialization.
    // 
    // Examples:
    //     >>> import pandas as pd
    //     >>> from lyfile import LyFile
    //     >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    //     >>> lyfile = LyFile("example.ly")
    //     >>> lyfile.write(df)
    #[pyo3(text_signature = "(self, data)")]
    fn write(&mut self, data: &PyAny, py: Python) -> PyResult<()> {
        if data.is_none() {
            return Err(PyValueError::new_err("Input data is None"));
        }
        use std::fs;

        if Path::new(&self.filepath).exists() {
            fs::remove_file(&self.filepath)?;
        }

        // 获取 PyArrow Table 并转换为 RecordBatch 列表
        let record_batches = {
            let table = if data.hasattr("__class__")? {
                let class_name = data.getattr("__class__")?.getattr("__name__")?;
                match class_name.extract::<String>()?.as_str() {
                    "DataFrame" => {
                        let pa = PyModule::import(py, "pyarrow")?;
                        pa.getattr("Table")?.call_method1("from_pandas", (data,))?
                    },
                    "Table" => data.into(),
                    "dict" => {
                        let pa = PyModule::import(py, "pyarrow")?;
                        pa.getattr("Table")?.call_method1("from_pydict", (data,))?
                    },
                    _ => return Err(PyValueError::new_err("Unsupported input type")),
                }
            } else {
                return Err(PyValueError::new_err("Invalid input type"));
            };

            // 修改这里: 直接从 table 获取 batches，而不是先获取 schema
            let batches = table.call_method0("to_batches")?
                .extract::<Vec<PyArrowType<RecordBatch>>>()?;
            
            if batches.is_empty() {
                return Err(PyValueError::new_err("Empty table"));
            }

            // 将 batches 按照 CHUNK_MAX_ROWS 大小分割
            let mut result_batches = Vec::new();
            for batch in batches {
                let batch = batch.0;
                let num_rows = batch.num_rows();
                let mut start_row = 0;
                
                while start_row < num_rows {
                    let end_row = (start_row + CHUNK_MAX_ROWS).min(num_rows);
                    let slice = batch.slice(start_row, end_row - start_row);
                    result_batches.push(slice);
                    start_row = end_row;
                }
            }
            
            if result_batches.is_empty() {
                return Err(PyValueError::new_err("Empty table"));
            }
            result_batches
        };

        // 释放 GIL
        let (schema_clone, metadata_clone) = py.allow_threads(|| -> PyResult<(SchemaRef, Metadata)> {
            let schema = record_batches[0].schema();
            let serialized_schema = handle_serde_error(serde_json::to_vec(&SerializableSchema::from(schema.as_ref())))?;
            let schema_length = serialized_schema.len() as u32;

            let mut file = handle_io_error(
                OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .open(&self.filepath)
            )?;

            handle_io_error(file.seek(std::io::SeekFrom::Start(0)))?;

            let header_placeholder = vec![0u8; FILE_HEADER_SIZE];
            handle_io_error(file.write_all(&header_placeholder))?;

            handle_io_error(file.write_all(&serialized_schema))?;

            let data_offset = FILE_HEADER_SIZE as u64 + schema_length as u64;
            let mut current_offset = data_offset;

            // 写入多个 Chunks
            let num_chunks = record_batches.len();
            let mut chunks = Vec::new();

            // 并行处理每个 chunk 的数据，但顺序写入文件
            for batch in record_batches {
                let chunk_offset = current_offset;
                let chunk_info = self.write_chunk(&mut file, &batch, &mut current_offset)?;
                chunks.push((chunk_offset, chunk_info));
            }

            let index_offset = current_offset;

            self.write_index_region(&mut file, &chunks)?;

            let metadata = Metadata {
                schema: SerializableSchema::from(schema.as_ref()),
                chunks: chunks.iter().map(|(_, info)| info.clone()).collect(),
                vec_region: None,
            };
            self.write_footer(&mut file, &metadata)?;

            handle_io_error(file.seek(std::io::SeekFrom::Start(0)))?;
            self.write_file_header(
                &mut file,
                data_offset,
                index_offset,
                schema_length,
                FEATURE_FLAGS,
                num_chunks as u32,
            )?;

            handle_io_error(file.flush())?;

            Ok((schema.clone(), metadata))
        })?;

        {
            let mut schema_lock = self.schema.write().unwrap();
            *schema_lock = Some(schema_clone);
        }
        {
            let mut chunks_lock = self.chunks.write().unwrap();
            *chunks_lock = metadata_clone.chunks;
        }

        Ok(())
    }

    /// Reads data from the custom file format.
    ///
    /// Args:
    ///     columns (Optional[Union[str, List[str]]]):
    ///         The names of columns to read. If None, all columns are read.
    ///
    /// Returns:
    ///     pyarrow.Table: The data read from the file.
    ///
    /// Raises:
    ///     ValueError: If the file does not exist or the column names are invalid.
    ///     IOError: If an error occurs while reading the file.
    ///
    /// Examples:
    ///     >>> from lyfile import LyFile
    ///     >>> lyfile = LyFile("example.ly")
    ///     >>> data = lyfile.read()
    ///     >>> print(data)
    #[pyo3(text_signature = "(self, columns=None)")]
    fn read(&self, columns: Option<&PyAny>, py: Python) -> PyResult<PyObject> {
        // 检查文件是否存在
        if !Path::new(&self.filepath).exists() {
            return Err(PyValueError::new_err("File does not exist"));
        }

        // 处理列选择
        let selected_column_names: Vec<String> = if let Some(col_selection) = columns {
            if col_selection.is_instance_of::<pyo3::types::PyString>() {
                vec![col_selection.extract::<String>()?]
            } else if col_selection.is_instance_of::<pyo3::types::PyList>() {
                col_selection.extract::<Vec<String>>()?
            } else {
                return Err(PyValueError::new_err(
                    "columns must be either a string or a list of strings",
                ));
            }
        } else {
            Vec::new() // 空向量表示未指定列，将在后续代码中处理
        };

        // 释放 GIL
        let (batches, arrow_schema, metadata_chunks) = py.allow_threads(|| -> PyResult<(Vec<RecordBatch>, SchemaRef, Vec<ChunkInfo>)> {
            let mut file = handle_io_error(File::open(&self.filepath))?;

            // 读取文件头
            let mut header_buffer = [0u8; FILE_HEADER_SIZE];
            handle_io_error(file.read_exact(&mut header_buffer))?;

            let mut cursor = Cursor::new(&header_buffer);

            // MAGIC
            let mut magic = [0u8; 8];
            cursor.read_exact(&mut magic)?;
            if magic != MAGIC_BYTES {
                return Err(PyValueError::new_err("Invalid file format"));
            }

            // 版本号
            let _version = cursor.read_u32::<LittleEndian>()?;

            // 数据偏移
            let _data_offset = cursor.read_u64::<LittleEndian>()?;

            // 索引偏移
            let _index_offset = cursor.read_u64::<LittleEndian>()?;

            // Schema 长度
            let schema_length = cursor.read_u32::<LittleEndian>()?;

            // 特征标记
            let _feature_flags = cursor.read_u32::<LittleEndian>()?;

            // Chunk 数量
            let _num_chunks = cursor.read_u32::<LittleEndian>()?;

            // 读取 Schema
            let mut schema_bytes = vec![0u8; schema_length as usize];
            handle_io_error(file.read_exact(&mut schema_bytes))?;
            let schema: SerializableSchema = handle_serde_error(serde_json::from_slice(&schema_bytes))?;
            let arrow_schema = schema.to_arrow_schema();
            let arrow_schema_ref = Arc::new(arrow_schema.clone());

            // 读取 Footer，获取元数据

            // 读取 MAGIC_BYTES（最后的8字节）
            handle_io_error(file.seek(std::io::SeekFrom::End(-8)))?;
            let mut footer_magic = [0u8; 8];
            handle_io_error(file.read_exact(&mut footer_magic))?;
            if footer_magic != MAGIC_BYTES {
                return Err(PyValueError::new_err("Invalid footer magic"));
            }

            // 读取元数据长度（倒数第12到第9字节）
            handle_io_error(file.seek(std::io::SeekFrom::End(-12)))?;
            let metadata_length = file.read_u32::<LittleEndian>()?;

            // 计算元数据的起始偏移
            handle_io_error(file.seek(std::io::SeekFrom::End(-(12 + metadata_length as i64))))?;

            // 读取元数据
            let mut metadata_bytes = vec![0u8; metadata_length as usize];
            handle_io_error(file.read_exact(&mut metadata_bytes))?;

            let metadata: Metadata = handle_serde_error(serde_json::from_slice(&metadata_bytes))?;

            // 如果未指定列，读取所有列
            let selected_column_names = if selected_column_names.is_empty() {
                arrow_schema.fields().iter().map(|f| f.name().clone()).collect()
            } else {
                selected_column_names.clone()
            };

            // 验证选择的列是否存在
            for col in &selected_column_names {
                if !arrow_schema.fields().iter().any(|f| f.name() == col) {
                    return Err(PyValueError::new_err(format!("Column '{}' not found in schema", col)));
                }
            }

            // 收集所有的 RecordBatch
            let batches: Vec<RecordBatch> = metadata.chunks.par_iter().map(|chunk| {
                self.read_chunk(&self.filepath, &arrow_schema, &selected_column_names, chunk)
            }).collect::<PyResult<Vec<_>>>()?;

            Ok((batches, arrow_schema_ref, metadata.chunks.clone()))
        })?;

        // 更新 self.schema 和 self.chunks
        {
            let mut schema_lock = self.schema.write().unwrap();
            *schema_lock = Some(arrow_schema);
        }
        {
            let mut chunks_lock = self.chunks.write().unwrap();
            *chunks_lock = metadata_chunks;
        }

        // 转换为 PyArrow Table
        Python::with_gil(|py| {
            let pyarrow = PyModule::import(py, "pyarrow")?;
            let pybatches = batches
                .into_iter()
                .map(|batch| batch.to_pyarrow(py).map_err(|e| PyValueError::new_err(format!("PyArrow conversion error: {}", e))))
                .collect::<PyResult<Vec<_>>>()?;
            let table = pyarrow.getattr("Table")?
                .call_method1("from_batches", (pybatches,))?;
            Ok(table.into())
        })
    }

    /// Returns the shape of the data stored in the file.
    #[getter]
    fn shape(&self) -> PyResult<(usize, usize)> {
        let chunks_lock = self.chunks.read().unwrap();
        let total_rows: usize = chunks_lock.iter().map(|chunk| chunk.rows).sum();
        let columns = if let Some(schema_ref) = &*self.schema.read().unwrap() {
            schema_ref.fields().len()
        } else {
            0
        };
        Ok((total_rows, columns))
    }

    /// Appends data to the existing file.
    ///
    /// Args:
    ///     data (Union[pandas.DataFrame, dict, pyarrow.Table]):
    ///         The input data to be appended.
    ///         Supported types include:
    ///         - Pandas DataFrame
    ///         - Python dictionary
    ///         - PyArrow Table
    ///
    /// Raises:
    ///     ValueError: If the input data type is not supported or if schemas do not match.
    ///     IOError: If an error occurs while writing to the file.
    ///     ArrowError: If there is an error with Arrow serialization.
    ///     SerializationError: If an error occurs during metadata serialization.
    ///
    /// Examples:
    ///     >>> import pandas as pd
    ///     >>> from lyfile import LyFile
    ///     >>> df = pd.DataFrame({'col1': [5, 6], 'col2': [7, 8]})
    ///     >>> lyfile = LyFile("example.ly")
    ///     >>> lyfile.append(df)
    #[pyo3(text_signature = "(self, data)")]
    fn append(&mut self, data: &PyAny, py: Python) -> PyResult<()> {
        // 首先检查文件是否存在
        if !Path::new(&self.filepath).exists() {
            // 如果文件不存在，直接调用 write 方法
            return self.write(data, py);
        }

        // 获取 RecordBatch
        let record_batch = {
            let table = if data.hasattr("__class__")? {
                let class_name = data.getattr("__class__")?.getattr("__name__")?;
                match class_name.extract::<String>()?.as_str() {
                    "DataFrame" => {
                        let pa = PyModule::import(py, "pyarrow")?;
                        pa.getattr("Table")?.call_method1("from_pandas", (data,))?
                    },
                    "Table" => data.into(),
                    "dict" => {
                        let pa = PyModule::import(py, "pyarrow")?;
                        pa.getattr("Table")?.call_method1("from_pydict", (data,))?
                    },
                    _ => return Err(PyValueError::new_err("Unsupported input type")),
                }
            } else {
                return Err(PyValueError::new_err("Invalid input type"));
            };

            let batches = table.call_method0("to_batches")?
                .extract::<Vec<PyArrowType<RecordBatch>>>()?;
            if batches.is_empty() {
                return Err(PyValueError::new_err("Empty table"));
            }
            batches[0].0.clone()
        };

        // 释放 GIL
        let (metadata_clone, _new_chunk_info) = py.allow_threads(|| -> PyResult<(Metadata, ChunkInfo)> {
            // 打开文件，进行读写操
            let mut file = handle_io_error(
                OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&self.filepath)
            )?;

            // 读取文件的元数据
            let metadata = self.read_metadata(&mut file)?;

            // 验证新数据的 Schema 是否与文件中的一致
            let file_schema = metadata.schema.to_arrow_schema();
            if record_batch.schema().fields() != file_schema.fields() {
                return Err(PyValueError::new_err("Schema of new data does not match the file schema"));
            }

            // 定位到 Footer 的开始位置，即 Footer 前面的偏移量
            let footer_offset = handle_io_error(file.seek(std::io::SeekFrom::End(-(8 + 4))))?;
            let metadata_length = file.read_u32::<LittleEndian>()?;
            let metadata_start = footer_offset - metadata_length as u64;

            // 定位到数据的开始位置，准备追加新的 Chunk
            let new_chunk_offset = metadata_start;

            // 将文件指针移动到元数据的开始位置，准备写入新的 Chunk
            handle_io_error(file.seek(std::io::SeekFrom::Start(new_chunk_offset)))?;

            let mut current_offset = new_chunk_offset;

            // 写入新的 Chunk
            let chunk_info = self.write_chunk(&mut file, &record_batch, &mut current_offset)?;

            // 更新元数据
            let mut new_metadata = metadata.clone();
            new_metadata.chunks.push(chunk_info.clone());

            // 重新写入 Footer
            self.write_footer(&mut file, &new_metadata)?;

            // 更新文件头部信息中的 Chunk 数量
            let num_chunks = new_metadata.chunks.len() as u32;
            handle_io_error(file.seek(std::io::SeekFrom::Start(0)))?;
            self.update_chunk_count_in_header(&mut file, num_chunks)?;

            // 刷新文件
            handle_io_error(file.flush())?;

            Ok((new_metadata, chunk_info))
        })?;

        // 更新 self.chunks
        {
            let mut chunks_lock = self.chunks.write().unwrap();
            *chunks_lock = metadata_clone.chunks;
        }

        Ok(())
    }

    /// Writes vector data to the file.
    ///
    /// Args:
    ///     name (str): Name of the vector
    ///     data (numpy.ndarray): Vector data to write
    ///
    /// The first dimension of the vector must match the number of rows in the file.
    #[pyo3(text_signature = "(self, name, data)")]
    fn write_vec(&mut self, name: String, data: &PyAny, py: Python) -> PyResult<()> {
        // 验证输入是否为 numpy array
        if !data.hasattr("__array_interface__")? {
            return Err(PyValueError::new_err("Input must be a numpy array"));
        }

        // 获取数组接口和形状信息
        let array_interface = data.getattr("__array_interface__")?;
        let shape: Vec<usize> = array_interface.get_item("shape")?.extract()?;
        let typestr: String = array_interface.get_item("typestr")?.extract()?;
        
        // 获取数据指针和长度
        let data_ptr = array_interface.get_item("data")?
            .extract::<(usize, bool)>()?;
        let ptr = data_ptr.0 as *const u8;
        let total_bytes = shape.iter().product::<usize>() * match typestr.as_str() {
            "<f4" | "float32" => 4,
            "<f8" | "float64" => 8,
            _ => return Err(PyValueError::new_err(format!("Unsupported dtype: {}", typestr))),
        };

        // 验证维度
        if shape.len() < 2 || shape.len() > 3 {
            return Err(PyValueError::new_err("Vector must be 2D or 3D"));
        }

        // 验证行数是否匹配
        let (n_rows, _) = self.shape()?;
        if shape[0] != n_rows {
            return Err(PyValueError::new_err(
                format!("First dimension must match number of rows (expected {}, got {})", 
                    n_rows, shape[0])
            ));
        }

        // 使用 Python GIL 保护的方式读取数据
        let data_vec = unsafe {
            std::slice::from_raw_parts(ptr, total_bytes).to_vec()
        };

        py.allow_threads(|| {
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&self.filepath)?;

            // 读取现有元数据
            let mut metadata = self.read_metadata(&mut file)?;

            // 如果是第一个向量，创建向量区域
            if metadata.vec_region.is_none() {
                file.seek(std::io::SeekFrom::End(-12))?;
                let footer_pos = file.stream_position()?;
                metadata.vec_region = Some(VecRegionInfo {
                    offset: footer_pos,
                    size: 0,
                    vectors: Vec::new(),
                });
            }

            // 获取向量写入位置
            let vec_region = metadata.vec_region.as_mut().unwrap();
            let vector_offset = vec_region.offset + vec_region.size;

            // 预分配文件空间
            file.set_len(vector_offset + total_bytes as u64)?;
            file.seek(std::io::SeekFrom::Start(vector_offset))?;
            
            // 使用作用域来确保 writer 被正确释放
            {
                let mut writer = std::io::BufWriter::with_capacity(1024 * 1024, &mut file);
                writer.write_all(&data_vec)?;
                writer.flush()?;
            }

            // 更新向量信息
            vec_region.vectors.push(VectorInfo {
                name,
                offset: vector_offset,
                size: total_bytes as u64,
                shape,
                dtype: typestr,
            });
            vec_region.size += total_bytes as u64;

            // 重写 footer
            let new_pos = vector_offset + total_bytes as u64;
            file.seek(std::io::SeekFrom::Start(new_pos))?;
            
            // 序列化并写入元数据
            let metadata_bytes = serde_json::to_vec(&metadata)
                .map_err(|e| PyValueError::new_err(format!("Failed to serialize metadata: {}", e)))?;
            let metadata_length = metadata_bytes.len() as u32;
            
            file.write_all(&metadata_bytes)?;
            file.write_u32::<LittleEndian>(metadata_length)?;
            file.write_all(MAGIC_BYTES)?;
            file.flush()?;

            Ok(())
        })
    }

    /// Appends vector data to the existing file.
    ///
    /// Args:
    ///     data (dict): Dictionary where keys are vector names and values are numpy.ndarray
    ///
    /// Raises:
    ///     ValueError: If the input data type is not supported or if vector names do not match.
    ///     IOError: If an error occurs while writing to the file.
    #[pyo3(text_signature = "(self, data)")]
    fn _append_vec(&mut self, data: &PyAny, py: Python) -> PyResult<()> {
        // 验证输入是否为字典
        if !data.is_instance_of::<pyo3::types::PyDict>() {
            return Err(PyValueError::new_err("Input must be a dictionary"));
        }

        let data_dict: &pyo3::types::PyDict = data.downcast()?;
        let vector_names: Vec<String> = data_dict.keys().iter()
            .map(|k| k.extract::<String>())
            .collect::<Result<_, _>>()?;

        // 读取现有的向量信息
        let mut file = File::open(&self.filepath)?;
        let mut metadata = self.read_metadata(&mut file)?;

        let vec_region = metadata.vec_region
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("No vector data in file"))?;

        // 验证向量名称是否匹配
        if vector_names.len() != vec_region.vectors.len() {
            return Err(PyValueError::new_err("Number of vectors does not match"));
        }

        for vector_info in &vec_region.vectors {
            if !vector_names.contains(&vector_info.name) {
                return Err(PyValueError::new_err(format!("Vector '{}' not found in input data", vector_info.name)));
            }
        }

        // 追加数据
        for vector_info in &mut vec_region.vectors {
            let array = data_dict.get_item(&vector_info.name).unwrap();
            if !array.hasattr("__array_interface__")? {
                return Err(PyValueError::new_err("All values must be numpy arrays"));
            }

            // 获取数组接口和形状信息
            let array_interface = array.getattr("__array_interface__")?;
            let shape: Vec<usize> = array_interface.get_item("shape")?.extract()?;
            let typestr: String = array_interface.get_item("typestr")?.extract()?;

            // 验证数据类型是否匹配
            if typestr != vector_info.dtype {
                return Err(PyValueError::new_err(format!("Data type mismatch for vector '{}'", vector_info.name)));
            }

            // 使用 numpy 的 tofile 方法直接写入临时文件
            let temp_path = format!("{}.temp.vec", self.filepath);
            array.call_method1("tofile", (temp_path.as_str(),))?;

            // 读取临时文件内容
            py.allow_threads(|| -> PyResult<()> {
                let mut temp_file = File::open(&temp_path)?;
                let temp_metadata = temp_file.metadata()?;
                let data_len = temp_metadata.len() as usize;
                let mut data_vec = vec![0u8; data_len];
                temp_file.read_exact(&mut data_vec)?;

                // 删除临时文件
                std::fs::remove_file(&temp_path)?;

                // 打开目标文件
                let file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&self.filepath)?;

                // 获取向量写入位置
                let vector_offset = vector_info.offset + vector_info.size;

                // 预分配文件空间
                file.set_len(vector_offset + data_len as u64)?;

                // 使用内存映射进行快速写入
                let mut mmap = unsafe {
                    MmapOptions::new()
                        .offset(vector_offset)
                        .len(data_len)
                        .map_mut(&file)?
                };

                // 一次性写入所有数据
                mmap.copy_from_slice(&data_vec);
                mmap.flush()?;

                // 更新向量信息
                vector_info.size += data_len as u64;
                vector_info.shape[0] += shape[0]; // 假设追加行数

                Ok(())
            })?;
        }

        // 重写 footer
        let new_pos = vec_region.offset + vec_region.size;
        file.seek(std::io::SeekFrom::Start(new_pos))?;
        
        // 序列化并写入元数据
        let metadata_bytes = serde_json::to_vec(&metadata)
            .map_err(|e| PyValueError::new_err(format!("Failed to serialize metadata: {}", e)))?;
        let metadata_length = metadata_bytes.len() as u32;
        
        file.write_all(&metadata_bytes)?;
        file.write_u32::<LittleEndian>(metadata_length)?;
        file.write_all(MAGIC_BYTES)?;
        file.flush()?;

        Ok(())
    }
    
    /// Reads vector data from the file.
    ///
    /// Args:
    ///     name (str): Name of the vector to read
    ///
    /// Returns:
    ///     numpy.ndarray: The vector data
    #[pyo3(text_signature = "(self, name)")]
    fn read_vec(&self, name: String, py: Python) -> PyResult<PyObject> {
        let mut file = File::open(&self.filepath)?;
        let metadata = self.read_metadata(&mut file)?;
        
        let vec_region = metadata.vec_region
            .ok_or_else(|| PyValueError::new_err("No vector data in file"))?;
        
        let vector_info = vec_region.vectors
            .iter()
            .find(|v| v.name == name)
            .ok_or_else(|| PyValueError::new_err(format!("Vector '{}' not found", name)))?;
        
        // 使用 memmap 读取数据
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .offset(vector_info.offset)
                .len(vector_info.size as usize)
                .map(&file)?
        };
        
        // 获取 numpy
        let np = py.import("numpy")?;
        
        // 创建正确的数据类型
        let dtype = match vector_info.dtype.as_str() {
            "<f4" | "float32" => "float32",
            "<f8" | "float64" => "float64",
            _ => return Err(PyValueError::new_err(format!("Unsupported dtype: {}", vector_info.dtype))),
        };
        
        // 使用 frombuffer 创建 numpy 数组
        let array = np.getattr("frombuffer")?.call1((
            pyo3::types::PyBytes::new(py, &mmap),
            dtype,
        ))?;
        
        // 重塑数组
        let shape_tuple = PyTuple::new(py, &vector_info.shape);
        let reshaped = array.call_method1("reshape", (shape_tuple,))?;
        
        // 确保返回的是连续的数组
        let result = reshaped.call_method0("copy")?;
        
        Ok(result.into())
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
    ///     query_vectors (numpy.ndarray): 查询向量，shape为(n_queries, dim)
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
}


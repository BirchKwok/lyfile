use std::fs::File;
use std::io::{self, Write, Seek, Read, Cursor};
use std::sync::{Arc, RwLock};
use std::fs::OpenOptions;
use std::path::Path;

use arrow::datatypes::{Schema, SchemaRef, Field, DataType};
use arrow::record_batch::RecordBatch;
use arrow::ipc::{writer, reader};
use arrow::error::ArrowError;
use arrow::pyarrow::{PyArrowType, ToPyArrow};

use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::exceptions::PyValueError;

use serde::{Serialize, Deserialize};
use serde_json;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use zstd::stream::{Encoder as ZstdEncoder, Decoder as ZstdDecoder};

use rayon::prelude::*;

const MAGIC_BYTES: &[u8] = b"LYFILE01";
const VERSION: u32 = 1;
const FOOTER_SIZE: usize = 8; // 存储 footer 位置的固定字节数
const COMPRESSION_LEVEL: i32 = 1; // 压缩级别，1 为最快压缩
const FILE_HEADER_SIZE: usize = 0x28; // 40 bytes
const CHUNK_MAGIC: &[u8] = b"LYCHUNK1";
const PAGE_MAGIC: &[u8] = b"LYPAGE01";
const FEATURE_FLAGS: u32 = 0;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableSchema {
    fields: Vec<SerializableField>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableField {
    name: String,
    data_type: String,
    nullable: bool,
}

impl From<&Schema> for SerializableSchema {
    fn from(schema: &Schema) -> Self {
        SerializableSchema {
            fields: schema.fields().iter()
                .map(|f| SerializableField {
                    name: f.name().clone(),
                    data_type: format!("{:?}", f.data_type()),
                    nullable: f.is_nullable(),
                })
                .collect(),
        }
    }
}

impl SerializableSchema {
    fn to_arrow_schema(&self) -> Schema {
        Schema::new(
            self.fields.iter()
                .map(|f| {
                    Field::new(
                        &f.name,
                        parse_datatype(&f.data_type),
                        f.nullable,
                    )
                })
                .collect::<Vec<Field>>()
        )
    }

    fn get_fields(&self) -> &Vec<SerializableField> {
        &self.fields
    }
}

fn parse_datatype(type_str: &str) -> DataType {
    // 处理 List 类型
    if type_str.starts_with("List(") {
        // 提取内部类型
        let inner_type_str = &type_str[5..type_str.len() - 1]; // 去掉 "List(" 和 ")"
        // 解析内部的 Field
        let field_str = inner_type_str.trim_start_matches("Field { name: \"item\", data_type: ").trim_end_matches(", nullable: true, dict_id: 0, dict_is_ordered: false, metadata: {} }");
        let inner_data_type = parse_datatype(field_str);
        return DataType::List(Arc::new(Field::new("item", inner_data_type, true)));
    }

    match type_str {
        "Int8" => DataType::Int8,
        "Int16" => DataType::Int16,
        "Int32" => DataType::Int32,
        "Int64" => DataType::Int64,
        "UInt8" => DataType::UInt8,
        "UInt16" => DataType::UInt16,
        "UInt32" => DataType::UInt32,
        "UInt64" => DataType::UInt64,
        "Float32" => DataType::Float32,
        "Float64" => DataType::Float64,
        "Boolean" => DataType::Boolean,
        "Utf8" => DataType::Utf8,
        "Binary" => DataType::Binary,
        "Date32" => DataType::Date32,
        "Date64" => DataType::Date64,
        _ => {
            println!("Unknown type: {}", type_str);
            DataType::Null
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Metadata {
    schema: SerializableSchema,
    chunks: Vec<ChunkInfo>,
    // 其他元数据
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChunkInfo {
    offset: u64,
    size: u64,
    rows: usize,
    columns: Vec<ColumnInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ColumnInfo {
    name: String,
    offset: u64,
    size: u64,
    compressed: bool, // 新增字段，标识该列是否压缩
}

fn handle_arrow_error<T>(result: Result<T, ArrowError>) -> PyResult<T> {
    result.map_err(|e| PyValueError::new_err(format!("Arrow error: {}", e)))
}

fn handle_io_error<T>(result: Result<T, io::Error>) -> PyResult<T> {
    result.map_err(|e| PyValueError::new_err(format!("IO error: {}", e)))
}

fn handle_serde_error<T>(result: Result<T, serde_json::Error>) -> PyResult<T> {
    result.map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))
}

#[pyclass]
struct LyFile {
    filepath: String,
    schema: Arc<RwLock<Option<SchemaRef>>>,
    chunks: Arc<RwLock<Vec<ChunkInfo>>>,
}

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

    /// Writes data to the custom file format.
    ///
    /// Args:
    ///     data (Union[pandas.DataFrame, dict, pyarrow.Table]):
    ///         The input data to be written.
    ///         Supported types include:
    ///         - Pandas DataFrame
    ///         - Python dictionary
    ///         - PyArrow Table
    ///
    /// Raises:
    ///     ValueError: If the input data type is not supported or is empty.
    ///     IOError: If an error occurs while writing to the file.
    ///     ArrowError: If there is an error with Arrow serialization.
    ///     SerializationError: If an error occurs during metadata serialization.
    ///
    /// Examples:
    ///     >>> import pandas as pd
    ///     >>> from lyfile import LyFile
    ///     >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    ///     >>> lyfile = LyFile("example.ly")
    ///     >>> lyfile.write(df)
    #[pyo3(text_signature = "(self, data)")]
    fn write(&mut self, data: &PyAny, py: Python) -> PyResult<()> {
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
        let (schema_clone, metadata_clone) = py.allow_threads(|| -> PyResult<(SchemaRef, Metadata)> {
            let schema = record_batch.schema();
            let serialized_schema = handle_serde_error(serde_json::to_vec(&SerializableSchema::from(schema.as_ref())))?;
            let schema_length = serialized_schema.len() as u32;

            // 打开或创建文件
            let mut file = handle_io_error(
                OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .open(&self.filepath)
            )?;

            // 移动文件指针到文件开头
            handle_io_error(file.seek(std::io::SeekFrom::Start(0)))?;

            // 写入文件头占位符
            let header_placeholder = vec![0u8; FILE_HEADER_SIZE];
            handle_io_error(file.write_all(&header_placeholder))?;

            // 写入 Schema
            handle_io_error(file.write_all(&serialized_schema))?;

            // 数据区域的起始偏移
            let data_offset = FILE_HEADER_SIZE as u64 + schema_length as u64;
            let mut current_offset = data_offset;

            // 写入数据区域（Chunks）
            let num_chunks = 1; // 这里假设只有一个 Chunk，可根据需要拆分
            let mut chunks = Vec::new();

            for _ in 0..num_chunks {
                let chunk_offset = current_offset;
                let chunk_info = self.write_chunk(&mut file, &record_batch, &mut current_offset)?;
                chunks.push((chunk_offset, chunk_info));
            }

            // 索引区域的起始偏移
            let index_offset = current_offset;

            // 写入索引区域
            self.write_index_region(&mut file, &chunks)?;

            // 写入 Footer
            let metadata = Metadata {
                schema: SerializableSchema::from(schema.as_ref()),
                chunks: chunks.iter().map(|(_, info)| info.clone()).collect(),
            };
            self.write_footer(&mut file, &metadata)?;

            // 回到文件开头，写入实际的文件头信息
            handle_io_error(file.seek(std::io::SeekFrom::Start(0)))?;
            self.write_file_header(
                &mut file,
                data_offset,
                index_offset,
                schema_length,
                FEATURE_FLAGS,
                num_chunks as u32,
            )?;

            // 刷新文件
            handle_io_error(file.flush())?;

            Ok((schema.clone(), metadata))
        })?;

        // 更新 self.schema 和 self.chunks
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
            // 打开文件，进行读写操作
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

            // 定位到元数据的开始位置，准备追加新的 Chunk
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
}


impl LyFile {
    fn write_file_header(
        &self,
        file: &mut File,
        data_offset: u64,
        index_offset: u64,
        schema_length: u32,
        feature_flags: u32,
        num_chunks: u32,
    ) -> PyResult<()> {
        let mut buffer = Vec::with_capacity(FILE_HEADER_SIZE);

        // MAGIC (8 bytes)
        buffer.extend_from_slice(MAGIC_BYTES);

        // 版本号 (4 bytes)
        buffer.write_u32::<LittleEndian>(VERSION)?;

        // 数据偏移 (8 bytes)
        buffer.write_u64::<LittleEndian>(data_offset)?;

        // 索引偏移 (8 bytes)
        buffer.write_u64::<LittleEndian>(index_offset)?;

        // Schema 长度 (4 bytes)
        buffer.write_u32::<LittleEndian>(schema_length)?;

        // 特征标记 (4 bytes)
        buffer.write_u32::<LittleEndian>(feature_flags)?;

        // Chunk 数量 (4 bytes)
        buffer.write_u32::<LittleEndian>(num_chunks)?;

        // 写入文件头
        handle_io_error(file.write_all(&buffer))?;

        Ok(())
    }

    fn write_chunk(
        &self,
        file: &mut File,
        record_batch: &RecordBatch,
        current_offset: &mut u64,
    ) -> PyResult<ChunkInfo> {
        // 写入 CHUNK_MAGIC
        handle_io_error(file.write_all(CHUNK_MAGIC))?;
        *current_offset += CHUNK_MAGIC.len() as u64;

        // 占位 Chunk 大小
        let size_position = *current_offset;
        file.write_u32::<LittleEndian>(0)?; // 占位符
        *current_offset += 4;

        // 行数
        let num_rows = record_batch.num_rows() as u32;
        file.write_u32::<LittleEndian>(num_rows)?;
        *current_offset += 4;

        // 列数
        let num_columns = record_batch.num_columns() as u32;
        file.write_u32::<LittleEndian>(num_columns)?;
        *current_offset += 4;

        // 占位统计信息偏移
        file.write_u64::<LittleEndian>(0)?; // 占位符
        *current_offset += 8;

        // 并行处理每一列，生成压缩后的数据缓冲区
        let columns_data: Vec<(String, Vec<u8>, bool)> = record_batch
            .columns()
            .par_iter()
            .enumerate()
            .map(|(i, column)| {
                let schema = record_batch.schema();
                let field = schema.field(i);
                let column_name = field.name().clone();

                // 判断列是否需要压缩
                let data_type = field.data_type();
                let should_compress = is_compressible(data_type);

                // 将列数据拆分为 Pages（目前每列只有一个 Page）
                let pages = self.divide_into_pages(column)?;

                let mut column_buffer = Vec::new();

                for page in pages {
                    let mut page_buffer = Vec::new();

                    // 写入 PAGE_MAGIC
                    page_buffer.extend_from_slice(PAGE_MAGIC);

                    // 占位 Page 大小
                    page_buffer.write_u32::<LittleEndian>(0)?; // 占位符

                    // Page 行数
                    let page_num_rows = page.num_rows() as u32;
                    page_buffer.write_u32::<LittleEndian>(page_num_rows)?;

                    // 序列化 Row Group 数据
                    let row_group_data = self.serialize_row_group(&page)?;

                    // 如果需要压缩，进行压缩
                    let final_data = if should_compress {
                        compress_data(&row_group_data)?
                    } else {
                        row_group_data
                    };

                    // 写入压缩标记（1 字节，0 表示未压缩，1 表示压缩）
                    page_buffer.write_u8(if should_compress { 1 } else { 0 })?;

                    // 写入数据长度（4 字节）
                    page_buffer.write_u32::<LittleEndian>(final_data.len() as u32)?;

                    // 写入数据
                    page_buffer.extend_from_slice(&final_data);

                    // 填写 Page 大小
                    let page_size = (page_buffer.len() - PAGE_MAGIC.len() - 4) as u32; // 除去 PAGE_MAGIC 和 Page 大小占位符

                    // 写入实际的 Page 大小
                    let page_size_pos = PAGE_MAGIC.len();
                    let mut page_size_bytes = &mut page_buffer[page_size_pos..page_size_pos + 4];
                    page_size_bytes.write_u32::<LittleEndian>(page_size)?;

                    // 将 Page 数据添加到列缓冲区
                    column_buffer.extend_from_slice(&page_buffer);
                }

                Ok((column_name, column_buffer, should_compress))
            })
            .collect::<PyResult<Vec<(String, Vec<u8>, bool)>>>()?;

        let mut column_infos = Vec::new();

        // 顺序写入列数据，并更新偏移量
        for (column_name, column_buffer, compressed) in columns_data {
            let column_offset = *current_offset;

            // 写入列数据
            handle_io_error(file.write_all(&column_buffer))?;
            let column_size = column_buffer.len() as u64;
            *current_offset += column_size;

            column_infos.push(ColumnInfo {
                name: column_name,
                offset: column_offset,
                size: column_size,
                compressed,
            });
        }

        // 填写 Chunk 大小
        let end_position = handle_io_error(file.seek(std::io::SeekFrom::Current(0)))?;
        let chunk_size = (end_position - size_position - 4) as u32;

        handle_io_error(file.seek(std::io::SeekFrom::Start(size_position)))?;
        file.write_u32::<LittleEndian>(chunk_size)?;

        // 返回到 Chunk 末尾
        handle_io_error(file.seek(std::io::SeekFrom::Start(end_position)))?;

        // 构造并返回 ChunkInfo
        let chunk_info = ChunkInfo {
            offset: size_position - CHUNK_MAGIC.len() as u64,
            size: chunk_size as u64 + CHUNK_MAGIC.len() as u64 + 4,
            rows: num_rows as usize,
            columns: column_infos,
        };

        Ok(chunk_info)
    }

    fn write_index_region(
        &self,
        file: &mut File,
        chunks: &Vec<(u64, ChunkInfo)>,
    ) -> PyResult<()> {
        // 写入 Chunk 数量
        let num_chunks = chunks.len() as u32;
        file.write_u32::<LittleEndian>(num_chunks)?;

        // 写入每个 Chunk 的索引信息
        for (offset, _chunk_info) in chunks {
            // 写入 Chunk 偏移
            file.write_u64::<LittleEndian>(*offset)?;
            // 可根据需要写入其他索引信息
        }

        Ok(())
    }

    // 添加辅助方法，用于读取元数据
    fn read_metadata(&self, file: &mut File) -> PyResult<Metadata> {
        // 读取 Footer 中的元数据
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
        Ok(metadata)
    }

    // 添加辅助方法，用于更新文件头部的 Chunk 数量
    fn update_chunk_count_in_header(&self, file: &mut File, num_chunks: u32) -> PyResult<()> {
        // 文件头部 Chunk 数量的偏移量为：
        // MAGIC (8 bytes) + VERSION (4 bytes) + DATA_OFFSET (8 bytes) + INDEX_OFFSET (8 bytes) + SCHEMA_LENGTH (4 bytes) + FEATURE_FLAGS (4 bytes)
        let chunk_count_offset = 8 + 4 + 8 + 8 + 4 + 4;
        handle_io_error(file.seek(std::io::SeekFrom::Start(chunk_count_offset as u64)))?;
        file.write_u32::<LittleEndian>(num_chunks)?;
        Ok(())
    }

    fn write_footer(&self, file: &mut File, metadata: &Metadata) -> PyResult<()> {
        // 序列化元数据
        let metadata_bytes = handle_serde_error(serde_json::to_vec(metadata))?;
        let metadata_length = metadata_bytes.len() as u32;

        // 写入元数据
        handle_io_error(file.write_all(&metadata_bytes))?;

        // 写入元数据长度
        file.write_u32::<LittleEndian>(metadata_length)?;

        // 写入 MAGIC
        handle_io_error(file.write_all(MAGIC_BYTES))?;

        Ok(())
    }


    fn divide_into_pages(&self, array: &Arc<dyn arrow::array::Array>) -> PyResult<Vec<RecordBatch>> {
        // 这里简单地将整个列作为一个 Page，可根据需要拆分
        let schema = Schema::new(vec![Field::new("column", array.data_type().clone(), true)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![array.clone()])
            .map_err(|e| PyValueError::new_err(format!("Arrow error: {}", e)))?;
        Ok(vec![batch])
    }

    fn serialize_row_group(&self, batch: &RecordBatch) -> PyResult<Vec<u8>> {
        let mut buffer = Vec::new();

        {
            let mut stream_writer = writer::StreamWriter::try_new(&mut buffer, batch.schema().as_ref())
                .map_err(|e| PyValueError::new_err(format!("Arrow error: {}", e)))?;
            stream_writer.write(batch)
                .map_err(|e| PyValueError::new_err(format!("Arrow error: {}", e)))?;
            stream_writer.finish()
                .map_err(|e| PyValueError::new_err(format!("Arrow error: {}", e)))?;
        }

        Ok(buffer)
    }

    fn read_chunk(
        &self,
        file_path: &str,
        arrow_schema: &Schema,
        selected_columns: &Vec<String>,
        chunk_info: &ChunkInfo,
    ) -> PyResult<RecordBatch> {
        // 准备数组和字段
        let arrays_fields: Vec<(Arc<dyn arrow::array::Array>, Field)> = selected_columns
            .par_iter()
            .map(|col_name| {
                // 每个线程独立打开文件
                let mut file = handle_io_error(File::open(file_path))?;

                // 查找对应的 ColumnInfo
                let col_info = chunk_info
                    .columns
                    .iter()
                    .find(|c| &c.name == col_name)
                    .ok_or_else(|| PyValueError::new_err(format!("Column '{}' not found in chunk", col_name)))?;

                // 移动到列的偏移位置
                handle_io_error(file.seek(std::io::SeekFrom::Start(col_info.offset)))?;

                // 读取 PAGE_MAGIC
                let mut page_magic = [0u8; 8];
                handle_io_error(file.read_exact(&mut page_magic))?;
                if page_magic != PAGE_MAGIC {
                    return Err(PyValueError::new_err("Invalid page magic"));
                }

                // 读取 Page 大小和行数
                let page_size = file.read_u32::<LittleEndian>()?;
                let _page_num_rows = file.read_u32::<LittleEndian>()?;

                // 读取压缩标记
                let compressed_flag = file.read_u8()?;
                let is_compressed = compressed_flag == 1;

                // 读取数据长度
                let data_length = file.read_u32::<LittleEndian>()? as usize;

                // 读取数据
                let mut data = vec![0u8; data_length];
                handle_io_error(file.read_exact(&mut data))?;

                // 如果数据被压缩，解压缩
                let row_group_data = if is_compressed {
                    decompress_data(&data)?
                } else {
                    data
                };

                // 反序列化为 RecordBatch
                let cursor = Cursor::new(row_group_data);
                let mut arrow_reader = reader::StreamReader::try_new(cursor, None)
                    .map_err(|e| PyValueError::new_err(format!("Arrow error: {}", e)))?;

                if let Some(batch) = arrow_reader.next() {
                    let batch = batch.map_err(|e| PyValueError::new_err(format!("Arrow error: {}", e)))?;
                    let array = batch.column(0).clone();

                    // 获取字段信息
                    let field = arrow_schema
                        .field_with_name(col_name)
                        .map_err(|_| PyValueError::new_err(format!("Field '{}' not found in schema", col_name)))?;

                    Ok((array, field.clone()))
                } else {
                    Err(PyValueError::new_err("No data in column"))
                }
            })
            .collect::<PyResult<Vec<_>>>()?;

        let (arrays, fields): (Vec<_>, Vec<_>) = arrays_fields.into_iter().unzip();

        // 构建 RecordBatch
        let schema = Schema::new(fields);
        handle_arrow_error(RecordBatch::try_new(Arc::new(schema), arrays))
    }
}

// 辅助函数：判断数据类型是否需要压缩
fn is_compressible(data_type: &DataType) -> bool {
    match data_type {
        DataType::Utf8
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64 => true,
        _ => false,
    }
}

// 辅助函数：压缩数据
fn compress_data(data: &[u8]) -> PyResult<Vec<u8>> {
    let mut encoder = ZstdEncoder::new(Vec::new(), COMPRESSION_LEVEL)
        .map_err(|e| PyValueError::new_err(format!("Zstd compression error: {}", e)))?;
    encoder.write_all(data)
        .map_err(|e| PyValueError::new_err(format!("Zstd compression error: {}", e)))?;
    let compressed_data = encoder.finish()
        .map_err(|e| PyValueError::new_err(format!("Zstd compression error: {}", e)))?;
    Ok(compressed_data)
}

// 辅助函数：解压缩数据
fn decompress_data(data: &[u8]) -> PyResult<Vec<u8>> {
    let mut decoder = ZstdDecoder::new(Cursor::new(data))
        .map_err(|e| PyValueError::new_err(format!("Zstd decompression error: {}", e)))?;
    let mut decompressed_data = Vec::new();
    handle_io_error(decoder.read_to_end(&mut decompressed_data))?;
    Ok(decompressed_data)
}

impl Drop for LyFile {
    fn drop(&mut self) {
        self.schema.write().unwrap().take();
        self.chunks.write().unwrap().clear();
    }
}

#[pymodule]
fn lyfile(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LyFile>()?;
    Ok(())
}

use std::fs::File;
use std::io::{Write, Seek, BufWriter, Cursor};
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

use lz4::block::{compress, decompress};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use rayon::prelude::*;
use std::sync::mpsc;
use memmap2::MmapOptions;

// 定义文件格式常量
const MAGIC_BYTES: &[u8] = b"LYFILE01";  // 新版本
const VERSION: u32 = 1;
const FOOTER_SIZE: u64 = 8;  // 存储 footer 位置的固定字节数
const COMPRESSION_LEVEL: i32 = 4;  // 添加压缩级别常量

// 自定义可序列化的 Schema 结构
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableSchema {
    fields: Vec<SerializableField>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableField {
    name: String,
    data_type: String,  // DataType 序列化为字符串
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

// 辅助函数：解析数据类型字符串
fn parse_datatype(type_str: &str) -> DataType {
    // 首先处理 List 类型
    if type_str.starts_with("List(") {
        // 提取内部类型
        let inner_type = type_str
            .trim_start_matches("List(")
            .trim_end_matches(')')
            .trim_start_matches("Field { name: \"item\", data_type: ")
            .split(',')
            .next()
            .unwrap_or("Null");
        
        return DataType::List(Arc::new(Field::new(
            "item",
            parse_datatype(inner_type),
            true
        )));
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
        "LargeBinary" => DataType::LargeBinary,
        "LargeUtf8" => DataType::LargeUtf8,
        "Date32" => DataType::Date32,
        "Date64" => DataType::Date64,
        "Time32(Second)" => DataType::Time32(arrow::datatypes::TimeUnit::Second),
        "Time32(Millisecond)" => DataType::Time32(arrow::datatypes::TimeUnit::Millisecond),
        "Time64(Microsecond)" => DataType::Time64(arrow::datatypes::TimeUnit::Microsecond),
        "Time64(Nanosecond)" => DataType::Time64(arrow::datatypes::TimeUnit::Nanosecond),
        "Timestamp(Second)" => DataType::Timestamp(arrow::datatypes::TimeUnit::Second, None),
        "Timestamp(Millisecond)" => DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None),
        "Timestamp(Microsecond)" => DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None),
        "Timestamp(Nanosecond)" => DataType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, None),
        "Decimal128(38, 10)" => DataType::Decimal128(38, 10),
        _ => {
            println!("Unknown type: {}", type_str);
            DataType::Null
        }
    }
}

// 定义元数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Metadata {
    schema: SerializableSchema,  // 使用可序列化的 Schema
    chunks: Vec<ChunkInfo>,
    column_index: HashMap<String, ColumnIndex>,
}

// 定义列索引结构
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ColumnIndex {
    name: String,
    offsets: Vec<u64>,  // 每个 chunk 中该列的偏移量
    sizes: Vec<u32>,    // 每个 chunk 中该列的大小
}

// 定义 chunk 信息结构
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChunkInfo {
    offset: u64,
    size: u64,
    rows: usize,
    columns: HashMap<String, ColumnChunkInfo>,
}

// 定义列 chunk 信息
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ColumnChunkInfo {
    offset: u64,
    size: u32,
    compressed: bool,
}

// 添加错误处理函数
fn handle_arrow_error<T>(result: Result<T, ArrowError>) -> PyResult<T> {
    result.map_err(|e| PyValueError::new_err(format!("Arrow error: {}", e)))
}

fn handle_io_error<T>(result: Result<T, std::io::Error>) -> PyResult<T> {
    result.map_err(|e| PyValueError::new_err(format!("IO error: {}", e)))
}

fn handle_serde_error<T>(result: Result<T, serde_json::Error>) -> PyResult<T> {
    result.map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))
}

fn handle_lz4_error<T>(result: Result<T, std::io::Error>) -> PyResult<T> {
    result.map_err(|e| PyValueError::new_err(format!("LZ4 compression error: {}", e)))
}

// 添加新的错误处理函数
fn handle_arrow_batch(batch: Option<Result<RecordBatch, ArrowError>>) -> PyResult<Option<RecordBatch>> {
    match batch {
        Some(result) => Ok(Some(handle_arrow_error(result)?)),
        None => Ok(None),
    }
}

/// Define the LyFile struct
#[pyclass]
struct LyFile {
    filepath: String,
    schema: Arc<RwLock<Option<SchemaRef>>>,
    chunks: Arc<RwLock<Vec<ChunkInfo>>>,
}

#[pymethods]
impl LyFile {
    #[new]
    fn new(filepath: String) -> Self {
        LyFile {
            filepath,
            schema: Arc::new(RwLock::new(None)),
            chunks: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Write data to file with new format
    fn write(&mut self, data: &PyAny, py: Python) -> PyResult<()> {
        // 在释放 GIL 之前获取所有需要的数据
        let record_batch = {
            let table = if data.hasattr("__class__")? {
                let class_name = data.getattr("__class__")?.getattr("__name__")?;
                match class_name.extract::<String>()?.as_str() {
                    "DataFrame" => {
                        // 检查 DataFrame 大小
                        let nrows: usize = data.getattr("shape")?.get_item(0)?.extract()?;
                        let ncols: usize = data.getattr("shape")?.get_item(1)?.extract()?;
                        if nrows > 1_000_000_000 || ncols > 1_000 {
                            return Err(PyValueError::new_err("DataFrame too large"));
                        }
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

            // 在这里完成所有 Python 对象操作
            let batches = table.call_method0("to_batches")?
                .extract::<Vec<PyArrowType<RecordBatch>>>()?;
            if batches.is_empty() {
                return Err(PyValueError::new_err("Empty table"));
            }
            batches[0].0.clone()
        };

        // 现在可以安全地使用 allow_threads
        py.allow_threads(|| -> PyResult<()> {
            // 检查内存使用
            let estimated_memory = record_batch.get_array_memory_size();
            if estimated_memory > 1024 * 1024 * 1024 { // 1GB
                return Err(PyValueError::new_err("Data too large for memory"));
            }

            let schema = record_batch.schema();
            let num_columns = record_batch.num_columns();

            // 创建文件并获取 BufWriter
            let file = handle_io_error(
                OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .open(&self.filepath)
            )?;
            
            // 预估所需空间大小
            let estimated_size = record_batch.get_array_memory_size() * 2;
            handle_io_error(file.set_len(estimated_size as u64))?;

            // 创建可写内存映射
            let mut mmap = handle_io_error(
                unsafe { MmapOptions::new().map_mut(&file) }
            )?;

            let mut current_pos = 0;
            let mut writer = BufWriter::new(&file);

            // 写入文件头
            mmap[current_pos..current_pos + MAGIC_BYTES.len()].copy_from_slice(MAGIC_BYTES);
            current_pos += MAGIC_BYTES.len();
            
            mmap[current_pos..current_pos + 4].copy_from_slice(&VERSION.to_le_bytes());

            // 创建 chunk 信息
            let mut chunk_info = ChunkInfo {
                offset: handle_io_error(writer.stream_position())?,
                size: 0,
                rows: record_batch.num_rows(),
                columns: HashMap::new(),
            };

            // 并行处理列数据
            let (tx, rx) = mpsc::channel();
            
            (0..num_columns).into_par_iter().try_for_each(|i| -> PyResult<()> {
                let column = record_batch.column(i);
                let field = schema.field(i);
                let column_name = field.name().to_string();

                // 处理列数据
                let column_data = {
                    let column_schema = Schema::new(vec![field.clone()]);
                    let column_batch = handle_arrow_error(
                        RecordBatch::try_new(
                            Arc::new(column_schema.clone()),
                            vec![column.clone()]
                        )
                    )?;

                    let mut buffer = Vec::new();
                    {
                        let mut stream_writer = handle_arrow_error(
                            writer::StreamWriter::try_new(&mut buffer, &column_schema)
                        )?;
                        handle_arrow_error(stream_writer.write(&column_batch))?;
                        handle_arrow_error(stream_writer.finish())?;
                    }
                    buffer
                };

                // 压缩数据
                let compressed_data = compress(
                    &column_data,
                    Some(lz4::block::CompressionMode::FAST(COMPRESSION_LEVEL)),
                    true
                ).map_err(|e| PyValueError::new_err(format!("Compression error: {}", e)))?;

                tx.send((i, column_name, compressed_data))
                    .map_err(|e| PyValueError::new_err(format!("Channel error: {}", e)))?;
                
                Ok(())
            })?;

            // 按顺序写入压缩后的数据
            let mut column_infos = HashMap::new();
            for _ in 0..num_columns {
                let (_i, column_name, compressed_data) = rx.recv()
                    .map_err(|e| PyValueError::new_err(format!("Channel receive error: {}", e)))?;
                
                let column_offset = handle_io_error(writer.stream_position())?;
                let column_size = compressed_data.len() as u32;

                column_infos.insert(column_name, ColumnChunkInfo {
                    offset: column_offset,
                    size: column_size,
                    compressed: true,
                });

                handle_io_error(writer.write_all(&column_size.to_le_bytes()))?;
                handle_io_error(writer.write_all(&compressed_data))?;
            }

            chunk_info.columns = column_infos;
            
            // 更新 chunk 大小
            chunk_info.size = handle_io_error(writer.stream_position())? - chunk_info.offset;

            // 写入元数据
            let metadata_offset = handle_io_error(writer.stream_position())?;
            let metadata = Metadata {
                schema: SerializableSchema::from(schema.as_ref()),
                chunks: vec![chunk_info],
                column_index: HashMap::new(),
            };

            let metadata_bytes = handle_serde_error(serde_json::to_vec(&metadata))?;

            // 写入元数据和 footer
            handle_io_error(writer.write_all(&metadata_bytes))?;
            handle_io_error(writer.write_all(&metadata_offset.to_le_bytes()))?;
            handle_io_error(writer.flush())?;

            // 调整文件大小并刷新内存映射
            handle_io_error(file.set_len(writer.stream_position()?))?;
            handle_io_error(mmap.flush())?;

            Ok(())
        })?;

        Ok(())
    }

    /// Read data with new format
    fn read(&self, columns: Option<&PyAny>, py: Python) -> PyResult<PyObject> {
        // 检查文件存在
        if !Path::new(&self.filepath).exists() {
            return Err(PyValueError::new_err("File does not exist"));
        }

        // 在释放 GIL 之前处理列选择
        let selected_column_names = if let Some(col_selection) = columns {
            if col_selection.is_instance_of::<pyo3::types::PyString>() {
                vec![col_selection.extract::<String>()?]
            } else if col_selection.is_instance_of::<pyo3::types::PyList>() {
                col_selection.extract::<Vec<String>>()?
            } else {
                return Err(PyValueError::new_err(
                    "columns must be either a string or a list of strings"
                ));
            }
        } else {
            Vec::new() // 空向量表示所有列
        };

        // 现在可以安全地释放 GIL
        let result = py.allow_threads(move || -> PyResult<RecordBatch> {
            let file = handle_io_error(File::open(&self.filepath))?;
            
            // 创建只读内存映射
            let mmap = handle_io_error(
                unsafe { MmapOptions::new().map(&file) }
            )?;

            // 读取 footer 位置
            let footer_pos = mmap.len() - FOOTER_SIZE as usize;
            let metadata_offset = u64::from_le_bytes(
                mmap[footer_pos..footer_pos + 8].try_into().unwrap()
            );

            // 读取元数据
            let metadata_bytes = &mmap[metadata_offset as usize..footer_pos];
            let metadata: Metadata = handle_serde_error(serde_json::from_slice(metadata_bytes))?;

            if metadata.schema.get_fields().is_empty() {
                return Err(PyValueError::new_err("Invalid schema: no fields"));
            }

            // 确定要读取的列
            let selected_columns = if selected_column_names.is_empty() {
                metadata.schema.get_fields().iter()
                    .map(|f| f.name.clone())
                    .collect::<Vec<_>>()
            } else {
                // 验证选中的列
                for column_name in &selected_column_names {
                    if !metadata.schema.get_fields().iter().any(|f| f.name == *column_name) {
                        return Err(PyValueError::new_err(format!(
                            "Column '{}' not found in schema", column_name
                        )));
                    }
                }
                selected_column_names
            };

            let arrow_schema = metadata.schema.to_arrow_schema();

            // 从元数据中获取第一个 chunk
            if metadata.chunks.is_empty() {
                return Err(PyValueError::new_err("No chunks found in file"));
            }
            let chunk = &metadata.chunks[0]; // 获取第一个 chunk

            // 并行读取列数据
            let (tx, rx) = mpsc::channel();
            selected_columns.par_iter().try_for_each(|column_name| -> PyResult<()> {
                if let Some(col_info) = chunk.columns.get(column_name) {
                    let start = col_info.offset as usize + 4; // Skip size bytes
                    let end = start + col_info.size as usize;
                    let compressed_data = &mmap[start..end];
                    
                    // 解压缩和处理数据
                    let column_data = handle_lz4_error(decompress(compressed_data, None))?;
                    
                    // 使用正确的参数类型
                    let mut arrow_reader = handle_arrow_error(
                        reader::StreamReader::try_new(
                            Cursor::new(&column_data),
                            None
                        )
                    )?;

                    // 验证读取的数据
                    if let Some(batch) = handle_arrow_batch(arrow_reader.next())? {
                        if batch.num_columns() > 0 {
                            tx.send((column_name.clone(), batch.column(0).clone()))
                                .map_err(|e| PyValueError::new_err(format!("Channel error: {}", e)))?;
                        } else {
                            return Err(PyValueError::new_err(format!(
                                "Empty batch for column '{}'", column_name
                            )));
                        }
                    } else {
                        return Err(PyValueError::new_err(format!(
                            "No data found for column '{}'", column_name
                        )));
                    }
                }
                Ok(())
            })?;

            // 按顺序收集结果
            let mut selected_arrays = Vec::new();
            let mut selected_fields = Vec::new();
            for _ in 0..selected_columns.len() {
                let (column_name, array) = rx.recv()
                    .map_err(|e| PyValueError::new_err(format!("Channel receive error: {}", e)))?;
                
                let field = arrow_schema.fields().iter()
                    .find(|f| f.name().as_str() == column_name.as_str())
                    .ok_or_else(|| PyValueError::new_err(format!(
                        "Field '{}' not found in schema", column_name
                    )))?;

                selected_arrays.push(array);
                selected_fields.push(field.clone());
            }

            let selected_schema = Schema::new(selected_fields);
            handle_arrow_error(
                RecordBatch::try_new(Arc::new(selected_schema), selected_arrays)
            )
        })?;

        // 转换为 PyArrow 对象
        Python::with_gil(|py| {
            let py_record_batch = result.to_pyarrow(py)
                .map_err(|e| PyValueError::new_err(format!("PyArrow conversion error: {}", e)))?;
            let pa = PyModule::import(py, "pyarrow")?;
            let table = pa.getattr("Table")?.call_method1("from_batches", ([py_record_batch],))?;
            Ok(table.into())
        })
    }

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
}

// 添加清理函数
impl Drop for LyFile {
    fn drop(&mut self) {
        // 确保所有资源都被正确释放
        self.schema.write().unwrap().take();
        self.chunks.write().unwrap().clear();
    }
}

#[pymodule]
fn lyfile(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LyFile>()?;
    Ok(())
}

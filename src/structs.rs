use std::io::{self, Write, Read};
use std::error::Error;
use std::sync::{Arc, RwLock};
use arrow::datatypes::{Schema, SchemaRef, Field, DataType};
use arrow::error::ArrowError;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use serde::{Serialize, Deserialize};
use serde_json;
use zstd::stream::{Encoder as ZstdEncoder, Decoder as ZstdDecoder};

pub const CHUNK_MAX_ROWS: usize = 10000;

pub const MAGIC_BYTES: &[u8] = b"LYFILE01";
pub const VERSION: u32 = 1;
pub const COMPRESSION_LEVEL: i32 = 1;
pub const FILE_HEADER_SIZE: usize = 0x28;
pub const CHUNK_MAGIC: [u8; 8] = *b"LYCHUNK0";
pub const PAGE_MAGIC: [u8; 8] = *b"LYPAGE00";
pub const FEATURE_FLAGS: u32 = 0;

pub const BASE_CHUNK_SIZE: usize = 32 * 1024;
pub const L1_CACHE_LINE: usize = 64;
pub const L2_CACHE_SIZE: usize = 256 * 1024;
pub const L3_CACHE_SIZE: usize = 8 * 1024 * 1024;
pub const MIN_PARALLEL_SIZE: usize = L2_CACHE_SIZE;
pub const PREFETCH_DISTANCE: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableSchema {
    pub fields: Vec<SerializableField>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableField {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
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
    pub fn to_arrow_schema(&self) -> Schema {
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
}

pub fn parse_datatype(type_str: &str) -> DataType {
    // handle List type
    if type_str.starts_with("List(") {
        // extract inner type
        let inner_type_str = &type_str[5..type_str.len() - 1]; // remove "List(" and ")"
        // parse inner field
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
pub struct Metadata {
    pub schema: SerializableSchema,
    pub chunks: Vec<ChunkInfo>,
    pub vec_region: Option<VecRegionInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkInfo {
    pub offset: u64,
    pub size: u64,
    pub rows: usize,
    pub columns: Vec<ColumnInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnInfo {
    pub name: String,
    pub offset: u64,
    pub size: u64,
    pub compressed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VecRegionInfo {
    pub offset: u64,
    pub size: u64,
    pub vectors: Vec<VectorInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorInfo {
    pub name: String,
    pub offset: u64,
    pub size: u64,
    pub shape: Vec<usize>,  // [n, m] or [n, m, o]
    pub dtype: String,      // data type, like "f32", "f64"
}

pub fn handle_io_error<T>(result: Result<T, io::Error>) -> PyResult<T> {
    result.map_err(|e| PyValueError::new_err(format!("IO error: {}", e)))
}

pub fn handle_serde_error<T>(result: Result<T, serde_json::Error>) -> PyResult<T> {
    result.map_err(|e| PyValueError::new_err(format!("Serialization error: {}", e)))
}


// 添加新的枚举类型用于返回值
#[derive(Debug)]
pub enum VectorData {
    F32(Vec<f32>, Vec<usize>),
    F64(Vec<f64>, Vec<usize>),
}


#[pyclass]
pub struct _LyFile {
    pub filepath: String,
    pub schema: Arc<RwLock<Option<SchemaRef>>>,
    pub chunks: Arc<RwLock<Vec<ChunkInfo>>>,
}

// helper function: determine if data needs compression
pub fn is_compressible(data_type: &DataType) -> bool {
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

// helper function: compress data
pub fn compress_data(data: &[u8]) -> PyResult<Vec<u8>> {
    let mut encoder = ZstdEncoder::new(Vec::new(), COMPRESSION_LEVEL)
        .map_err(|e| PyValueError::new_err(format!("Failed to create zstd encoder: {}", e)))?;
    
    encoder.write_all(data)
        .map_err(|e| PyValueError::new_err(format!("Failed to write data to encoder: {}", e)))?;
    
    encoder.finish()
        .map_err(|e| PyValueError::new_err(format!("Failed to finish compression: {}", e)))
}

// helper function: decompress data
pub fn decompress_data(data: &[u8]) -> PyResult<Vec<u8>> {
    let mut decoder = ZstdDecoder::new(data)
        .map_err(|e| PyValueError::new_err(format!("Failed to create zstd decoder: {}", e)))?;
    
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)
        .map_err(|e| PyValueError::new_err(format!("Failed to decompress data: {}", e)))?;
    
    Ok(decompressed)
}

impl Drop for _LyFile {
    fn drop(&mut self) {
        self.schema.write().unwrap().take();
        self.chunks.write().unwrap().clear();
    }
}

// add Arrow error conversion function
pub fn convert_arrow_error(err: ArrowError) -> PyErr {
    PyValueError::new_err(format!("Arrow error: {}", err))
}

// 定义自定义错误类型
#[derive(Debug)]
pub enum VectorError {
    IoError(io::Error),
    VectorNotFound(String),
    UnsupportedDataType(String),
    NoVectorRegion,
}

impl From<VectorError> for PyErr {
    fn from(err: VectorError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

impl std::fmt::Display for VectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorError::IoError(e) => write!(f, "IO error: {}", e),
            VectorError::VectorNotFound(name) => write!(f, "Vector '{}' not found", name),
            VectorError::UnsupportedDataType(dtype) => write!(f, "Unsupported dtype: {}. Only float32 and float64 are supported.", dtype),
            VectorError::NoVectorRegion => write!(f, "No vector data in file"),
        }
    }
}

impl Error for VectorError {}

impl From<io::Error> for VectorError {
    fn from(err: io::Error) -> VectorError {
        VectorError::IoError(err)
    }
}


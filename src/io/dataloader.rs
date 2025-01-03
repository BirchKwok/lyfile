use rayon::slice::ParallelSliceMut;

use std::alloc::{alloc, Layout};
use std::fs::File;
use std::io::{self, Seek, Read, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use arrow::datatypes::{Schema, SchemaRef, Field, DataType};
use arrow::record_batch::RecordBatch;

use memmap2::MmapOptions;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use byteorder::{LittleEndian, ReadBytesExt};

use arrow::pyarrow::IntoPyArrow;

use arrow::array::{
    Array, ArrayBuilder, ListBuilder,
    Int8Builder, Int16Builder, Int32Builder, Int64Builder,
    UInt8Builder, UInt16Builder, UInt32Builder, UInt64Builder,
    Float16Builder, Float32Builder, Float64Builder,
    BooleanBuilder, StringBuilder, BinaryBuilder,
    Date32Builder, Date64Builder,
    
    Int8Array, Int16Array, Int32Array, Int64Array,
    UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    Float16Array, Float32Array, Float64Array,
    BooleanArray, StringArray, BinaryArray,
    Date32Array, Date64Array, ListArray,
};

use arrow::buffer::Buffer;
use arrow::array::ArrayData;

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::hint::black_box;
use crate::structs::*;
use std::sync::Mutex;
use std::collections::HashMap;
use std::time::SystemTime;
use lazy_static::lazy_static;
use half::f16;

lazy_static! {
    static ref INDEX_CACHE: Mutex<HashMap<String, IndexCache>> = Mutex::new(HashMap::new());
}

macro_rules! append_value {
    ($column:expr, $row:expr, $array_type:ty, $builder_type:ty, $builder:expr) => {
        if let Some(array) = $column.as_any().downcast_ref::<$array_type>() {
            if array.is_null($row) {
                $builder.as_any_mut()
                    .downcast_mut::<$builder_type>()
                    .unwrap()
                    .append_null();
            } else {
                $builder.as_any_mut()
                    .downcast_mut::<$builder_type>()
                    .unwrap()
                    .append_value(array.value($row));
            }
        }
    };
}

impl _LyFile {
    pub fn read_chunk(
        &self,
        filepath: &str,
        schema: &SchemaRef,
        selected_columns: &[String],
        chunk_info: &ChunkInfo,
    ) -> PyResult<RecordBatch> {
        let mut file = File::open(filepath)?;
        let mut arrays = Vec::new();
        let mut fields = Vec::new();

        for col_name in selected_columns {
            let col_info = chunk_info.columns.iter()
                .find(|c| &c.name == col_name)
                .ok_or_else(|| PyValueError::new_err(format!("Column '{}' not found in chunk", col_name)))?;

            handle_io_error(file.seek(SeekFrom::Start(col_info.offset)))?;

            // read page magic
            let mut magic = [0u8; 8];
            handle_io_error(file.read_exact(&mut magic))?;
            if magic != PAGE_MAGIC {
                return Err(PyValueError::new_err(format!(
                    "Invalid page magic: expected {:?}, found {:?}",
                    PAGE_MAGIC, magic
                )));
            }

            // read page metadata
            let page_size = file.read_u32::<LittleEndian>()?;
            let num_rows = file.read_u32::<LittleEndian>()?;
            let compressed = file.read_u8()? == 1;

            // read data
            let mut data = vec![0u8; page_size as usize];
            handle_io_error(file.read_exact(&mut data))?;

            // decompress if needed
            let final_data = if compressed {
                decompress_data(&data)?
            } else {
                data
            };

            // get field info
            let field = schema.field_with_name(col_name)
                .map_err(convert_arrow_error)?
                .clone();

            // choose different deserialization method according to data type
            let array = if let DataType::List(field_ref) = field.data_type() {
                // List type (vector data) directly build array from bytes
                let values = bytemuck::cast_slice::<u8, f64>(&final_data);
                let vec_dim = values.len() / num_rows as usize;
                
                let value_data = ArrayData::builder(field_ref.data_type().clone())
                    .len(values.len())
                    .add_buffer(Buffer::from_slice_ref(values))
                    .build()
                    .map_err(convert_arrow_error)?;
                
                let mut offsets = Vec::with_capacity(num_rows as usize + 1);
                for i in 0..=num_rows as usize {
                    offsets.push((i * vec_dim) as i32);
                }
                
                let list_data = ArrayData::builder(DataType::List(field_ref.clone()))
                    .len(num_rows as usize)
                    .add_buffer(Buffer::from_slice_ref(&offsets))
                    .add_child_data(value_data)
                    .build()
                    .map_err(convert_arrow_error)?;
                
                Arc::new(ListArray::from(list_data)) as Arc<dyn Array>
            } else {
                // other types use Arrow IPC deserialization
                let cursor = std::io::Cursor::new(final_data);
                let mut arrow_reader = arrow::ipc::reader::FileReader::try_new(
                    cursor,
                    None,
                ).map_err(convert_arrow_error)?;
                
                let batch = arrow_reader.next()
                    .ok_or_else(|| PyValueError::new_err("No data in column"))?
                    .map_err(convert_arrow_error)?;
                
                batch.column(0).clone()
            };

            arrays.push(array);
            fields.push(field);
        }

        RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            arrays,
        ).map_err(convert_arrow_error)
    }

    pub fn read_vec(&self, name: String, py: Python) -> PyResult<PyObject> {
        let mut file = File::open(&self.filepath)?;
        let metadata = self.read_metadata(&mut file)?;

        let vec_region = metadata.vec_region
            .ok_or_else(|| PyValueError::new_err("No vector data in file"))?;

        let vector_info = vec_region.vectors.iter()
            .find(|v| v.name == name)
            .ok_or_else(|| PyValueError::new_err(format!("Vector '{}' not found", name)))?;

        // read vector data
        file.seek(std::io::SeekFrom::Start(vector_info.offset))?;
        let mut data = vec![0u8; vector_info.size as usize];
        file.read_exact(&mut data)?;

        // import numpy
        let np = py.import("numpy")?;
        
        // build numpy array
        let array = np.call_method1(
            "frombuffer",
            (data.as_slice(), vector_info.dtype.as_str()),
        )?;

        // reshape array
        Ok(array.call_method1("reshape", (vector_info.shape.clone(),))?.into_py(py))
    }

    pub fn read_table_data(&self, selected_columns: &[String], py: Python) -> PyResult<PyObject> {
        // check if file exists
        if !Path::new(&self.filepath).exists() {
            return Err(PyValueError::new_err("File does not exist"));
        }

        // read file header metadata
        let mut file = File::open(&self.filepath)?;
        let _metadata = self.read_metadata(&mut file)?;

        // get schema
        let schema = self.schema.read().unwrap();
        let schema = schema.as_ref()
            .ok_or_else(|| PyValueError::new_err("Schema not initialized"))?;

        // if specified columns, use all columns
        let columns_to_read: Vec<String> = if selected_columns.is_empty() {
            schema.fields()
                .iter()
                .map(|f| f.name().clone())
                .collect()
        } else {
            // check if all requested columns exist
            for col in selected_columns {
                if schema.field_with_name(col).is_err() {
                    return Err(PyValueError::new_err(format!("Column '{}' not found in schema", col)));
                }
            }
            selected_columns.to_vec()
        };

        // read all chunks
        let chunks = self.chunks.read().unwrap();
        let mut record_batches = Vec::new();

        for chunk_info in chunks.iter().map(|chunk| chunk) {
            let batch = self.read_chunk(
                &self.filepath,
                schema,
                &columns_to_read,
                chunk_info,
            )?;
            record_batches.push(batch);
        }

        let pa = py.import("pyarrow")?;
        
        // if no data, return empty table
        if record_batches.is_empty() {
            let empty_schema = Schema::new(
                columns_to_read.iter()
                    .filter_map(|name| schema.field_with_name(name).ok())
                    .map(|f| f.clone())
                    .collect::<Vec<Field>>()
            );
            let empty_batch = RecordBatch::new_empty(Arc::new(empty_schema));
            record_batches.push(empty_batch);
        }

        // convert record batches to PyArrow Table
        let table = arrow::compute::concat_batches(
            &record_batches[0].schema(),
            &record_batches,
        ).map_err(|e| PyValueError::new_err(format!("Failed to concatenate record batches: {}", e)))?;
        
        // create a new RecordBatch
        let batch = RecordBatch::try_new(
            table.schema(),
            table.columns().to_vec(),
        ).map_err(convert_arrow_error)?;

        // convert to PyArrow Table
        let py_batch = batch.into_pyarrow(py)?;
        Ok(pa.getattr("Table")?.call_method1("from_batches", ([py_batch],))?.into_py(py))
    }

    pub fn read_vec_with_mmap(&self, name: String, load_mmap_vec: bool, py: Python) -> PyResult<PyObject> {
        let mut file = File::open(&self.filepath)?;
        let metadata = self.read_metadata(&mut file)?;

        let vec_region = metadata.vec_region
            .ok_or_else(|| PyValueError::new_err("No vector data in file"))?;

        let vector_info = vec_region.vectors.iter()
            .find(|v| v.name == name)
            .ok_or_else(|| PyValueError::new_err(format!("Vector '{}' not found", name)))?;

        if load_mmap_vec {
            // import numpy
            let np = py.import("numpy")?;
            
            // calculate total number of elements
            let total_elements: usize = vector_info.shape.iter().product();
            
            // use numpy.memmap to create memory-mapped array
            let array = np.call_method(
                "memmap",
                (
                    &self.filepath,
                    vector_info.dtype.as_str(),
                    "r",  // read-only mode
                    vector_info.offset,  // file offset
                    total_elements,  // total number of elements
                ),
                None
            )?;
            
            // reshape array
            Ok(array.call_method1("reshape", (vector_info.shape.clone(),))?.into_py(py))
        } else {
            // read vector data in regular way
            self.read_vec(name, py)
        }
    }

    pub fn read_vec_native(&self, name: String) -> Result<VectorData, VectorError> {
        let mut file = File::open(&self.filepath)?;
        let metadata = self.read_metadata(&mut file)
            .map_err(|e| VectorError::IoError(io::Error::new(io::ErrorKind::Other, e.to_string())))?;

        let vec_region = metadata.vec_region
            .ok_or(VectorError::NoVectorRegion)?;

        let vector_info = vec_region.vectors.iter()
            .find(|v| v.name == name)
            .ok_or_else(|| VectorError::VectorNotFound(name.clone()))?;

        // create memory mapping
        let mmap = unsafe {
            MmapOptions::new()
                .offset(vector_info.offset)
                .len(vector_info.size as usize)
                .map_copy(&file)?
        };

        match vector_info.dtype.as_str() {
            // float type
            "<f4" | "float32" => handle_numeric_type::<f32>(&mmap, vector_info),
            "<f8" | "float64" => handle_numeric_type::<f64>(&mmap, vector_info),
            
            // integer type
            "<i4" | "int32" => handle_numeric_type::<i32>(&mmap, vector_info),
            "<i8" | "int64" => handle_numeric_type::<i64>(&mmap, vector_info),
            "<u4" | "uint32" => handle_numeric_type::<u32>(&mmap, vector_info),
            "<u8" | "uint64" => handle_numeric_type::<u64>(&mmap, vector_info),
            "<i2" | "int16" => handle_numeric_type::<i16>(&mmap, vector_info),
            "<u2" | "uint16" => handle_numeric_type::<u16>(&mmap, vector_info),
            "<i1" | "int8" => handle_numeric_type::<i8>(&mmap, vector_info),
            "<u1" | "uint8" => handle_numeric_type::<u8>(&mmap, vector_info),

            // boolean type
            "|b1" | "bool" => {
                let num_elements = vector_info.size as usize;
                let layout = Layout::from_size_align(num_elements, 16).unwrap();
                
                unsafe {
                    let ptr = alloc(layout) as *mut bool;
                    parallel_copy_with_prefetch(
                        mmap.as_ptr(),
                        ptr as *mut u8,
                        vector_info.size as usize
                    );
                    let vec = Vec::from_raw_parts(ptr, num_elements, num_elements);
                    Ok(VectorData::Bool(vec, vector_info.shape.clone()))
                }
            },

            "<f2" | "float16" => {
                let num_elements = vector_info.size as usize / std::mem::size_of::<f16>();
                let layout = Layout::from_size_align(
                    num_elements * std::mem::size_of::<f16>(),
                    16
                ).unwrap();
                
                unsafe {
                    let ptr = alloc(layout) as *mut f16;
                    parallel_copy_with_prefetch(
                        mmap.as_ptr(),
                        ptr as *mut u8,
                        vector_info.size as usize
                    );
                    let vec = Vec::from_raw_parts(ptr, num_elements, num_elements);
                    Ok(VectorData::F16(vec, vector_info.shape.clone()))
                }
            },

            _ => Err(VectorError::UnsupportedDataType(vector_info.dtype.clone())),
        }
    }

    pub fn get_cached_indices(&self) -> PyResult<Vec<RowGroupIndex>> {
        let mut cache = INDEX_CACHE.lock().unwrap();
        
        // check if file exists in cache and if it needs to be updated
        if let Some(cached_index) = cache.get(&self.filepath) {
            let metadata = std::fs::metadata(&self.filepath)?;
            if let Ok(modified_time) = metadata.modified() {
                if modified_time <= cached_index.last_modified {
                    return Ok(cached_index.row_groups.clone());
                }
            }
        }
        
        // if cache does not exist or needs to be updated, read again
        let mut file = File::open(&self.filepath)?;
        let metadata = self.read_metadata(&mut file)?;
        
        if let Some(index_region) = metadata.index_region {
            // update cache
            cache.insert(self.filepath.clone(), IndexCache {
                row_groups: index_region.row_groups.clone(),
                last_modified: SystemTime::now(),
            });
            
            Ok(index_region.row_groups)
        } else {
            Err(PyValueError::new_err("No index region found in file"))
        }
    }

    pub fn read_rows_by_indices(&self, row_indices: &[usize]) -> PyResult<RecordBatch> {
        // read file metadata to initialize schema and chunks
        let mut file = File::open(&self.filepath)?;
        let metadata = self.read_metadata(&mut file)?;
        
        // initialize schema
        {
            let mut schema_guard = self.schema.write().unwrap();
            if schema_guard.is_none() {
                let arrow_schema = Arc::new(metadata.schema.to_arrow_schema());
                *schema_guard = Some(arrow_schema);
            }
        }
        
        // initialize chunks
        {
            let mut chunks_guard = self.chunks.write().unwrap();
            if chunks_guard.is_empty() {
                *chunks_guard = metadata.chunks.clone();
            }
        }
        
        // verify chunks are not empty
        let chunks = self.chunks.read().unwrap();
        if chunks.is_empty() {
            return Err(PyValueError::new_err("No data chunks found in file"));
        }
        
        // verify row indices are valid
        let total_rows: usize = chunks.iter().map(|chunk| chunk.rows).sum();
        if let Some(&max_index) = row_indices.iter().max() {
            if max_index >= total_rows {
                return Err(PyValueError::new_err(format!(
                    "Row index {} out of bounds. Total rows: {}", 
                    max_index, 
                    total_rows
                )));
            }
        }
        
        // get index information
        let indices = self.get_cached_indices()?;
        
        // sort and deduplicate row indices
        let mut sorted_indices: Vec<usize> = row_indices.to_vec();
        sorted_indices.sort_unstable();
        sorted_indices.dedup();
        
        // map row indices to corresponding data blocks
        let mut chunk_rows: HashMap<usize, Vec<usize>> = HashMap::new();
        
        for &row_idx in &sorted_indices {
            let chunk = indices.iter()
                .find(|idx| row_idx >= idx.start_row && row_idx < idx.end_row)
                .ok_or_else(|| PyValueError::new_err(format!("Row {} is out of range", row_idx)))?;
            
            let chunk_row = row_idx - chunk.start_row;
            chunk_rows.entry(chunk.chunk_id)
                .or_insert_with(Vec::new)
                .push(chunk_row);
        }
        
        // read schema
        let schema = self.schema.read().unwrap();
        let schema = schema.as_ref()
            .ok_or_else(|| PyValueError::new_err("Schema not initialized"))?;
        
        let mut final_arrays = Vec::new();
        
        // process each field
        for field in schema.fields() {
            let mut array_builder: Box<dyn ArrayBuilder> = match field.data_type() {
                DataType::Int8 => Box::new(Int8Builder::new()),
                DataType::Int16 => Box::new(Int16Builder::new()),
                DataType::Int32 => Box::new(Int32Builder::new()),
                DataType::Int64 => Box::new(Int64Builder::new()),
                DataType::UInt8 => Box::new(UInt8Builder::new()),
                DataType::UInt16 => Box::new(UInt16Builder::new()),
                DataType::UInt32 => Box::new(UInt32Builder::new()),
                DataType::UInt64 => Box::new(UInt64Builder::new()),
                DataType::Float16 => Box::new(Float16Builder::new()),
                DataType::Float32 => Box::new(Float32Builder::new()),
                DataType::Float64 => Box::new(Float64Builder::new()),
                DataType::Boolean => Box::new(BooleanBuilder::new()),
                DataType::Utf8 => Box::new(StringBuilder::new()),
                DataType::Binary => Box::new(BinaryBuilder::new()),
                DataType::Date32 => Box::new(Date32Builder::new()),
                DataType::Date64 => Box::new(Date64Builder::new()),
                DataType::List(field_ref) => {
                    match field_ref.data_type() {
                        DataType::Int8 => Box::new(ListBuilder::new(Int8Builder::new())),
                        DataType::Int16 => Box::new(ListBuilder::new(Int16Builder::new())),
                        DataType::Int32 => Box::new(ListBuilder::new(Int32Builder::new())),
                        DataType::Int64 => Box::new(ListBuilder::new(Int64Builder::new())),
                        DataType::UInt8 => Box::new(ListBuilder::new(UInt8Builder::new())),
                        DataType::UInt16 => Box::new(ListBuilder::new(UInt16Builder::new())),
                        DataType::UInt32 => Box::new(ListBuilder::new(UInt32Builder::new())),
                        DataType::UInt64 => Box::new(ListBuilder::new(UInt64Builder::new())),
                        DataType::Float16 => Box::new(ListBuilder::new(Float16Builder::new())),
                        DataType::Float32 => Box::new(ListBuilder::new(Float32Builder::new())),
                        DataType::Float64 => Box::new(ListBuilder::new(Float64Builder::new())),
                        _ => return Err(PyValueError::new_err(
                            format!("Unsupported list element type: {:?}", field_ref.data_type())
                        )),
                    }
                },
                _ => return Err(PyValueError::new_err(
                    format!("Unsupported data type: {:?}", field.data_type())
                )),
            };
            
            // read data from each data block
            for (chunk_id, rows) in &chunk_rows {
                let chunk_info = &self.chunks.read().unwrap()[*chunk_id];
                let batch = self.read_chunk(
                    &self.filepath,
                    schema,
                    &[field.name().clone()],
                    chunk_info,
                )?;
                
                let column = batch.column(0);
                
                for &chunk_row in rows {
                    match field.data_type() {
                        DataType::Int8 => append_value!(column, chunk_row, Int8Array, Int8Builder, array_builder),
                        DataType::Int16 => append_value!(column, chunk_row, Int16Array, Int16Builder, array_builder),
                        DataType::Int32 => append_value!(column, chunk_row, Int32Array, Int32Builder, array_builder),
                        DataType::Int64 => append_value!(column, chunk_row, Int64Array, Int64Builder, array_builder),
                        DataType::UInt8 => append_value!(column, chunk_row, UInt8Array, UInt8Builder, array_builder),
                        DataType::UInt16 => append_value!(column, chunk_row, UInt16Array, UInt16Builder, array_builder),
                        DataType::UInt32 => append_value!(column, chunk_row, UInt32Array, UInt32Builder, array_builder),
                        DataType::UInt64 => append_value!(column, chunk_row, UInt64Array, UInt64Builder, array_builder),
                        DataType::Float16 => append_value!(column, chunk_row, Float16Array, Float16Builder, array_builder),
                        DataType::Float32 => append_value!(column, chunk_row, Float32Array, Float32Builder, array_builder),
                        DataType::Float64 => append_value!(column, chunk_row, Float64Array, Float64Builder, array_builder),
                        DataType::Boolean => append_value!(column, chunk_row, BooleanArray, BooleanBuilder, array_builder),
                        DataType::Utf8 => append_value!(column, chunk_row, StringArray, StringBuilder, array_builder),
                        DataType::Binary => append_value!(column, chunk_row, BinaryArray, BinaryBuilder, array_builder),
                        DataType::Date32 => append_value!(column, chunk_row, Date32Array, Date32Builder, array_builder),
                        DataType::Date64 => append_value!(column, chunk_row, Date64Array, Date64Builder, array_builder),
                        DataType::List(_) => {
                            if let Some(list_array) = column.as_any().downcast_ref::<ListArray>() {
                                if list_array.is_null(chunk_row) {
                                    let list_builder = array_builder.as_any_mut()
                                        .downcast_mut::<ListBuilder<Float32Builder>>()
                                        .unwrap();
                                    list_builder.append(false);
                                } else {
                                    let list_builder = array_builder.as_any_mut()
                                        .downcast_mut::<ListBuilder<Float32Builder>>()
                                        .unwrap();
                                    
                                    let values = list_array.value(chunk_row);
                                    list_builder.append(true);
                                    
                                    for i in 0..values.len() {
                                        match values.data_type() {
                                            DataType::Float32 => {
                                                let value = values.as_any()
                                                    .downcast_ref::<Float32Array>()
                                                    .unwrap()
                                                    .value(i);
                                                list_builder.values().append_value(value);
                                            },
                                            DataType::Float64 => {
                                                let value = values.as_any()
                                                    .downcast_ref::<Float64Array>()
                                                    .unwrap()
                                                    .value(i) as f32;
                                                list_builder.values().append_value(value);
                                            },
                                            _ => unreachable!(),
                                        }
                                    }
                                }
                            }
                        },
                        _ => unreachable!(),
                    }
                }
            }
            
            final_arrays.push(array_builder.finish());
        }
        
        // create final RecordBatch 
        RecordBatch::try_new(
            Arc::new(Schema::new(schema.fields().to_vec())),
            final_arrays,
        ).map_err(convert_arrow_error)
    }
}

#[inline]
fn calculate_chunk_size(total_size: usize) -> usize {
    if total_size <= L2_CACHE_SIZE {
        BASE_CHUNK_SIZE
    } else if total_size <= L3_CACHE_SIZE {
        BASE_CHUNK_SIZE * 2
    } else {
        BASE_CHUNK_SIZE * 4
    }
}

#[inline]
fn calculate_optimal_threads(size: usize, chunk_size: usize) -> usize {
    let cpu_threads = rayon::current_num_threads();
    let data_threads = (size / chunk_size).max(1);
    cpu_threads.min(data_threads)
}

#[inline(always)]
unsafe fn copy_chunk_optimized(src: *const u8, dst: *mut u8, size: usize) {
    let src_align = src as usize & (L1_CACHE_LINE - 1);
    let dst_align = dst as usize & (L1_CACHE_LINE - 1);
    
    if src_align == dst_align {
        let pre_align = L1_CACHE_LINE - src_align;
        if pre_align < size {
            std::ptr::copy_nonoverlapping(src, dst, pre_align);
            
            let aligned_src = src.add(pre_align);
            let aligned_dst = dst.add(pre_align);
            let aligned_size = (size - pre_align) & !(L1_CACHE_LINE - 1);
            
            copy_aligned_blocks(aligned_src, aligned_dst, aligned_size);
            
            let remainder_src = aligned_src.add(aligned_size);
            let remainder_dst = aligned_dst.add(aligned_size);
            let remainder_size = size - pre_align - aligned_size;
            if remainder_size > 0 {
                std::ptr::copy_nonoverlapping(remainder_src, remainder_dst, remainder_size);
            }
        } else {
            std::ptr::copy_nonoverlapping(src, dst, size);
        }
    } else {
        std::ptr::copy_nonoverlapping(src, dst, size);
    }
}

#[inline(always)]
unsafe fn copy_aligned_blocks(src: *const u8, dst: *mut u8, size: usize) {
    let mut offset = 0;
    while offset < size {
        std::ptr::copy_nonoverlapping(
            src.add(offset),
            dst.add(offset),
            L1_CACHE_LINE
        );
        offset += L1_CACHE_LINE;
    }
}

#[inline(always)]
fn parallel_copy_with_prefetch(src: *const u8, dst: *mut u8, size: usize) {
    if size < MIN_PARALLEL_SIZE {
        unsafe {
            copy_chunk_optimized(src, dst, size);
        }
        return;
    }

    let chunk_size = calculate_chunk_size(size);
    let num_threads = calculate_optimal_threads(size, chunk_size);
    let chunk_per_thread = (size + num_threads - 1) / num_threads;
    
    let chunk_per_thread = (chunk_per_thread + L1_CACHE_LINE - 1) & !(L1_CACHE_LINE - 1);
    
    let src_slice = unsafe { std::slice::from_raw_parts(src, size) };
    let dst_slice = unsafe { std::slice::from_raw_parts_mut(dst, size) };
    
    // use atomic counter to track progress
    let progress = AtomicUsize::new(0);

    dst_slice.par_chunks_mut(chunk_per_thread)
        .zip(src_slice.par_chunks(chunk_per_thread))
        .for_each(|(dst_chunk, src_chunk)| {
            let chunk_size = src_chunk.len();
            let mut offset = 0;
            
            // prefetch first batch of data
            for i in 0..PREFETCH_DISTANCE.min(chunk_size / chunk_size) {
                black_box(unsafe {
                    std::ptr::read_volatile(
                        src_chunk[(i * chunk_size)..].as_ptr()
                    )
                });
            }
            
            while offset + chunk_size <= chunk_size {
                // use optimized memory copy
                unsafe {
                    copy_chunk_optimized(
                        src_chunk[offset..].as_ptr(),
                        dst_chunk[offset..].as_mut_ptr(),
                        chunk_size
                    );
                }
                
                // prefetch next data block
                if offset + (PREFETCH_DISTANCE + 1) * chunk_size <= chunk_size {
                    black_box(unsafe {
                        std::ptr::read_volatile(
                            src_chunk[offset + PREFETCH_DISTANCE * chunk_size..].as_ptr()
                        )
                    });
                }
                
                offset += chunk_size;
                progress.fetch_add(1, Ordering::Relaxed);
            }
            
            // handle remaining bytes
            if offset < chunk_size {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_chunk[offset..].as_ptr(),
                        dst_chunk[offset..].as_mut_ptr(),
                        chunk_size - offset
                    );
                }
            }
        });
}

#[inline]
fn handle_numeric_type<T: Copy + Send + Sync + 'static>(
    mmap: &memmap2::MmapMut,
    vector_info: &VectorInfo,
) -> Result<VectorData, VectorError> {
    let num_elements = vector_info.size as usize / std::mem::size_of::<T>();
    let layout = Layout::from_size_align(
        num_elements * std::mem::size_of::<T>(),
        16
    ).unwrap();
    
    unsafe {
        let ptr = alloc(layout) as *mut T;
        parallel_copy_with_prefetch(
            mmap.as_ptr(),
            ptr as *mut u8,
            vector_info.size as usize
        );
        let vec = Vec::from_raw_parts(ptr, num_elements, num_elements);
        
        match std::any::TypeId::of::<T>() {
            t if t == std::any::TypeId::of::<f32>() => 
                Ok(VectorData::F32(std::mem::transmute(vec), vector_info.shape.clone())),
            t if t == std::any::TypeId::of::<f64>() => 
                Ok(VectorData::F64(std::mem::transmute(vec), vector_info.shape.clone())),
            t if t == std::any::TypeId::of::<i32>() => 
                Ok(VectorData::I32(std::mem::transmute(vec), vector_info.shape.clone())),
            t if t == std::any::TypeId::of::<i64>() => 
                Ok(VectorData::I64(std::mem::transmute(vec), vector_info.shape.clone())),
            t if t == std::any::TypeId::of::<u32>() => 
                Ok(VectorData::U32(std::mem::transmute(vec), vector_info.shape.clone())),
            t if t == std::any::TypeId::of::<u64>() => 
                Ok(VectorData::U64(std::mem::transmute(vec), vector_info.shape.clone())),
            t if t == std::any::TypeId::of::<i16>() => 
                Ok(VectorData::I16(std::mem::transmute(vec), vector_info.shape.clone())),
            t if t == std::any::TypeId::of::<u16>() => 
                Ok(VectorData::U16(std::mem::transmute(vec), vector_info.shape.clone())),
            t if t == std::any::TypeId::of::<i8>() => 
                Ok(VectorData::I8(std::mem::transmute(vec), vector_info.shape.clone())),
            t if t == std::any::TypeId::of::<u8>() => 
                Ok(VectorData::U8(std::mem::transmute(vec), vector_info.shape.clone())),
            t if t == std::any::TypeId::of::<f16>() => 
                Ok(VectorData::F16(std::mem::transmute(vec), vector_info.shape.clone())),
            _ => Err(VectorError::UnsupportedDataType(vector_info.dtype.clone())),
        }
    }
}

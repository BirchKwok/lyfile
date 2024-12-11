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

use arrow::array::{Array, ListArray, StringBuilder, StringArray, Int64Array, Float64Array, Float32Array, ArrayBuilder, Float32Builder, Int64Builder, Float64Builder};
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

lazy_static! {
    static ref INDEX_CACHE: Mutex<HashMap<String, IndexCache>> = Mutex::new(HashMap::new());
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
            "<f4" | "float32" => {
                let num_elements = vector_info.size as usize / std::mem::size_of::<f32>();
                
                // Create 16-byte aligned memory allocation
                let layout = Layout::from_size_align(
                    num_elements * std::mem::size_of::<f32>(),
                    16
                ).unwrap();
                
                unsafe {
                    let ptr = alloc(layout) as *mut f32;
                    
                    parallel_copy_with_prefetch(
                        mmap.as_ptr(),
                        ptr as *mut u8,
                        vector_info.size as usize
                    );
                    
                    let vec = Vec::from_raw_parts(ptr, num_elements, num_elements);
                    Ok(VectorData::F32(vec, vector_info.shape.clone()))
                }
            },
            "<f8" | "float64" => {
                let num_elements = vector_info.size as usize / std::mem::size_of::<f64>();
                
                let layout = Layout::from_size_align(
                    num_elements * std::mem::size_of::<f64>(),
                    16
                ).unwrap();
                
                unsafe {
                    let ptr = alloc(layout) as *mut f64;
                    
                    parallel_copy_with_prefetch(
                        mmap.as_ptr(),
                        ptr as *mut u8,
                        vector_info.size as usize
                    );
                    
                    let vec = Vec::from_raw_parts(ptr, num_elements, num_elements);
                    Ok(VectorData::F64(vec, vector_info.shape.clone()))
                }
            },
            _ => Err(VectorError::UnsupportedDataType(vector_info.dtype.clone())),
        }
    }

    pub fn get_cached_indices(&self) -> PyResult<Vec<RowGroupIndex>> {
        let mut cache = INDEX_CACHE.lock().unwrap();
        
        // 检查文件是否存在于缓存中，且是否需要更新
        if let Some(cached_index) = cache.get(&self.filepath) {
            let metadata = std::fs::metadata(&self.filepath)?;
            if let Ok(modified_time) = metadata.modified() {
                if modified_time <= cached_index.last_modified {
                    return Ok(cached_index.row_groups.clone());
                }
            }
        }
        
        // 如果缓存不存在或需要更新，则重新读取
        let mut file = File::open(&self.filepath)?;
        let metadata = self.read_metadata(&mut file)?;
        
        if let Some(index_region) = metadata.index_region {
            // 更新缓存
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
        let indices = self.get_cached_indices()?;
        let mut chunk_rows: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        
        // 将行号映射到对应的数据块和块内偏移
        for &row_idx in row_indices {
            let chunk = indices.iter()
                .find(|idx| row_idx >= idx.start_row && row_idx < idx.end_row)
                .ok_or_else(|| PyValueError::new_err(format!("Row {} is out of range", row_idx)))?;
            
            let chunk_row = row_idx - chunk.start_row;
            chunk_rows.entry(chunk.chunk_id)
                .or_insert_with(Vec::new)
                .push((row_idx, chunk_row));
        }

        // 读取并处理每个需要的数据块
        let schema = self.schema.read().unwrap();
        let schema = schema.as_ref()
            .ok_or_else(|| PyValueError::new_err("Schema not initialized"))?;
        
        let mut final_arrays = Vec::new();
        
        for field in schema.fields() {
            // 根据字段类型创建相应的数组构建器
            let mut array_builder: Box<dyn ArrayBuilder> = match field.data_type() {
                DataType::Float32 => Box::new(Float32Builder::new()),
                DataType::Float64 => Box::new(Float64Builder::new()),
                DataType::Int64 => Box::new(Int64Builder::new()),
                DataType::Utf8 => Box::new(StringBuilder::new()),
                _ => return Err(PyValueError::new_err(
                    format!("Unsupported data type: {:?}", field.data_type())
                )),
            };
            
            for (chunk_id, rows) in &chunk_rows {
                let chunk_info = &self.chunks.read().unwrap()[*chunk_id];
                let batch = self.read_chunk(
                    &self.filepath,
                    schema,
                    &[field.name().clone()],
                    chunk_info,
                )?;
                
                let column = batch.column(0);
                for &(_, chunk_row) in rows {
                    match field.data_type() {
                        DataType::Float32 => {
                            if let Some(array) = column.as_any().downcast_ref::<Float32Array>() {
                                if array.is_null(chunk_row) {
                                    array_builder.as_any_mut().downcast_mut::<Float32Builder>()
                                        .unwrap()
                                        .append_null();
                                } else {
                                    array_builder.as_any_mut().downcast_mut::<Float32Builder>()
                                        .unwrap()
                                        .append_value(array.value(chunk_row));
                                }
                            }
                        },
                        DataType::Float64 => {
                            if let Some(array) = column.as_any().downcast_ref::<Float64Array>() {
                                if array.is_null(chunk_row) {
                                    array_builder.as_any_mut().downcast_mut::<Float64Builder>()
                                        .unwrap()
                                        .append_null();
                                } else {
                                    array_builder.as_any_mut().downcast_mut::<Float64Builder>()
                                        .unwrap()
                                        .append_value(array.value(chunk_row));
                                }
                            }
                        },
                        DataType::Int64 => {
                            if let Some(array) = column.as_any().downcast_ref::<Int64Array>() {
                                if array.is_null(chunk_row) {
                                    array_builder.as_any_mut().downcast_mut::<Int64Builder>()
                                        .unwrap()
                                        .append_null();
                                } else {
                                    array_builder.as_any_mut().downcast_mut::<Int64Builder>()
                                        .unwrap()
                                        .append_value(array.value(chunk_row));
                                }
                            }
                        },
                        DataType::Utf8 => {
                            if let Some(array) = column.as_any().downcast_ref::<StringArray>() {
                                if array.is_null(chunk_row) {
                                    array_builder.as_any_mut().downcast_mut::<StringBuilder>()
                                        .unwrap()
                                        .append_null();
                                } else {
                                    array_builder.as_any_mut().downcast_mut::<StringBuilder>()
                                        .unwrap()
                                        .append_value(array.value(chunk_row));
                                }
                            }
                        },
                        _ => unreachable!(),
                    }
                }
            }
            
            final_arrays.push(array_builder.finish());
        }

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

use std::fs::{File, OpenOptions};
use std::io::{Write, Seek, Read, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use arrow::datatypes::{Schema, Field, DataType};
use arrow::pyarrow::PyArrowType;
use arrow::record_batch::RecordBatch;
use arrow::ipc::writer;

use memmap2::MmapOptions;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyDict;

use serde_json;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use arrow::array::{Array, ListArray, Float64Array};

use crate::structs::*;


impl _LyFile {
    pub fn write_table_data(&mut self, data: &PyAny, py: Python) -> PyResult<()> {
        if data.is_none() {
            return Err(PyValueError::new_err("Input data is None"));
        }
        use std::fs;

        if Path::new(&self.filepath).exists() {
            fs::remove_file(&self.filepath)?;
        }

        // get record batches with size limit
        let record_batches = if data.getattr("__class__")?.getattr("__name__")?.extract::<String>()? == "Table" {
            // specify max rows when using to_batches
            let batches = data.call_method1("to_batches", (PyDict::new(py).set_item("max_chunksize", CHUNK_MAX_ROWS)?,))?;
            batches.extract::<Vec<PyArrowType<RecordBatch>>>()?
        } else {
            // convert to PyArrow Table
            let table = if data.is_instance_of::<pyo3::types::PyDict>() {
                let pa = PyModule::import(py, "pyarrow")?;
                pa.getattr("Table")?.call_method1("from_pydict", (data,))?
            } else {
                let pa = PyModule::import(py, "pyarrow")?;
                pa.getattr("Table")?.call_method1("from_pandas", (data,))?
            };
            
            // get batches with specified chunk size
            let batches = table.call_method1("to_batches", (PyDict::new(py).set_item("max_chunksize", CHUNK_MAX_ROWS)?,))?;
            batches.extract::<Vec<PyArrowType<RecordBatch>>>()?
        };

        // handle empty record batches
        if record_batches.is_empty() {
            return Err(PyValueError::new_err("Empty table"));
        }

        // get schema
        let schema = record_batches[0].0.schema();
        let schema_bytes = serde_json::to_vec(&SerializableSchema::from(schema.as_ref()))
            .map_err(|e| PyValueError::new_err(format!("Failed to serialize schema: {}", e)))?;

        // create new file
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.filepath)?;

        // write file header
        let schema_length = schema_bytes.len() as u32;
        let data_offset = FILE_HEADER_SIZE as u64 + schema_length as u64;
        let index_offset = 0; // not used for now
        self.write_file_header(
            &mut file,
            data_offset,
            index_offset,
            schema_length,
            FEATURE_FLAGS,
            record_batches.len() as u32,
        )?;

        // write schema
        file.write_all(&schema_bytes)?;

        // write data
        let mut current_offset = data_offset;
        let mut chunks = Vec::new();

        // write each batch to file
        for batch in record_batches {
            let chunk_info = self.write_chunk(&mut file, &batch.0, &mut current_offset)?;
            chunks.push((chunk_info.offset, chunk_info));
        }

        // write index region
        let index_region = self.write_index_region(&mut file, &chunks)?;

        // update metadata
        let metadata = Metadata {
            schema: SerializableSchema::from(schema.as_ref()),
            chunks: chunks.into_iter().map(|(_, info)| info).collect(),
            vec_region: Some(VecRegionInfo {
                offset: current_offset,
                size: 0,
                vectors: Vec::new(),
            }),
            index_region: Some(index_region),
        };

        self.write_footer(&mut file, &metadata)?;

        Ok(())
    }

    pub fn append_table_data(&mut self, data: &PyAny, py: Python) -> PyResult<()> {
        if !Path::new(&self.filepath).exists() {
            return self.write_table_data(data, py);
        }

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

        let (metadata_clone, _new_chunk_info) = py.allow_threads(|| -> PyResult<(Metadata, ChunkInfo)> {
            let mut file = handle_io_error(
                OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&self.filepath)
            )?;

            // read metadata
            let metadata = self.read_metadata(&mut file)?;

            // validate new data schema
            let file_schema = metadata.schema.to_arrow_schema();
            if record_batch.schema().fields() != file_schema.fields() {
                return Err(PyValueError::new_err("Schema of new data does not match the file schema"));
            }

            // locate footer start, which is the offset before footer
            let footer_offset = handle_io_error(file.seek(std::io::SeekFrom::End(-(8 + 4))))?;
            let metadata_length = file.read_u32::<LittleEndian>()?;
            let metadata_start = footer_offset - metadata_length as u64;

            // locate data start, prepare to append new chunk
            let new_chunk_offset = metadata_start;

            // move file pointer to metadata start, prepare to write new chunk
            handle_io_error(file.seek(std::io::SeekFrom::Start(new_chunk_offset)))?;

            let mut current_offset = new_chunk_offset;

            // write new chunk
            let chunk_info = self.write_chunk(&mut file, &record_batch, &mut current_offset)?;

            // update metadata
            let mut new_metadata = metadata.clone();
            new_metadata.chunks.push(chunk_info.clone());

            self.write_footer(&mut file, &new_metadata)?;

            // update chunk count in file header
            let num_chunks = new_metadata.chunks.len() as u32;
            handle_io_error(file.seek(std::io::SeekFrom::Start(0)))?;
            self.update_chunk_count_in_header(&mut file, num_chunks)?;

            // flush file
            handle_io_error(file.flush())?;

            Ok((new_metadata, chunk_info))
        })?;

        // update self.chunks
        {
            let mut chunks_lock = self.chunks.write().unwrap();
            *chunks_lock = metadata_clone.chunks;
        }

        Ok(())
    }

    pub fn write_vec(&mut self, name: String, data: &PyAny, py: Python) -> PyResult<()> {
        // validate input is numpy array
        if !data.hasattr("__array_interface__")? {
            return Err(PyValueError::new_err("Input must be a numpy array"));
        }

        // get array interface and shape
        let array_interface = data.getattr("__array_interface__")?;
        let shape: Vec<usize> = array_interface.get_item("shape")?.extract()?;
        let typestr: String = array_interface.get_item("typestr")?.extract()?;
        
        // get data pointer and length
        let data_ptr = array_interface.get_item("data")?
            .extract::<(usize, bool)>()?;
        let ptr = data_ptr.0 as *const u8;
        let total_bytes = shape.iter().product::<usize>() * match typestr.as_str() {
            "<f4" | "float32" => 4,
            "<f8" | "float64" => 8,
            _ => return Err(PyValueError::new_err(format!("Unsupported dtype: {}", typestr))),
        };

        // validate dimension
        if shape.len() < 2 || shape.len() > 3 {
            return Err(PyValueError::new_err("Vector must be 2D or 3D"));
        }

        // read existing metadata (if file exists)
        let n_rows: usize = if Path::new(&self.filepath).exists() {
            let mut file = File::open(&self.filepath)?;
            let metadata = self.read_metadata(&mut file)?;
            metadata.chunks.iter().map(|chunk| chunk.rows).sum()
        } else {
            // if file does not exist, get row count from chunks
            self.chunks.read().unwrap().iter().map(|chunk| chunk.rows).sum()
        };

        // validate row count
        if n_rows > 0 && shape[0] != n_rows {
            return Err(PyValueError::new_err(
                format!("First dimension must match number of rows (expected {}, got {})", 
                    n_rows, shape[0])
            ));
        }

        // read data using Python GIL protected way
        let data_vec = unsafe {
            std::slice::from_raw_parts(ptr, total_bytes).to_vec()
        };

        py.allow_threads(|| {
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&self.filepath)?;

            // read existing metadata
            let mut metadata = self.read_metadata(&mut file)?;

            // if it is the first vector, create vector region
            if metadata.vec_region.is_none() {
                file.seek(std::io::SeekFrom::End(-12))?;
                let footer_pos = file.stream_position()?;
                metadata.vec_region = Some(VecRegionInfo {
                    offset: footer_pos,
                    size: 0,
                    vectors: Vec::new(),
                });
            }

            // get vector write position
            let vec_region = metadata.vec_region.as_mut().unwrap();
            let vector_offset = vec_region.offset + vec_region.size;

            // preallocate file space
            file.set_len(vector_offset + total_bytes as u64)?;
            file.seek(std::io::SeekFrom::Start(vector_offset))?;
            
            // ensure writer is correctly released
            {
                let mut writer = std::io::BufWriter::with_capacity(1024 * 1024, &mut file);
                writer.write_all(&data_vec)?;
                writer.flush()?;
            }

            // update vector info
            vec_region.vectors.push(VectorInfo {
                name,
                offset: vector_offset,
                size: total_bytes as u64,
                shape,
                dtype: typestr,
            });
            vec_region.size += total_bytes as u64;

            // rewrite footer
            let new_pos = vector_offset + total_bytes as u64;
            file.seek(std::io::SeekFrom::Start(new_pos))?;
            
            // serialize and write metadata
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

    pub fn append_vec(&mut self, data: &PyAny, py: Python) -> PyResult<()> {
        if !data.is_instance_of::<pyo3::types::PyDict>() {
            return Err(PyValueError::new_err("Input must be a dictionary"));
        }

        let data_dict: &pyo3::types::PyDict = data.downcast()?;
        let vector_names: Vec<String> = data_dict.keys().iter()
            .map(|k| k.extract::<String>())
            .collect::<Result<_, _>>()?;

        // read existing vector info
        let mut file = File::open(&self.filepath)?;
        let mut metadata = self.read_metadata(&mut file)?;

        let vec_region = metadata.vec_region
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("No vector data in file"))?;

        // validate vector names
        if vector_names.len() != vec_region.vectors.len() {
            return Err(PyValueError::new_err("Number of vectors does not match"));
        }

        for vector_info in &vec_region.vectors {
            if !vector_names.contains(&vector_info.name) {
                return Err(PyValueError::new_err(format!("Vector '{}' not found in input data", vector_info.name)));
            }
        }

        // append data
        for vector_info in &mut vec_region.vectors {
            let array = data_dict.get_item(&vector_info.name).unwrap();
            if !array.hasattr("__array_interface__")? {
                return Err(PyValueError::new_err("All values must be numpy arrays"));
            }

            // get array interface and shape info
            let array_interface = array.getattr("__array_interface__")?;
            let shape: Vec<usize> = array_interface.get_item("shape")?.extract()?;
            let typestr: String = array_interface.get_item("typestr")?.extract()?;

            // validate data type
            if typestr != vector_info.dtype {
                return Err(PyValueError::new_err(format!("Data type mismatch for vector '{}'", vector_info.name)));
            }

            // use numpy's tofile method to write to temp file
            let temp_path = format!("{}.temp.vec", self.filepath);
            array.call_method1("tofile", (temp_path.as_str(),))?;

            // read temp file content
            py.allow_threads(|| -> PyResult<()> {
                let mut temp_file = File::open(&temp_path)?;
                let temp_metadata = temp_file.metadata()?;
                let data_len = temp_metadata.len() as usize;
                let mut data_vec = vec![0u8; data_len];
                temp_file.read_exact(&mut data_vec)?;

                // delete temp file
                std::fs::remove_file(&temp_path)?;

                // open target file
                let file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&self.filepath)?;

                // get vector write position
                let vector_offset = vector_info.offset + vector_info.size;

                // preallocate file space
                file.set_len(vector_offset + data_len as u64)?;

                // use memory mapping to write data quickly
                let mut mmap = unsafe {
                    MmapOptions::new()
                        .offset(vector_offset)
                        .len(data_len)
                        .map_mut(&file)?
                };

                // write all data at once
                mmap.copy_from_slice(&data_vec);
                mmap.flush()?;

                // update vector info
                vector_info.size += data_len as u64;
                vector_info.shape[0] += shape[0]; // assume append rows

                Ok(())
            })?;
        }

        // rewrite footer
        let new_pos = vec_region.offset + vec_region.size;
        file.seek(std::io::SeekFrom::Start(new_pos))?;
        
        // serialize and write metadata
        let metadata_bytes = serde_json::to_vec(&metadata)
            .map_err(|e| PyValueError::new_err(format!("Failed to serialize metadata: {}", e)))?;
        let metadata_length = metadata_bytes.len() as u32;
        
        file.write_all(&metadata_bytes)?;
        file.write_u32::<LittleEndian>(metadata_length)?;
        file.write_all(MAGIC_BYTES)?;
        file.flush()?;

        Ok(())
    }

    pub fn write_file_header(
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

        // version (4 bytes)
        buffer.write_u32::<LittleEndian>(VERSION)?;

        // data offset (8 bytes)
        buffer.write_u64::<LittleEndian>(data_offset)?;

        // index offset (8 bytes)
        buffer.write_u64::<LittleEndian>(index_offset)?;

        // schema length (4 bytes)
        buffer.write_u32::<LittleEndian>(schema_length)?;

        // feature flags (4 bytes)
        buffer.write_u32::<LittleEndian>(feature_flags)?;

        // chunk count (4 bytes)
        buffer.write_u32::<LittleEndian>(num_chunks)?;

        // write file header
        handle_io_error(file.write_all(&buffer))?;

        Ok(())
    }

    pub fn write_chunk(
        &self,
        file: &mut File,
        record_batch: &RecordBatch,
        current_offset: &mut u64,
    ) -> PyResult<ChunkInfo> {
        let chunk_start = *current_offset;
        
        // write CHUNK_MAGIC
        handle_io_error(file.write_all(&CHUNK_MAGIC))?;
        *current_offset += CHUNK_MAGIC.len() as u64;

        // reserve chunk size
        let size_position = *current_offset;
        file.write_u32::<LittleEndian>(0)?;
        *current_offset += 4;

        // write row count and column count
        let num_rows = record_batch.num_rows() as u32;
        let num_columns = record_batch.num_columns() as u32;
        file.write_u32::<LittleEndian>(num_rows)?;
        file.write_u32::<LittleEndian>(num_columns)?;
        *current_offset += 8;

        // process each column
        let mut column_infos = Vec::with_capacity(num_columns as usize);
        
        for i in 0..num_columns as usize {
            let column = record_batch.column(i);
            let schema = record_batch.schema();
            let field = schema.field(i).clone();
            let column_name = field.name().to_string();
            let column_offset = *current_offset;

            // write page magic
            handle_io_error(file.write_all(&PAGE_MAGIC))?;
            *current_offset += PAGE_MAGIC.len() as u64;

            // serialize column data
            let buffer = if let DataType::List(_) = field.data_type() {
                // List type not compressed, convert to bytes directly
                let list_array = column.as_any()
                    .downcast_ref::<ListArray>()
                    .ok_or_else(|| PyValueError::new_err("Failed to downcast to ListArray"))?;
                
                let values = list_array.values();
                let float_array = values.as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| PyValueError::new_err("Failed to downcast to Float64Array"))?;
                
                unsafe {
                    std::slice::from_raw_parts(
                        float_array.values().as_ptr() as *const u8,
                        float_array.values().len() * std::mem::size_of::<f64>()
                    ).to_vec()
                }
            } else {
                // other types use Arrow IPC serialization
                let mut buffer = Vec::new();
                let schema = Schema::new(vec![field.clone()]);
                let single_column_batch = RecordBatch::try_new(
                    Arc::new(schema.clone()),
                    vec![column.clone()],
                ).map_err(convert_arrow_error)?;

                {
                    let mut writer = writer::FileWriter::try_new(
                        &mut buffer,
                        &schema,
                    ).map_err(convert_arrow_error)?;

                    writer.write(&single_column_batch).map_err(convert_arrow_error)?;
                    writer.finish().map_err(convert_arrow_error)?;
                }
                buffer
            };

            // determine if compression is needed
            let should_compress = is_compressible(field.data_type()) && buffer.len() > 1024;
            let (final_buffer, is_compressed) = if should_compress {
                (compress_data(&buffer)?, true)
            } else {
                (buffer, false)
            };

            // write page metadata
            file.write_u32::<LittleEndian>(final_buffer.len() as u32)?;  // data size
            file.write_u32::<LittleEndian>(num_rows)?;  // row count
            file.write_u8(is_compressed as u8)?;  // compression flag
            *current_offset += 9;

            // write data
            handle_io_error(file.write_all(&final_buffer))?;
            *current_offset += final_buffer.len() as u64;

            column_infos.push(ColumnInfo {
                name: column_name,
                offset: column_offset,
                size: PAGE_MAGIC.len() as u64 + 9 + final_buffer.len() as u64,
                compressed: is_compressed,
            });
        }

        // update chunk size
        let chunk_end = *current_offset;
        let chunk_size = (chunk_end - size_position - 4) as u32;
        handle_io_error(file.seek(SeekFrom::Start(size_position)))?;
        file.write_u32::<LittleEndian>(chunk_size)?;
        handle_io_error(file.seek(SeekFrom::Start(chunk_end)))?;

        Ok(ChunkInfo {
            offset: chunk_start,
            size: (chunk_end - chunk_start) as u64,
            rows: num_rows as usize,
            columns: column_infos,
        })
    }

    pub fn write_index_region(
        &self,
        file: &mut File,
        chunks: &Vec<(u64, ChunkInfo)>,
    ) -> PyResult<IndexRegion> {
        let index_start = file.seek(SeekFrom::Current(0))?;
        
        // 创建行组索引
        let mut row_groups = Vec::new();
        let mut current_row = 0;
        
        for (chunk_id, (offset, chunk_info)) in chunks.iter().enumerate() {
            let row_group = RowGroupIndex {
                start_row: current_row,
                end_row: current_row + chunk_info.rows,
                chunk_id,
                offset: *offset,
            };
            row_groups.push(row_group);
            current_row += chunk_info.rows;
        }

        // 序列化并写入索引数据
        let index_bytes = serde_json::to_vec(&row_groups)
            .map_err(|e| PyValueError::new_err(format!("Failed to serialize index: {}", e)))?;
        
        file.write_all(&index_bytes)?;
        let index_end = file.seek(SeekFrom::Current(0))?;

        Ok(IndexRegion {
            row_groups,
            offset: index_start,
            size: index_end - index_start,
        })
    }

    // add helper method for reading metadata
    pub fn read_metadata(&self, file: &mut File) -> PyResult<Metadata> {
        // read metadata from footer
        // read MAGIC_BYTES (last 8 bytes)
        handle_io_error(file.seek(std::io::SeekFrom::End(-8)))?;
        let mut footer_magic = [0u8; 8];
        handle_io_error(file.read_exact(&mut footer_magic))?;
        if footer_magic != MAGIC_BYTES {
            return Err(PyValueError::new_err("Invalid footer magic"));
        }

        // read metadata length (from 12th to 9th byte from the end)
        handle_io_error(file.seek(std::io::SeekFrom::End(-12)))?;
        let metadata_length = file.read_u32::<LittleEndian>()?;

        // calculate metadata start offset
        handle_io_error(file.seek(std::io::SeekFrom::End(-(12 + metadata_length as i64))))?;

        // read metadata
        let mut metadata_bytes = vec![0u8; metadata_length as usize];
        handle_io_error(file.read_exact(&mut metadata_bytes))?;

        let metadata: Metadata = handle_serde_error(serde_json::from_slice(&metadata_bytes))?;
        Ok(metadata)
    }

    // add helper method for updating chunk count in header
    pub fn update_chunk_count_in_header(&self, file: &mut File, num_chunks: u32) -> PyResult<()> {
        // the offset of chunk count in the header is:
        // MAGIC (8 bytes) + VERSION (4 bytes) + DATA_OFFSET (8 bytes) + INDEX_OFFSET (8 bytes) + SCHEMA_LENGTH (4 bytes) + FEATURE_FLAGS (4 bytes)
        let chunk_count_offset = 8 + 4 + 8 + 8 + 4 + 4;
        handle_io_error(file.seek(std::io::SeekFrom::Start(chunk_count_offset as u64)))?;
        file.write_u32::<LittleEndian>(num_chunks)?;
        Ok(())
    }

    pub fn write_footer(&self, file: &mut File, metadata: &Metadata) -> PyResult<()> {
        // serialize metadata
        let metadata_bytes = handle_serde_error(serde_json::to_vec(metadata))?;
        let metadata_length = metadata_bytes.len() as u32;

        // write metadata
        handle_io_error(file.write_all(&metadata_bytes))?;

        // write metadata length
        file.write_u32::<LittleEndian>(metadata_length)?;

        // write MAGIC_BYTES
        handle_io_error(file.write_all(MAGIC_BYTES))?;

        Ok(())
    }

    pub fn divide_into_pages(&self, array: &Arc<dyn arrow::array::Array>) -> PyResult<Vec<RecordBatch>> {
        // here simply treat the whole column as a page, can be divided according to needs
        let schema = Schema::new(vec![Field::new("column", array.data_type().clone(), true)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![array.clone()])
            .map_err(|e| PyValueError::new_err(format!("Arrow error: {}", e)))?;
        Ok(vec![batch])
    }

    pub fn serialize_row_group(&self, batch: &RecordBatch) -> PyResult<Vec<u8>> {
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
}


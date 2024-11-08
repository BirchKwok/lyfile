cimport numpy as np
import numpy as np
from libc.stdio cimport (
    FILE, fopen, fclose, fread, fwrite, fseek, SEEK_SET
)
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, strncpy
from libc.stdint cimport uint32_t
import os

cdef struct NnpHeader:
    uint32_t rows
    char dtype[30]
    uint32_t shape


def save_nnp(str filename, np.ndarray data, bint append=True):
    cdef:
        FILE* f
        NnpHeader header
        char* raw_data
        Py_ssize_t data_size
        uint32_t current_rows, new_rows
        const char* mode
        bytes dtype_bytes
        np.ndarray data_2d

    if data.ndim == 1:
        data_2d = data.reshape(1, -1)
    else:
        data_2d = data

    data_size = data_2d.nbytes
    file_exists = os.path.exists(filename)
    if append and not file_exists:
        append = False  # If file does not exist, force non-append mode
    
    mode = 'rb+' if (append and file_exists) else 'wb+'
    
    f = fopen(filename.encode('utf-8'), mode)
    if f == NULL:
        raise IOError(f"Unable to open file {filename}")

    try:
        if append and file_exists:  # Only read header if file exists and append is true
            # Read existing header
            if fread(&header, sizeof(NnpHeader), 1, f) != 1:
                fclose(f)
                raise IOError("Failed to read header")
            current_rows = header.rows
            new_rows = current_rows + data_2d.shape[0]

            # Update row count
            fseek(f, 0, SEEK_SET)
            header.rows = new_rows
            if fwrite(&header, sizeof(NnpHeader), 1, f) != 1:
                fclose(f)
                raise IOError("Failed to write header")

            # Move to the end of the data
            fseek(f, sizeof(NnpHeader) + current_rows * data_2d.shape[1] * data_2d.itemsize, SEEK_SET)
        else:
            # Create new header
            header.rows = data_2d.shape[0]
            dtype_bytes = str(data_2d.dtype).encode('utf-8')
            strncpy(header.dtype, dtype_bytes, 29)
            header.dtype[29] = 0
            header.shape = data_2d.shape[1]

            if fwrite(&header, sizeof(NnpHeader), 1, f) != 1:
                fclose(f)
                raise IOError("Failed to write header")

        # Write data in one go
        raw_data = <char*>data_2d.data
        if fwrite(raw_data, data_size, 1, f) != 1:
            fclose(f)
            raise IOError("Failed to write data")

    finally:
        fclose(f)


def load_nnp(str filename, bint mmap_mode=False):
    cdef:
        FILE* f
        NnpHeader header
        np.ndarray result
        tuple shape
        np.dtype dtype_obj

    if mmap_mode:
        # Use memory mapping
        f = fopen(filename.encode('utf-8'), 'rb')
        if f == NULL:
            raise IOError(f"Unable to open file {filename}")
        try:
            if fread(&header, sizeof(NnpHeader), 1, f) != 1:
                fclose(f)
                raise IOError("Failed to read header")
            shape = (int(header.rows), int(header.shape))
            dtype_obj = np.dtype(header.dtype.decode('utf-8').strip())
            fclose(f)
            return np.memmap(filename, mode='r', dtype=dtype_obj,
                             shape=shape, offset=sizeof(NnpHeader))
        except:
            fclose(f)
            raise

    f = fopen(filename.encode('utf-8'), 'rb')
    if f == NULL:
        raise IOError(f"Unable to open file {filename}")

    try:
        # Read header
        if fread(&header, sizeof(NnpHeader), 1, f) != 1:
            fclose(f)
            raise IOError("Failed to read header")

        # Create numpy array
        shape = (int(header.rows), int(header.shape))
        dtype_obj = np.dtype(header.dtype.decode('utf-8').strip())
        result = np.empty(shape, dtype=dtype_obj)

        # Read data in one go
        if fread(<char*>result.data, result.nbytes, 1, f) != 1:
            fclose(f)
            raise IOError("Failed to read data")

        return result

    finally:
        fclose(f)

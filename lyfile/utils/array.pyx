import cython
cimport numpy as cnp
import numpy as np
from libc.stdlib cimport malloc, free

cnp.import_array()

cdef struct ArrayInfo:
    cnp.float64_t *data_ptr
    cnp.npy_intp shape0
    cnp.npy_intp strides0

cdef class ArrayView:
    cdef:
        ArrayInfo *arrays  # Pointer array for array information
        cnp.npy_intp *start_rows  # Array of start row indices for each block
        list views  # Reference to original arrays to prevent garbage collection
        cython.int num_blocks
        cnp.npy_intp total_rows
        cython.int vector_dim

    def __init__(self, list views_list, total_rows, vector_dim):
        cdef cnp.npy_intp current_row = 0
        cdef cython.int i
        cdef object view

        self.num_blocks = len(views_list)
        self.arrays = <ArrayInfo *>malloc(self.num_blocks * sizeof(ArrayInfo))
        self.start_rows = <cnp.npy_intp *>malloc(self.num_blocks * sizeof(cnp.npy_intp))
        self.views = []
        self.vector_dim = vector_dim

        if self.arrays == NULL or self.start_rows == NULL:
            raise MemoryError("Memory allocation failed")

        for i in range(self.num_blocks):
            view = views_list[i]
            self.start_rows[i] = current_row
            current_row += view.shape[0]

            # Get C-level information of the array
            self.arrays[i].data_ptr = <cnp.float64_t *>cnp.PyArray_DATA(view)
            self.arrays[i].shape0 = view.shape[0]
            self.arrays[i].strides0 = view.strides[0]

            # Keep reference to the original array
            self.views.append(view)

        self.total_rows = current_row

    def __dealloc__(self):
        if self.arrays != NULL:
            free(self.arrays)
        if self.start_rows != NULL:
            free(self.start_rows)
        # No need to manually release self.views, Python garbage collection will handle it
        self.views = None

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline cython.int find_block(self, cython.int idx) nogil:
        cdef cython.int left = 0
        cdef cython.int right = self.num_blocks - 1
        cdef cython.int mid
        cdef cnp.npy_intp *start_rows = self.start_rows

        while left <= right:
            mid = (left + right) // 2
            if start_rows[mid] <= idx:
                if mid == self.num_blocks - 1 or start_rows[mid + 1] > idx:
                    return mid
                left = mid + 1
            else:
                right = mid - 1
        return -1  # Index out of range

    def __getitem__(self, idx):
        cdef cython.int normalized_idx
        cdef cnp.float64_t *data_ptr
        cdef object result
        cdef cython.int block_idx
        cdef cnp.npy_intp dims[1]

        if isinstance(idx, (int, np.integer)):
            normalized_idx = idx if idx >= 0 else self.total_rows + idx
            if normalized_idx < 0 or normalized_idx >= self.total_rows:
                raise IndexError("Index out of range")

            block_idx = self.find_block(normalized_idx)
            if block_idx == -1:
                raise IndexError("Index out of range")

            data_ptr = self.get_row_ptr(normalized_idx)
            if data_ptr == NULL:
                raise IndexError("Index out of range")

            # Create NumPy array
            dims[0] = self.vector_dim
            result = cnp.PyArray_SimpleNewFromData(
                1, dims, cnp.NPY_FLOAT64, <void *>data_ptr)
            if result is None:
                raise MemoryError("Failed to create array")

            # Set base object to prevent memory from being reclaimed
            cnp.PyArray_SetBaseObject(<cnp.ndarray>result, self.views[block_idx])

            return result

        else:
            raise TypeError("Only integer index is supported")

    cdef cnp.float64_t *get_row_ptr(self, cython.int idx) nogil:
        cdef cython.int block_idx
        cdef cnp.npy_intp block_start, row_idx
        cdef ArrayInfo *array_info

        block_idx = self.find_block(idx)
        if block_idx == -1:
            return NULL
        block_start = self.start_rows[block_idx]
        row_idx = idx - block_start
        array_info = &self.arrays[block_idx]

        if row_idx < 0 or row_idx >= array_info.shape0:
            return NULL

        return array_info.data_ptr + row_idx * (array_info.strides0 // sizeof(cnp.float64_t))

    def __array__(self):
        """NumPy array interface, return a complete array copy"""
        result = np.empty((int(self.total_rows), self.vector_dim), dtype=np.float64)
        cdef cnp.npy_intp i
        cdef cnp.npy_intp start_idx = 0
        cdef cnp.npy_intp num_rows
        cdef object view

        for i in range(self.num_blocks):
            view = self.views[i]
            num_rows = view.shape[0]
            result[int(start_idx):int(start_idx + num_rows), :] = view
            start_idx += num_rows

        return result

    def copy(self):
        """Create a copy of the array"""
        return self.__array__()

    def __len__(self):
        """Return the number of rows"""
        return self.total_rows

    @property
    def shape(self):
        return (self.total_rows, self.vector_dim)

    @property
    def dtype(self):
        return np.float64

    @property
    def ndim(self):
        """Array dimension"""
        return 2

    @property
    def size(self):
        """Total number of array elements"""
        return self.total_rows * self.vector_dim

    def reshape(self, *shape):
        """Reshape array"""
        return self.__array__().reshape(*shape)

    def to_numpy(self):
        """Convert to NumPy array"""
        return self.__array__()
    
    def tolist(self):
        """Convert to nested list"""
        return self.__array__().tolist()

    cdef str _format_row(self, cnp.float64_t *data_ptr):
        cdef list row_values = []
        cdef cnp.npy_intp j
        cdef cnp.float64_t value
        cdef int vector_dim = self.vector_dim
        cdef int max_cols = 6
        cdef int half_cols = max_cols // 2

        if vector_dim <= max_cols:
            # Display all columns
            for j in range(vector_dim):
                value = data_ptr[j]
                row_values.append(f"{value:<.8f}")
            return f"[{', '.join(row_values)}]"
        else:
            # Display partial columns
            for j in range(half_cols):
                value = data_ptr[j]
                row_values.append(f"{value:<.8f}")
            row_values.append("...")
            for j in range(vector_dim - half_cols, vector_dim):
                value = data_ptr[j]
                row_values.append(f"{value:<.8f}")
            return f"[{', '.join(row_values)}]"

    def __repr__(self):
        """NumPy-style string representation"""
        cdef:
            int max_rows = 12  # Maximum number of rows to display
            int half_rows = max_rows // 2  # Number of rows to display in each half
            cnp.npy_intp total_rows = self.total_rows
            cnp.npy_intp i
            list rows = []
            cnp.float64_t *data_ptr

        if total_rows <= max_rows:
            # Display all rows
            for i in range(total_rows):
                data_ptr = self.get_row_ptr(i)
                rows.append(self._format_row(data_ptr))
        else:
            # Display top rows
            for i in range(half_rows):
                data_ptr = self.get_row_ptr(i)
                rows.append(self._format_row(data_ptr))
            # Add ellipsis
            rows.append("...")
            # Display bottom rows
            for i in range(total_rows - half_rows, total_rows):
                data_ptr = self.get_row_ptr(i)
                rows.append(self._format_row(data_ptr))

        # Combine final string
        content = ",\n ".join(rows)
        return f"ArrayView([\n {content}\n])"

    def __str__(self):
        """String style"""
        return self.__repr__()
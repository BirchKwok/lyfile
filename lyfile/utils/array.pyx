# array.pyx
import cython
cimport numpy as cnp
import numpy as np
from libc.stdlib cimport malloc, free

# 初始化 NumPy
cnp.import_array()

cdef struct ArrayInfo:
    cnp.float64_t *data_ptr
    cnp.npy_intp shape0
    cnp.npy_intp strides0

cdef class ArrayView:
    cdef:
        ArrayInfo *arrays  # 数组信息的指针数组
        cnp.npy_intp *start_rows  # 每个块的起始行索引数组
        list views  # 保持对原始数组的引用，防止被回收
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
            raise MemoryError("内存分配失败")

        for i in range(self.num_blocks):
            view = views_list[i]
            self.start_rows[i] = current_row
            current_row += view.shape[0]

            # 获取数组的 C 级别信息
            self.arrays[i].data_ptr = <cnp.float64_t *>cnp.PyArray_DATA(view)
            self.arrays[i].shape0 = view.shape[0]
            self.arrays[i].strides0 = view.strides[0]

            # 保持对原始数组的引用
            self.views.append(view)

        self.total_rows = current_row

    def __dealloc__(self):
        if self.arrays != NULL:
            free(self.arrays)
        if self.start_rows != NULL:
            free(self.start_rows)
        # 无需手动释放 self.views，Python 垃圾回收会处理
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
        return -1  # 索引超出范围

    def __getitem__(self, idx):
        cdef cython.int normalized_idx
        cdef cnp.float64_t *data_ptr
        cdef object result
        cdef cython.int block_idx
        cdef cnp.npy_intp dims[1]

        if isinstance(idx, (int, np.integer)):
            normalized_idx = idx if idx >= 0 else self.total_rows + idx
            if normalized_idx < 0 or normalized_idx >= self.total_rows:
                raise IndexError("索引超出范围")

            block_idx = self.find_block(normalized_idx)
            if block_idx == -1:
                raise IndexError("索引超出范围")

            data_ptr = self.get_row_ptr(normalized_idx)
            if data_ptr == NULL:
                raise IndexError("索引超出范围")

            # 创建 NumPy 数组
            dims[0] = self.vector_dim
            result = cnp.PyArray_SimpleNewFromData(
                1, dims, cnp.NPY_FLOAT64, <void *>data_ptr)
            if result is None:
                raise MemoryError("无法创建数组")

            # 设置 base 对象，防止内存被回收
            cnp.PyArray_SetBaseObject(<cnp.ndarray>result, self.views[block_idx])

            return result

        else:
            raise TypeError("只支持整数索引")

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
        """NumPy数组接口，返回完整的数组副本"""
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
        """创建数组的副本"""
        return self.__array__()

    def __len__(self):
        """返回行数"""
        return self.total_rows

    @property
    def shape(self):
        return (self.total_rows, self.vector_dim)

    @property
    def dtype(self):
        return np.float64

    @property
    def ndim(self):
        """数组维度"""
        return 2  # 二维数组

    @property
    def size(self):
        """数组元素总数"""
        return self.total_rows * self.vector_dim

    @property
    def T(self):
        """转置视图"""
        return self.transpose()

    def transpose(self, *axes):
        """转置操作"""
        return self.__array__().transpose(*axes)

    def reshape(self, *shape):
        """重塑数组形状"""
        return self.__array__().reshape(*shape)

    def astype(self, dtype):
        """转换数据类型"""
        if dtype == np.float64:
            return self
        return np.array(self, dtype=dtype)

    def tolist(self):
        """转换为嵌套列表"""
        return self.__array__().tolist()

    def __repr__(self):
        """NumPy风格的字符串表示"""
        cdef:
            int max_rows = 12  # 最多显示的行数
            int half_rows = max_rows // 2  # 每一半显示的行数
            int max_cols = 6  # 最多显示的列数
            int half_cols = max_cols // 2  # 每一半显示的列数
            int total_rows = self.total_rows
            int vector_dim = self.vector_dim
            int i, j
            list rows = []
            cnp.float64_t *data_ptr
            cnp.float64_t value

        # 定义格式化单行向量的函数
        def format_row(data_ptr):
            cdef list row_values = []
            if vector_dim <= max_cols:
                # 显示所有列
                for j in range(vector_dim):
                    value = data_ptr[j]
                    row_values.append(f"{value:<.8f}")
                return f"[{', '.join(row_values)}]"
            else:
                # 显示部分列
                for j in range(half_cols):
                    value = data_ptr[j]
                    row_values.append(f"{value:<.8f}")
                row_values.append("...")
                for j in range(vector_dim - half_cols, vector_dim):
                    value = data_ptr[j]
                    row_values.append(f"{value:<.8f}")
                return f"[{', '.join(row_values)}]"

        if total_rows <= max_rows:
            # 显示所有行
            for i in range(total_rows):
                data_ptr = self.get_row_ptr(i)
                rows.append(format_row(data_ptr))
        else:
            # 显示头部行
            for i in range(half_rows):
                data_ptr = self.get_row_ptr(i)
                rows.append(format_row(data_ptr))
            # 添加省略号
            rows.append("...")
            # 显示尾部行
            for i in range(total_rows - half_rows, total_rows):
                data_ptr = self.get_row_ptr(i)
                rows.append(format_row(data_ptr))

        # 组合最终字符串
        content = ",\n ".join(rows)
        return f"ArrayView([\n {content}\n])"

    def __str__(self):
        """字符串风格"""
        return self.__repr__()

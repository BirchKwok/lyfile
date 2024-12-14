use std::cmp::Ordering;

use pyo3::prelude::*;
use pyo3::types::PyAny;
use numpy::PyArray2;
use pyo3::PyResult;
use simsimd::SpatialSimilarity;
use num_traits::{Float, FromPrimitive, ToPrimitive};


pub fn search_vector_internal<T>(
    base_vecs: Vec<T>, 
    query_vectors: &PyAny, 
    top_k: usize, 
    metric: &str,
    py: Python<'_>
) -> PyResult<(Py<PyArray2<i32>>, Py<PyArray2<f32>>)> 
where 
    T: SpatialSimilarity + Copy + 'static + PartialOrd + Float + numpy::Element + FromPrimitive,
    for<'py> &'py PyArray2<T>: FromPyObject<'py>,
{
    let query_array = if query_vectors.getattr("dtype")?.str()?.to_string().to_lowercase().contains(std::any::type_name::<T>()) {
        query_vectors.extract::<&PyArray2<T>>()?
    } else {
        let dtype = if std::any::type_name::<T>() == "f32" { "float32" } else { "float64" };
        let array = query_vectors.call_method1("astype", (dtype,))?;
        array.extract::<&PyArray2<T>>()?
    };
    
    let query_slice = unsafe { query_array.as_slice()? };
    let query_shape = query_array.shape();
    let n_queries = query_shape[0];
    let dim = query_shape[1];
    
    let mut indices = vec![vec![0i32; top_k]; n_queries];
    let mut distances = vec![vec![0f32; top_k]; n_queries];
    
    let base_vecs: Vec<T> = base_vecs.into_iter()
        .map(|x| num_traits::cast(x).unwrap_or_else(T::zero))
        .collect();
    
    for (query_idx, query_vec) in query_slice.chunks(dim).enumerate() {
        let mut distances_with_indices: Vec<_> = base_vecs.chunks(dim)
            .enumerate()
            .map(|(i, base_vec)| {
                let dist = match metric {
                    "l2" => {
                        T::sqeuclidean(query_vec, base_vec)
                            .map(|x| x.to_f64().unwrap_or(1e30))
                            .unwrap_or(1e30)
                    },
                    "cosine" => {
                        T::cosine(query_vec, base_vec)
                                .map(|x| x.to_f64().unwrap_or(-1e30))
                                .unwrap_or(-1e30)
                    },
                    "ip" => {
                        T::dot(query_vec, base_vec)
                            .map(|x| x.to_f64().unwrap_or(-1e30))
                            .unwrap_or(-1e30)
                    },
                    _ => unreachable!(),
                };
                (T::from_f64(dist).unwrap_or_else(T::zero), i as i32)
            })
            .collect();

        if metric == "ip" {
            // IP距离：值越大越相似
            block_partition_sort(&mut distances_with_indices, top_k, true);
        } else {
            // L2和cosine距离：值越小越相似
            block_partition_sort(&mut distances_with_indices, top_k, false);
        }

        // 确保只保留top_k个结果
        distances_with_indices.truncate(top_k);
        
        for k in 0..top_k {
            let (dist, idx) = distances_with_indices[k];
            indices[query_idx][k] = idx;
            distances[query_idx][k] = dist.to_f32().unwrap_or(0.0);
        }
    }
    
    let indices_array = PyArray2::from_vec2(py, &indices)?;
    let distances_array = PyArray2::from_vec2(py, &distances)?;
    
    Ok((indices_array.to_owned(), distances_array.to_owned()))
}

// 获取pivot的辅助函数
fn get_pivot<T: PartialOrd + Copy>(arr: &[(T, i32)], left: usize, right: usize) -> usize {
    let mid = left + (right - left) / 2;
    let (a, b, c) = (arr[left].0, arr[mid].0, arr[right].0);
    
    if a <= b {
        if b <= c { mid }
        else if a <= c { right }
        else { left }
    } else {
        if a <= c { left }
        else if b <= c { right }
        else { mid }
    }
}

// 1. 插入排序 - 升序
fn insertion_sort<T: PartialOrd + Copy>(arr: &mut [(T, i32)], left: usize, right: usize) {
    for i in (left + 1)..=right {
        let mut j = i;
        while j > left && arr[j - 1].0 > arr[j].0 {
            arr.swap(j - 1, j);
            j -= 1;
        }
    }
}

// 2. 块排序优化
fn block_sort<T: PartialOrd + Copy>(arr: &mut [(T, i32)], left: usize, right: usize) {
    const BLOCK_SIZE: usize = 32;  // 可以根据实际情况调整
    let len = right - left + 1;
    
    if len <= 1 {
        return;
    }
    
    if len <= BLOCK_SIZE {
        insertion_sort(arr, left, right);
        return;
    }
    
    // 将数组分成若干块并分别排序
    for i in (left..=right).step_by(BLOCK_SIZE) {
        let end = (i + BLOCK_SIZE - 1).min(right);
        insertion_sort(arr, i, end);
    }
}

// 3. 采样预处理
fn sample_and_presort<T: PartialOrd + Copy>(arr: &mut [(T, i32)], left: usize, right: usize) {
    const SAMPLE_SIZE: usize = 5;
    let len = right - left + 1;
    
    if len <= SAMPLE_SIZE {
        return;
    }
    
    // 采样并排序
    let step = len / SAMPLE_SIZE;
    let mut samples = Vec::with_capacity(SAMPLE_SIZE);
    
    for i in 0..SAMPLE_SIZE {
        let idx = left + i * step;
        samples.push((arr[idx].0, idx));
    }
    
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    
    // 将采样点移动到合适的位置
    for (i, &(_, idx)) in samples.iter().enumerate() {
        let target = left + i * step;
        if target != idx {
            arr.swap(target, idx);
        }
    }
}

// 4. 修改quick_select函数
fn quick_select_k_largest<T: PartialOrd + Copy>(arr: &mut [(T, i32)], k: usize) {
    if k >= arr.len() {
        return;
    }
    
    const BLOCK_SIZE: usize = 32;
    let mut left = 0;
    let mut right = arr.len() - 1;
    let target = k - 1;
    
    while right - left > BLOCK_SIZE {
        // 对大数组进行采样预处理
        if right - left > 1000 {
            sample_and_presort(arr, left, right);
        }
        
        let (lt, gt) = partition3way_desc(arr, left, right);
        
        if target <= gt {
            right = gt;
        } else if target >= lt {
            left = lt;
        } else {
            // 找到目标区间后，对该区间进行块排序
            block_sort_desc(arr, left.min(target), right.min(target + BLOCK_SIZE));
            return;
        }
    }
    
    // 对小区间使用块排序
    block_sort_desc(arr, left, right);
}

// 同样修改quick_select_k_smallest
fn quick_select_k_smallest<T: PartialOrd + Copy>(arr: &mut [(T, i32)], k: usize) {
    if k >= arr.len() {
        return;
    }
    
    const BLOCK_SIZE: usize = 32;
    let mut left = 0;
    let mut right = arr.len() - 1;
    
    while right - left > BLOCK_SIZE {
        if right - left > 1000 {
            sample_and_presort(arr, left, right);
        }
        
        let (lt, gt) = partition3way_asc(arr, left, right);
        
        if k <= lt {
            right = lt - 1;
        } else if k > gt {
            left = gt + 1;
        } else {
            block_sort(arr, left.min(k), right.min(k + BLOCK_SIZE));
            return;
        }
    }
    
    block_sort(arr, left, right);
}

// 降序版本(用于quick_select_k_largest)
fn partition3way_desc<T: PartialOrd + Copy>(arr: &mut [(T, i32)], left: usize, right: usize) 
    -> (usize, usize) 
{
    let pivot_idx = get_pivot(arr, left, right);
    let pivot_val = arr[pivot_idx].0;
    
    arr.swap(pivot_idx, left);
    
    let mut lt = left;      // 大于pivot的右边界
    let mut i = left + 1;   // 当前处理的元素
    let mut gt = right;     // 小于pivot的左边界
    
    while i <= gt {
        match arr[i].0.partial_cmp(&pivot_val).unwrap_or(Ordering::Equal) {
            Ordering::Greater => {
                arr.swap(lt, i);
                lt += 1;
                i += 1;
            },
            Ordering::Less => {
                arr.swap(i, gt);
                if gt > 0 {
                    gt -= 1;
                }
            },
            Ordering::Equal => {
                i += 1;
            }
        }
    }
    
    (lt, gt)
}

// 升序版本(用于quick_select_k_smallest)
fn partition3way_asc<T: PartialOrd + Copy>(arr: &mut [(T, i32)], left: usize, right: usize) 
    -> (usize, usize) 
{
    let pivot_idx = get_pivot(arr, left, right);
    let pivot_val = arr[pivot_idx].0;
    
    arr.swap(pivot_idx, left);
    
    let mut lt = left;      // 小于pivot的右边界
    let mut i = left + 1;   // 当前处理的元素
    let mut gt = right;     // 大于pivot的左边界
    
    while i <= gt {
        match arr[i].0.partial_cmp(&pivot_val).unwrap_or(Ordering::Equal) {
            Ordering::Less => {
                arr.swap(lt, i);
                lt += 1;
                i += 1;
            },
            Ordering::Greater => {
                arr.swap(i, gt);
                if gt > 0 {
                    gt -= 1;
                }
            },
            Ordering::Equal => {
                i += 1;
            }
        }
    }
    
    (lt, gt)
}

// 添加降序块排序
fn block_sort_desc<T: PartialOrd + Copy>(arr: &mut [(T, i32)], left: usize, right: usize) {
    const BLOCK_SIZE: usize = 32;
    let len = right - left + 1;
    
    if len <= 1 {
        return;
    }
    
    if len <= BLOCK_SIZE {
        insertion_sort_desc(arr, left, right);
        return;
    }
    
    for i in (left..=right).step_by(BLOCK_SIZE) {
        let end = (i + BLOCK_SIZE - 1).min(right);
        insertion_sort_desc(arr, i, end);
    }
}

// 降序插入排序
fn insertion_sort_desc<T: PartialOrd + Copy>(arr: &mut [(T, i32)], left: usize, right: usize) {
    for i in (left + 1)..=right {
        let mut j = i;
        while j > left && arr[j - 1].0 < arr[j].0 {  // 修改比较符号为 <
            arr.swap(j - 1, j);
            j -= 1;
        }
    }
}

// 分块处理函数
fn block_partition_sort<T: PartialOrd + Copy>(
    arr: &mut [(T, i32)], 
    k: usize,
    is_max: bool  // true表示找最大的k个，false表示找最小的k个
) {
    let len = arr.len();
    if len <= k {
        return;
    }

    // 确定块大小，每个块至少要k个元素
    let block_size = (len / (len / k).max(1)).max(k);
    let num_blocks = (len + block_size - 1) / block_size;
    
    // 存储每个块的top-k结果
    let mut block_tops = Vec::with_capacity(num_blocks * k);
    
    // 1. 对每个块进行局部排序，选出top-k
    for block_idx in 0..num_blocks {
        let start = block_idx * block_size;
        let end = (start + block_size).min(len);
        let block = &mut arr[start..end];
        
        // 对块内元素进行快速选择
        if is_max {
            quick_select_k_largest(block, k.min(block.len()));
            block_tops.extend_from_slice(&block[..k.min(block.len())]);
        } else {
            quick_select_k_smallest(block, k.min(block.len()));
            block_tops.extend_from_slice(&block[..k.min(block.len())]);
        }
    }
    
    // 2. 对所有块的top-k结果进行最终排序
    if is_max {
        quick_select_k_largest(&mut block_tops, k);
        block_tops[..k].sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    } else {
        quick_select_k_smallest(&mut block_tops, k);
        block_tops[..k].sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    }
    
    // 3. 将最终结果复制回原数组
    arr[..k].copy_from_slice(&block_tops[..k]);
}



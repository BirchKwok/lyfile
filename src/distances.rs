// distances.rs
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::{Arc, Mutex};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray2, PyReadonlyArray2, IntoPyArray};
use ndarray::{Array2, s, ArrayView1, parallel::prelude::*,ArrayBase, ViewRepr, Dim};
use rayon::prelude::*;

// 添加常量定义
const BATCH_SIZE: usize = 1024;  // 批处理大小
const MIN_PARALLEL_SIZE: usize = 1000;  // 最小并行处理阈值

// 优化 DistanceItem 结构体的内存布局
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
struct DistanceItem {
    distance: f32,  // 把 distance 放在前面以优化内存对齐
    index: i32,
}

// 为最小堆实现 Ord
impl Ord for DistanceItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // 使用小于号来实现最大堆（较小的值具有较低的优先级）
        // 这样 pop 会移除最大的元素，留下最小的元素
        self.distance.partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for DistanceItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for DistanceItem {}

// 新增用于 SIMD 计算的辅助函数
#[inline(always)]
fn compute_l2_distance_simd(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { compute_l2_distance_avx2(a.as_ptr(), b.as_ptr(), a.len()) }
        } else if is_x86_feature_detected!("sse4.1") {
            unsafe { compute_l2_distance_sse41(a.as_ptr(), b.as_ptr(), a.len()) }
        } else {
            compute_l2_distance_fallback(a, b)
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { compute_l2_distance_neon(a.as_ptr(), b.as_ptr(), a.len()) }
        } else {
            compute_l2_distance_fallback(a, b)
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        compute_l2_distance_fallback(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn compute_l2_distance_avx2(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::*;
    
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    
    // 每次处理8个f32
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
        i += 8;
    }
    
    // 水平求和
    let mut result = 0.0;
    let sum_array = std::mem::transmute::<__m256, [f32; 8]>(sum);
    for &x in &sum_array {
        result += x;
    }
    
    // 处理剩余元素
    while i < len {
        let diff = *a.add(i) - *b.add(i);
        result += diff * diff;
        i += 1;
    }
    
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn compute_l2_distance_sse41(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::*;
    
    let mut sum = _mm_setzero_ps();
    let mut i = 0;
    
    // 每次处理4个f32
    while i + 4 <= len {
        let va = _mm_loadu_ps(a.add(i));
        let vb = _mm_loadu_ps(b.add(i));
        let diff = _mm_sub_ps(va, vb);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        i += 4;
    }
    
    // 水平求和
    let mut result = 0.0;
    let sum_array = std::mem::transmute::<__m128, [f32; 4]>(sum);
    for &x in &sum_array {
        result += x;
    }
    
    // 处理剩余元素
    while i < len {
        let diff = *a.add(i) - *b.add(i);
        result += diff * diff;
        i += 1;
    }
    
    result
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn compute_l2_distance_neon(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::aarch64::*;
    
    let mut sum = vdupq_n_f32(0.0);
    let mut i = 0;
    
    // 每次处理4个f32
    while i + 4 <= len {
        let va = vld1q_f32(a.add(i));
        let vb = vld1q_f32(b.add(i));
        let diff = vsubq_f32(va, vb);
        sum = vfmaq_f32(sum, diff, diff);
        i += 4;
    }
    
    // 水平求和
    let mut result = 0.0;
    let sum_array = std::mem::transmute::<float32x4_t, [f32; 4]>(sum);
    for &x in &sum_array {
        result += x;
    }
    
    // 处理剩余元素
    while i < len {
        let diff = *a.add(i) - *b.add(i);
        result += diff * diff;
        i += 1;
    }
    
    result
}

#[inline(always)]
fn compute_l2_distance_fallback(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

// 优化后的主计算函数
#[pyfunction]
#[pyo3(signature = (query_vectors, base_vectors, top_k, metric="l2"))]
pub fn compute_distances(
    py: Python,
    query_vectors: PyReadonlyArray2<f32>,
    base_vectors: PyReadonlyArray2<f32>,
    top_k: usize,
    metric: &str,
) -> PyResult<(Py<PyArray2<i32>>, Py<PyArray2<f32>>)> {
    let query_array = query_vectors.as_array();
    let base_array = base_vectors.as_array();
    
    let n_queries = query_array.shape()[0];
    let _n_base = base_array.shape()[0];
    
    // 预分配结果数组
    let mut indices = vec![0; n_queries * top_k];
    let mut distances = vec![0.0; n_queries * top_k];
    
    // 根据数据规模决定是否使用并行处理
    if n_queries >= MIN_PARALLEL_SIZE {
        indices.par_chunks_mut(top_k)
            .zip(distances.par_chunks_mut(top_k))
            .enumerate()
            .for_each(|(query_idx, (indices_chunk, distances_chunk))| {
                process_single_query(
                    query_array.slice(s![query_idx, ..]),
                    &base_array.view(),
                    top_k,
                    metric,
                    indices_chunk,
                    distances_chunk,
                );
            });
    } else {
        for query_idx in 0..n_queries {
            process_single_query(
                query_array.slice(s![query_idx, ..]),
                &base_array.view(),
                top_k,
                metric,
                &mut indices[query_idx * top_k..(query_idx + 1) * top_k],
                &mut distances[query_idx * top_k..(query_idx + 1) * top_k],
            );
        }
    }
    
    // 转换为 Python 数组
    let indices_array = Array2::from_shape_vec((n_queries, top_k), indices)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let distances_array = Array2::from_shape_vec((n_queries, top_k), distances)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    Ok((
        indices_array.into_pyarray(py).to_owned(),
        distances_array.into_pyarray(py).to_owned(),
    ))
}

// 处理单个查询向量的函数
#[inline(always)]
fn process_single_query(
    query: ArrayView1<f32>,
    base_array: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
    top_k: usize,
    metric: &str,
    indices_out: &mut [i32],
    distances_out: &mut [f32],
) {
    let mut heap = BinaryHeap::with_capacity(top_k + 1);
    
    // 批量处理基向量
    for chunk_start in (0..base_array.shape()[0]).step_by(BATCH_SIZE) {
        let chunk_end = (chunk_start + BATCH_SIZE).min(base_array.shape()[0]);
        
        // 计算当前批次的所有距离
        for base_idx in chunk_start..chunk_end {
            let base_vec = base_array.slice(s![base_idx, ..]);
            
            let dist = match metric {
                "l2" => compute_l2_distance_simd(query, base_vec),
                "ip" => -query.dot(&base_vec),
                "cosine" => {
                    let norm_q = query.dot(&query).sqrt();
                    let norm_b = base_vec.dot(&base_vec).sqrt();
                    -query.dot(&base_vec) / (norm_q * norm_b)
                }
                _ => panic!("Unsupported metric"),
            };
            
            if heap.len() < top_k {
                heap.push(DistanceItem {
                    index: base_idx as i32,
                    distance: dist,
                });
            } else if dist < heap.peek().unwrap().distance {
                heap.pop();
                heap.push(DistanceItem {
                    index: base_idx as i32,
                    distance: dist,
                });
            }
        }
    }
    
    // 将堆中的结果转换为排序后的数组
    let mut items: Vec<_> = heap.into_vec();
    items.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    
    // 填充输出数组
    for (i, item) in items.iter().enumerate() {
        indices_out[i] = item.index;
        distances_out[i] = item.distance;
    }
}

// 添加新的泛型函数来处理不同精度
pub fn compute_distances_generic<T>(
    py: Python,
    query_vectors: PyReadonlyArray2<T>,
    base_vectors: PyReadonlyArray2<T>,
    top_k: usize,
    metric: &str,
) -> PyResult<(Py<PyArray2<i32>>, Py<PyArray2<T>>)>
where
    T: num_traits::Float + numpy::Element + Copy + Send + Sync + std::fmt::Debug + 'static,
    ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<[usize; 2]>>: IntoPyArray,
{
    let query_array = query_vectors.as_array();
    let base_array = base_vectors.as_array();
    
    let n_queries = query_array.shape()[0];
    let indices = Arc::new(Mutex::new(vec![0; n_queries * top_k]));
    let distances = Arc::new(Mutex::new(vec![T::zero(); n_queries * top_k]));
    
    rayon::scope(|s| {
        for query_idx in 0..n_queries {
            let query = query_array.slice(s![query_idx, ..]);
            let base = base_array.view();
            let indices = Arc::clone(&indices);
            let distances = Arc::clone(&distances);
            
            s.spawn(move |_| {
                let mut heap = BinaryHeap::with_capacity(top_k + 1);
                
                for (base_idx, base_vec) in base.outer_iter().enumerate() {
                    let dist = match metric {
                        "l2" => {
                            let mut sum = T::zero();
                            for (q, b) in query.iter().zip(base_vec.iter()) {
                                let diff = *q - *b;
                                // 防止溢出，限制单个差值的平方
                                let squared_diff = if diff.abs() > T::from(1e6).unwrap() {
                                    T::from(1e12).unwrap()
                                } else {
                                    diff * diff
                                };
                                sum = sum + squared_diff;
                            }
                            sum
                        },
                        "ip" => {
                            let mut sum = T::zero();
                            for (q, b) in query.iter().zip(base_vec.iter()) {
                                sum = sum + (*q * *b);
                            }
                            -sum
                        },
                        "cosine" => {
                            let mut dot = T::zero();
                            let mut norm_q = T::zero();
                            let mut norm_b = T::zero();
                            
                            for (q, b) in query.iter().zip(base_vec.iter()) {
                                dot = dot + (*q * *b);
                                norm_q = norm_q + (*q * *q);
                                norm_b = norm_b + (*b * *b);
                            }
                            
                            let norm_q = norm_q.sqrt();
                            let norm_b = norm_b.sqrt();
                            
                            if norm_q > T::zero() && norm_b > T::zero() {
                                -(dot / (norm_q * norm_b))
                            } else {
                                T::max_value()
                            }
                        },
                        _ => T::max_value(),
                    };
                    
                    if !dist.is_nan() && !dist.is_infinite() {  // 添加对无穷大的检查
                        let dist_f32 = dist.to_f32().unwrap_or(f32::MAX);
                        if dist_f32.is_finite() {  // 再次检查确保是有限值
                            heap.push(DistanceItem {
                                index: base_idx as i32,
                                distance: dist_f32,
                            });
                            
                            if heap.len() > top_k {
                                heap.pop();
                            }
                        }
                    }
                }
                
                let mut items: Vec<_> = heap.into_vec();
                items.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
                
                let mut indices = indices.lock().unwrap();
                let mut distances = distances.lock().unwrap();
                
                for (i, item) in items.iter().enumerate() {
                    indices[query_idx * top_k + i] = item.index;
                    distances[query_idx * top_k + i] = T::from(item.distance).unwrap();
                }
            });
        }
    });
    
    let indices_array = Array2::from_shape_vec((n_queries, top_k), 
        Arc::try_unwrap(indices).unwrap().into_inner().unwrap())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let distances_array = Array2::from_shape_vec((n_queries, top_k), 
        Arc::try_unwrap(distances).unwrap().into_inner().unwrap())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    Ok((
        PyArray2::from_owned_array(py, indices_array).to_owned(),
        PyArray2::from_owned_array(py, distances_array).to_owned()
    ))
}

// 具体类型的包装函数
pub fn compute_distances_f32(
    py: Python,
    query_vectors: PyReadonlyArray2<f32>,
    base_vectors: PyReadonlyArray2<f32>,
    top_k: usize,
    metric: &str,
) -> PyResult<(Py<PyArray2<i32>>, Py<PyArray2<f32>>)> {
    compute_distances_generic(py, query_vectors, base_vectors, top_k, metric)
}

pub fn compute_distances_f64(
    py: Python,
    query_vectors: PyReadonlyArray2<f64>,
    base_vectors: PyReadonlyArray2<f64>,
    top_k: usize,
    metric: &str,
) -> PyResult<(Py<PyArray2<i32>>, Py<PyArray2<f64>>)> {
    compute_distances_generic(py, query_vectors, base_vectors, top_k, metric)
}

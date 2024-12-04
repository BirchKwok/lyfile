// distances.rs
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray2, PyReadonlyArray2, IntoPyArray};
use ndarray::{Array2, s};
use std::sync::{Arc, Mutex};


// 用于存储距离计算结果的结构
#[derive(Copy, Clone, PartialEq, Debug)]
struct DistanceItem {
    index: i32,
    distance: f32,
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

// 向量距离计算的主要实现
#[pyfunction]
#[pyo3(signature = (query_vectors, base_vectors, top_k, metric="l2"))]
pub fn compute_distances(
    py: Python,
    query_vectors: PyReadonlyArray2<f32>,
    base_vectors: PyReadonlyArray2<f32>,
    top_k: usize,
    metric: &str,
) -> PyResult<(Py<PyArray2<i32>>, Py<PyArray2<f32>>)> {
    // 将 Python 数组转换为 Rust 数组
    let query_array = query_vectors.as_array();
    let base_array = base_vectors.as_array();
    
    let n_queries = query_array.shape()[0];
    // let n_base = base_array.shape()[0];
    
    let indices = Arc::new(Mutex::new(vec![0; n_queries * top_k]));
    let distances = Arc::new(Mutex::new(vec![0.0; n_queries * top_k]));
    
    // 使用 rayon 并行处理
    rayon::scope(|s| {
        for query_idx in 0..n_queries {
            let query = query_array.slice(s![query_idx, ..]);
            let base = base_array.view();
            let indices = Arc::clone(&indices);
            let distances = Arc::clone(&distances);
            
            s.spawn(move |_| {
                let mut heap = BinaryHeap::with_capacity(top_k + 1);
                
                // 计算距离
                for (base_idx, base_vec) in base.outer_iter().enumerate() {
                    let dist = match metric {
                        "l2" => {
                            let diff = &base_vec - &query;
                            diff.dot(&diff)
                        }
                        "ip" => -query.dot(&base_vec),
                        "cosine" => {
                            let norm_q = (query.dot(&query)).sqrt();
                            let norm_b = (base_vec.dot(&base_vec)).sqrt();
                            -query.dot(&base_vec) / (norm_q * norm_b)
                        }
                        _ => panic!("Unsupported metric"),
                    };
                    
                    heap.push(DistanceItem {
                        index: base_idx as i32,
                        distance: dist,
                    });
                    
                    if heap.len() > top_k {
                        heap.pop();
                    }
                }
                
                // 将堆中的结果转换为排序后的数组
                let mut items: Vec<_> = heap.into_vec();
                items.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
                
                let mut indices = indices.lock().unwrap();
                let mut distances = distances.lock().unwrap();
                
                for (i, item) in items.iter().enumerate() {
                    indices[query_idx * top_k + i] = item.index;
                    distances[query_idx * top_k + i] = item.distance;
                }
            });
        }
    });
    
    // 转换回 Python 数组
    let indices_array = Array2::from_shape_vec((n_queries, top_k), Arc::try_unwrap(indices).unwrap().into_inner().unwrap())
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py)
        .to_owned();
    let distances_array = Array2::from_shape_vec((n_queries, top_k), Arc::try_unwrap(distances).unwrap().into_inner().unwrap())
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py)
        .to_owned();
    
    let distances_array: Py<PyArray2<f32>> = distances_array;
    Ok((indices_array, distances_array))
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

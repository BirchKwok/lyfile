use pyo3::prelude::*;
use pyo3::types::PyAny;
use numpy::PyArray2;
use pyo3::PyResult;
use simsimd::SpatialSimilarity;
use num_traits::{Float, FromPrimitive, ToPrimitive};

macro_rules! compute_distance {
    ($query_vec:expr, $base_vec:expr, $metric:expr, $T:ty) => {{
        match $metric {
            "l2" => {
                <$T>::sqeuclidean($query_vec, $base_vec)
                    .map(|x| x.to_f64().unwrap_or(1e30))
                    .unwrap_or(1e30)
            },
            "cosine" => {
                <$T>::cosine($query_vec, $base_vec)
                    .map(|x| x.to_f64().unwrap_or(-1e30))
                    .unwrap_or(-1e30)
            },
            "ip" => {
                <$T>::dot($query_vec, $base_vec)
                    .map(|x| x.to_f64().unwrap_or(-1e30))
                    .unwrap_or(-1e30)
            },
            _ => unreachable!(),
        }
    }};
}

struct BinaryHeap<T> {
    data: Vec<(T, i32)>,
    is_max: bool,
}

impl<T: PartialOrd + Copy> BinaryHeap<T> {
    fn new(is_max: bool) -> Self {
        BinaryHeap {
            data: Vec::new(),
            is_max,
        }
    }

    fn push(&mut self, item: (T, i32)) {
        self.data.push(item);
        self.sift_up(self.data.len() - 1);
    }

    fn pop(&mut self) -> Option<(T, i32)> {
        if self.data.is_empty() {
            return None;
        }
        let result = self.data[0];
        let last = self.data.pop().unwrap();
        if !self.data.is_empty() {
            self.data[0] = last;
            self.sift_down(0);
        }
        Some(result)
    }

    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if (self.is_max && self.data[idx].0 > self.data[parent].0) ||
               (!self.is_max && self.data[idx].0 < self.data[parent].0) {
                self.data.swap(idx, parent);
                idx = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut idx: usize) {
        loop {
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;
            let mut largest = idx;

            if left < self.data.len() && 
               ((self.is_max && self.data[left].0 > self.data[largest].0) ||
                (!self.is_max && self.data[left].0 < self.data[largest].0)) {
                largest = left;
            }

            if right < self.data.len() && 
               ((self.is_max && self.data[right].0 > self.data[largest].0) ||
                (!self.is_max && self.data[right].0 < self.data[largest].0)) {
                largest = right;
            }

            if largest != idx {
                self.data.swap(idx, largest);
                idx = largest;
            } else {
                break;
            }
        }
    }
}

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
                let dist = compute_distance!(query_vec, base_vec, metric, T);
                (T::from_f64(dist).unwrap_or_else(T::zero), i as i32)
            })
            .collect();

        if metric == "ip" {
            // ip distance: the larger the better
            block_partition_sort(&mut distances_with_indices, top_k, true);
        } else {
            // L2 and cosine distance: the smaller the better
            block_partition_sort(&mut distances_with_indices, top_k, false);
        }

        // ensure only top_k results are kept
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

// block partition sort function
fn block_partition_sort<T: PartialOrd + Copy>(
    arr: &mut [(T, i32)], 
    k: usize,
    is_max: bool
) {
    let len = arr.len();
    if len <= k {
        return;
    }

    // use heap to maintain top-k
    let mut heap = BinaryHeap::new(!is_max); // note the negation here, because we want to keep k largest/smallest values
    
    // push first k elements into heap
    for i in 0..k {
        heap.push(arr[i]);
    }
    
    // process remaining elements
    for i in k..len {
        if (is_max && arr[i].0 > heap.data[0].0) ||
           (!is_max && arr[i].0 < heap.data[0].0) {
            heap.pop();
            heap.push(arr[i]);
        }
    }
    
    // put elements back into array in order
    let mut idx = k;
    while let Some(item) = heap.pop() {
        idx -= 1;
        arr[idx] = item;
    }
}

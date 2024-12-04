// lib.rs
mod py_api;
mod structs;
mod distances;

use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::PyResult; 

use structs::LyFile;

#[pymodule]
fn _lib_lyfile(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LyFile>()?;
    m.add_function(wrap_pyfunction!(distances::compute_distances, m)?)?;
    Ok(())
}

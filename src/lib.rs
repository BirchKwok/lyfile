// lib.rs
mod py_api;
mod structs;
mod io;
mod neighbors;

use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::PyResult; 

use structs::_LyFile;

#[pymodule]
fn _lib_lyfile(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<_LyFile>()?;
    Ok(())
}

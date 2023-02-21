use ahash::HashSet;
use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn nanogpt_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(unique_characters, m)?)?;
    Ok(())
}

#[pyfunction]
fn unique_characters(s: &str) -> HashSet<char> {
    HashSet::from_iter(s.chars())
}

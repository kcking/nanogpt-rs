use std::{
    collections::HashSet,
    hash::{BuildHasher, Hasher},
};

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
fn unique_characters(s: &str) -> HashSet<char, CharHasher> {
    HashSet::from_iter(s.chars())
}

#[derive(Debug, Clone, Default)]
struct CharHasher {
    c: char,
}
impl Hasher for CharHasher {
    fn finish(&self) -> u64 {
        self.c as _
    }

    fn write(&mut self, bytes: &[u8]) {
        if bytes.len() != 4 {
            return;
        }
        //  Gracefully fall-back to `0`
        let i = u32::from_ne_bytes(bytes.try_into().unwrap_or([0; 4]));
        //  Gracefully fall-back to 'a'
        self.c = char::from_u32(i).unwrap_or('a');
    }
}

impl BuildHasher for CharHasher {
    type Hasher = Self;

    fn build_hasher(&self) -> Self::Hasher {
        Self::default()
    }
}

use pyo3::prelude::*;

mod benchmarks;
mod ml;
mod tribles;
mod stdio;

/// Run bench on mnist784 dataset.
#[pyfunction]
fn bench_mnist_hnsw(fname: String, parallel: bool) {
    let mut stdout = stdio::stdout();
    benchmarks::ann_mnist_784_euclidean::run_hnsw(&mut stdout, fname, parallel).unwrap();
}

/// A Python module implemented in Rust.
#[pymodule]
fn dorf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    benchmarks::ann_mnist_784_euclidean::bench(m)?;
    m.add_function(wrap_pyfunction!(bench_mnist_hnsw, m)?)?;
    tribles::tribles_module(m)?;
    Ok(())
}

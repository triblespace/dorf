use pyo3::prelude::*;

mod ml;
mod benchmarks;
mod stdio;

/// Run bench on mnist784 dataset.
#[pyfunction]
fn bench_mnist_hnsw(fname: String, parallel: bool) {
    let mut stdout = stdio::stdout();
    benchmarks::ann_mnist_784_euclidean::run_hnsw(&mut stdout, fname, parallel).unwrap();
}

/// Run bench on mnist784 dataset.
#[pyfunction]
fn bench_mnist_dorf(fname: String, parallel: bool) {
    let mut stdout = stdio::stdout();
    benchmarks::ann_mnist_784_euclidean::run_dorf(&mut stdout, fname, parallel).unwrap();
}

/// A Python module implemented in Rust.
#[pymodule]
fn dorf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bench_mnist_hnsw, m)?)?;
    m.add_function(wrap_pyfunction!(bench_mnist_dorf, m)?)?;
    Ok(())
}

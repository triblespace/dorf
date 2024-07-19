use std::ops::Deref;

use pyo3::prelude::*;
use tribles;

#[pyclass(frozen)]
pub struct PyTrible(tribles::trible::Trible);

#[pyclass(frozen)]
pub struct PyTribleSet(tribles::TribleSet);

#[pymethods]
impl PyTribleSet {
    #[new]
    fn new() -> Self {
        PyTribleSet(tribles::TribleSet::new())
    }

    pub fn union(&self, other: &Bound<'_, Self>) -> Self {
        let mut result = self.0.clone();
        result.union(other.get().0.clone());
        PyTribleSet(result)
    }

    pub fn len(&self) -> usize {
        return self.0.eav.len() as usize;
    }

    pub fn insert(&self, trible: &Bound<'_, PyTrible>) -> Self {
        let mut result = self.0.clone();
        result.insert(&trible.get().0);
        PyTribleSet(result)
    }
}

#[pyclass(frozen)]
pub struct PyDuration();


/// The `tribles.types` python module.
pub fn types_module(pm: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(pm.py(), "types")?;
    m.add_class::<PyDuration>()?;
    pm.add_submodule(&m)?;
    Ok(())
}

/// The `tribles` python module.
pub fn tribles_module(pm: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(pm.py(), "tribles")?;
    m.add_class::<PyTribleSet>()?;
    types_module(&m);
    pm.add_submodule(&m)?;
    Ok(())
}

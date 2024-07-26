use pyo3::{prelude::*, types::PyBytes};
use tribles::{self, trible::TRIBLE_LEN, TribleSet};

#[pyclass]
pub struct PyTribleSet(tribles::TribleSet);

#[pymethods]
impl PyTribleSet {
    #[staticmethod]
    pub fn from_bytes(tribles: &Bound<'_, PyBytes>) -> Self {
        let tribles = tribles.as_bytes();
        assert!(tribles.len() % TRIBLE_LEN == 0);

        let mut set = tribles::TribleSet::new();

        for trible in tribles.chunks_exact(TRIBLE_LEN) {
            set.insert_raw(trible.try_into().unwrap());
        }

        PyTribleSet(set)
    }

    #[staticmethod]
    pub fn empty() -> Self {
        PyTribleSet(tribles::TribleSet::new())
    }

    pub fn __add__(&self, other: &Bound<'_, Self>) -> Self {
        let mut result = self.0.clone();
        result.union(other.borrow().0.clone());
        PyTribleSet(result)
    }

    pub fn consume(&mut self, other: &Bound<'_, Self>) {
        let set = &mut self.0;
        let other_set = std::mem::replace(&mut other.borrow_mut().0, TribleSet::new());
        set.union(other_set);
    }

    pub fn len(&self) -> usize {
        return self.0.eav.len() as usize;
    }
}

/*
#[pyclass(frozen)]
pub struct PyValue {
    bytes: [u8; 32],
    schema: [u8; 16]
}
*/

/// The `tribles` python module.
pub fn tribles_module(pm: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(pm.py(), "tribles")?;
    //m.add_class::<PyValue>()?;
    m.add_class::<PyTribleSet>()?;
    pm.add_submodule(&m)?;
    Ok(())
}

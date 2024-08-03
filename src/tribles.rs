use std::{collections::HashMap, sync::Arc};

use pyo3::{prelude::*, types::{PyBytes, PyList}};
use tribles::{self, query::{Binding, ConstantConstraint, Constraint, IntersectionConstraint, Query, Variable, VariableContext}, trible::TRIBLE_LEN, RawValue, TribleSet, Value};

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

    pub fn __iadd__(&mut self, other: &Bound<'_, Self>) {
        let set = &mut self.0;
        set.union(other.borrow().0.clone());
    }

    pub fn fork(&mut self) -> Self {
        PyTribleSet(self.0.clone())
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

#[pyclass(frozen)]
pub struct PyValue {
    bytes: [u8; 32],
    schema: [u8; 16]
}

#[pyclass]
pub struct PyQuery {
    query: Query<Arc<dyn Constraint<'static> + Send + Sync>, fn(&Binding) -> PyBinding, PyBinding>
}

#[pyclass(frozen)]
pub struct PyBinding {
    binding: Binding
}

#[pyclass(frozen)]
pub struct PyConstraint {
    constraint: Arc<dyn Constraint<'static> + Send + Sync>
}

/// Build a constraint for the intersection of the provided constraints.
#[pyfunction]
pub fn constant(index: u8, constant: &Bound<'_, PyValue>) -> PyConstraint {
    let constraint = Arc::new(ConstantConstraint::new(
        Variable::<RawValue>::new(index),
        Value::<RawValue>::new(constant.get().bytes)));

    PyConstraint {
        constraint
    }
}


/// Build a constraint for the intersection of the provided constraints.
#[pyfunction]
pub fn and(constraints: Vec<Py<PyConstraint>>) -> PyConstraint {
    let constraints = constraints.iter().map(|py| py.get().constraint.clone()).collect();
    let constraint = Arc::new(IntersectionConstraint::new(constraints));

    PyConstraint {
        constraint
    }
}

fn postprocessing(binding: &Binding) -> PyBinding {
    PyBinding { binding: binding.clone() }
}

/// Find solutions for the provided constraint.
#[pyfunction]
pub fn solve(constraint: &Bound<'_, PyConstraint>) -> PyQuery {
    let constraint = constraint.get().constraint.clone();

    let query = tribles::query::Query::new(constraint, postprocessing as fn(&Binding) -> PyBinding);

    PyQuery {
        query
    }
}

#[pymethods]
impl PyQuery {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PyBinding> {
        slf.query.next()
    }
}

/// The `tribles` python module.
pub fn tribles_module(pm: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(pm.py(), "tribles")?;
    //m.add_class::<PyValue>()?;
    m.add_class::<PyTribleSet>()?;
    m.add_class::<PyBinding>()?;
    m.add_class::<PyConstraint>()?;
    m.add_class::<PyQuery>()?;
    m.add_function(wrap_pyfunction!(constant, &m)?)?;
    m.add_function(wrap_pyfunction!(and, &m)?)?;
    m.add_function(wrap_pyfunction!(solve, &m)?)?;
    pm.add_submodule(&m)?;
    Ok(())
}

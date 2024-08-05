use std::{borrow::Cow, collections::HashMap, sync::Arc};

use pyo3::{prelude::*, types::PyBytes};
use tribles::{self, query::{Binding, ConstantConstraint, Constraint, IntersectionConstraint, Query, TriblePattern, Variable}, trible::TRIBLE_LEN, RawValue, TribleSet, Value};

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

    pub fn pattern(&self, ev: u8, av: u8, vv: u8) -> PyConstraint {
        PyConstraint {
            constraint: Arc::new(self.0.pattern(Variable::new(ev), Variable::new(av), Variable::<RawValue>::new(vv)))
        }
    }
}

#[pyclass(frozen)]
pub struct PyValue {
    bytes: [u8; 32],
    schema: [u8; 16]
}

#[pymethods]
impl PyValue {
    pub fn schema(&self) -> PyId {
        PyId {
            bytes: self.schema
        }
    }

    pub fn bytes(&self) -> Cow<[u8]> {
        (&self.bytes).into()
    }
}

#[pyclass(frozen)]
pub struct PyId {
    bytes: [u8; 16],
}

#[pyclass]
pub struct PyQuery {
    query: Query<Arc<dyn Constraint<'static> + Send + Sync>, Box<dyn Fn(&Binding) -> HashMap<u8, PyValue> + Send>, HashMap<u8, PyValue>>
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
pub fn intersect(constraints: Vec<Py<PyConstraint>>) -> PyConstraint {
    let constraints = constraints.iter().map(|py| py.get().constraint.clone()).collect();
    let constraint = Arc::new(IntersectionConstraint::new(constraints));

    PyConstraint {
        constraint
    }
}

/// Find solutions for the provided constraint.
#[pyfunction]
pub fn solve(projected: HashMap<u8, Py<PyId>> ,constraint: &Bound<'_, PyConstraint>) -> PyQuery {
    let constraint = constraint.get().constraint.clone();

    let postprocessing = Box::new(move |binding: &Binding| {
        let mut map = HashMap::new();
        for (&k, v) in &projected {
            map.insert(k, PyValue {
                bytes: binding.get(k).expect("constraint should contain projected variables"),
                schema: v.get().bytes
            });
        }
        map
    }) as Box<dyn Fn(&Binding) -> HashMap<u8, PyValue> + Send>;

    let query = tribles::query::Query::new(constraint, postprocessing);

    PyQuery {
        query
    }
}

#[pymethods]
impl PyQuery {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<HashMap<u8, PyValue>> {
        slf.query.next()
    }
}

/// The `tribles` python module.
pub fn tribles_module(pm: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(pm.py(), "tribles")?;
    //m.add_class::<PyValue>()?;
    m.add_class::<PyTribleSet>()?;
    m.add_class::<PyId>()?;
    m.add_class::<PyValue>()?;
    m.add_class::<PyConstraint>()?;
    m.add_class::<PyQuery>()?;
    m.add_function(wrap_pyfunction!(constant, &m)?)?;
    m.add_function(wrap_pyfunction!(intersect, &m)?)?;
    m.add_function(wrap_pyfunction!(solve, &m)?)?;
    pm.add_submodule(&m)?;
    Ok(())
}

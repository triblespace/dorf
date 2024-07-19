use pyo3::prelude::*;

use cpu_time::ProcessTime;
use std::{
    io::Write,
    time::{Duration, SystemTime},
};
use tribles::{types::hash::Blake3, BlobSet, types::zc::ZC};

use anndists::dist::*;
use hnsw_rs::prelude::*;

use crate::ml::{Embedding, Metric, MetricL2, SW};

pub fn run_hnsw(stdout: &mut impl Write, fname: String, parallel: bool) -> Result<(), hdf5::Error> {
    // # load dataset
    let file = hdf5::File::open(&fname)?;

    // load distance data
    let test_distances = file.dataset("distances")?.read_2d::<f32>()?;

    // load neighbours
    let _ = file.dataset("neighbors")?.read_2d::<i32>()?;

    // load test data
    let test_data: Vec<_> = file
        .dataset("test")?
        .read_2d::<f32>()?
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();

    // load train data
    let train_data: Vec<_> = file
        .dataset("train")?
        .read_2d::<f32>()?
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();

    let knbn_max = test_distances.dim().1;
    let nb_elem = train_data.len();
    writeln!(
        stdout,
        "Train size : {}, test size : {}",
        nb_elem,
        test_data.len()
    )
    .unwrap();
    writeln!(stdout, "Nb neighbours answers for test data : {}", knbn_max).unwrap();
    //
    let max_nb_connection = 24;
    let nb_layer = 16.min((nb_elem as f32).ln().trunc() as usize);
    let ef_c = 400;
    writeln!(
        stdout,
        " number of elements to insert {:?} , setting max nb layer to {:?} ef_construction {:?}",
        nb_elem, nb_layer, ef_c
    )
    .unwrap();
    writeln!(
        stdout,
        " ====================================================================================="
    )
    .unwrap();
    let nb_search = test_data.len();
    writeln!(stdout, " number of search {:?}", nb_search).unwrap();

    let mut hnsw = Hnsw::<f32, DistL2>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistL2 {});
    hnsw.set_extend_candidates(false);

    // parallel insertion
    let mut start = ProcessTime::now();

    let mut now = SystemTime::now();
    let data_for_par_insertion = train_data
        .iter()
        .enumerate()
        .map(|(i, x)| (x.as_slice(), i))
        .collect();

    if parallel {
        writeln!(stdout, " \n parallel insertion").unwrap();
        hnsw.parallel_insert_slice(&data_for_par_insertion);
    } else {
        writeln!(stdout, " \n serial insertion").unwrap();
        for d in data_for_par_insertion {
            hnsw.insert_slice(d);
        }
    }
    let mut cpu_time: Duration = start.elapsed();
    //
    writeln!(
        stdout,
        "\n hnsw data insertion cpu time  {:?}  system time {:?} ",
        cpu_time,
        now.elapsed()
    )
    .unwrap();

    hnsw.dump_layer_info();
    writeln!(
        stdout,
        " hnsw data nb point inserted {:?}",
        hnsw.get_nb_point()
    )
    .unwrap();
    //
    //  Now the bench with 10 neighbours
    //
    let mut recalls = Vec::<usize>::with_capacity(nb_elem);
    let mut nb_returned = Vec::<usize>::with_capacity(nb_elem);
    let mut last_distances_ratio = Vec::<f32>::with_capacity(nb_elem);
    let mut knn_neighbours_for_tests = Vec::<Vec<Neighbour>>::with_capacity(nb_elem);
    hnsw.set_searching_mode(true);
    let knbn = 10;
    let ef_c = max_nb_connection;
    writeln!(stdout, "\n searching with ef : {:?}", ef_c).unwrap();
    start = ProcessTime::now();
    now = SystemTime::now();
    // search
    if parallel {
        writeln!(stdout, " \n parallel search").unwrap();
        knn_neighbours_for_tests = hnsw.parallel_search(&test_data, knbn, ef_c);
    } else {
        writeln!(stdout, " \n serial search").unwrap();
        for t in test_data {
            let knn_neighbours: Vec<Neighbour> = hnsw.search(&t, knbn, ef_c);
            knn_neighbours_for_tests.push(knn_neighbours);
        }
    }
    cpu_time = start.elapsed();
    let search_sys_time = now.elapsed().unwrap().as_micros() as f32;
    let search_cpu_time = cpu_time.as_micros() as f32;
    writeln!(
        stdout,
        "total cpu time for search requests {:?} , system time {:?} ",
        search_cpu_time, search_sys_time
    )
    .unwrap();
    // now compute recall rate
    for (neighbours, true_distances) in knn_neighbours_for_tests
        .into_iter()
        .zip(test_distances.rows())
    {
        let max_dist = true_distances[knbn - 1];
        let mut _knn_neighbours_id: Vec<usize> = neighbours.iter().map(|p| p.d_id).collect();
        let knn_neighbours_dist: Vec<f32> = neighbours.iter().map(|p| p.distance).collect();
        nb_returned.push(knn_neighbours_dist.len());
        // count how many distances of knn_neighbours_dist are less than
        let recall = knn_neighbours_dist
            .iter()
            .filter(|x| *x <= &max_dist)
            .count();
        recalls.push(recall);
        let mut ratio = 0.;
        if knn_neighbours_dist.len() >= 1 {
            ratio = knn_neighbours_dist[knn_neighbours_dist.len() - 1] / max_dist;
        }
        last_distances_ratio.push(ratio);
    }
    let mean_fraction_returned =
        (nb_returned.iter().sum::<usize>() as f32) / ((nb_returned.len() * knbn) as f32);
    writeln!(
        stdout,
        "mean fraction returned by search {:?} ",
        mean_fraction_returned
    )
    .unwrap();

    let last_distances_ratio =
        last_distances_ratio.iter().sum::<f32>() / last_distances_ratio.len() as f32;
    writeln!(stdout, "last distances ratio {:?}", last_distances_ratio).unwrap();

    let mean_recall = (recalls.iter().sum::<usize>() as f32) / ((knbn * recalls.len()) as f32);
    writeln!(stdout, "recall rate: {:?}", mean_recall).unwrap();

    let reqs = (nb_search as f32) * 1.0e+6_f32 / search_sys_time;
    writeln!(stdout, "req/s {:?}", reqs).unwrap();

    Ok(())
}

#[pyclass]
pub struct MNISTSW(SW<Blake3, ZC<Embedding<784, f32>>, MetricL2>);

/// Setup mnist784 dorf small world.
#[pyfunction]
pub fn setup(fname: String, random_layers: usize) -> MNISTSW {
    // # load dataset
    let file = hdf5::File::open(&fname).unwrap();

    // load train data
    let train_data: Vec<ZC<Embedding<784, f32>>> = file
        .dataset("train").unwrap()
        .read_2d::<f32>().unwrap()
        .rows()
        .into_iter()
        .map(|row| {
            let slice = row.as_slice().unwrap();
            let embedding: Embedding<784, f32> = slice.try_into().unwrap();
            let zc_embedding: ZC<Embedding<784, f32>> = embedding.into();
            zc_embedding
        })
        .collect();

    let blobs: BlobSet<Blake3> = BlobSet::new();
    let mut sw = SW::new(blobs);

    for d in &train_data {
        sw.insert(d.clone());
    }

    sw.prepare(random_layers);

    MNISTSW(sw)
}

#[pyclass]
pub struct StepResult {
    #[pyo3(get)]
    cpu_time: Duration,
    #[pyo3(get)]
    changes: usize,
    #[pyo3(get)]
    layer_explored: usize
}

/// Run step on mnist784 dorf small world.
#[pyfunction]
pub fn step(sw: &Bound<'_, MNISTSW>) -> StepResult {
    let sw = &mut sw.borrow_mut().0;

    let start = ProcessTime::now();
    let (layer_explored, changes) = sw.multilayer_step();
    let cpu_time = start.elapsed();

    StepResult {
        cpu_time,
        changes,
        layer_explored
    }
}

#[pyclass]
pub struct EvalResult {
    #[pyo3(get)]
    avg_cpu_time: Duration,
    #[pyo3(get)]
    avg_distance: f32
}

/// Run evaluation on mnist784 dorf small world.
#[pyfunction]
pub fn eval(sw: &Bound<'_, MNISTSW>, fname: String, search_span: usize) -> EvalResult {
    let sw = &mut sw.borrow_mut().0;

    // # load dataset
    let file = hdf5::File::open(&fname).unwrap();

    // load distance data
    let _ = file.dataset("distances").unwrap().read_2d::<f32>().unwrap();

    // load neighbours
    let _ = file.dataset("neighbors").unwrap().read_2d::<i32>().unwrap();

    // load test data
    let test_data: Vec<ZC<Embedding<784, f32>>> = file
        .dataset("test").unwrap()
        .read_2d::<f32>().unwrap()
        .rows()
        .into_iter()
        .map(|row| {
            let slice = row.as_slice().unwrap();
            let embedding: Embedding<784, f32> = slice.try_into().unwrap();
            let zc_embedding: ZC<Embedding<784, f32>> = embedding.into();
            zc_embedding
        })
        .collect();

    let mut distance = 0.;
    let start = ProcessTime::now();
    for query in &test_data {
        let handle = sw.search(query, search_span);
        let result = sw.blobs.get(handle).unwrap().unwrap();
        distance += MetricL2::distance(query, &result);
    }
    let avg_cpu_time = start.elapsed() / test_data.len() as u32;
    let avg_distance = distance / test_data.len() as f32;

    EvalResult{
        avg_cpu_time,
        avg_distance
    }
}

/// A Python module implemented in Rust.
pub fn bench(pm: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(pm.py(), "bench")?;
    m.add_class::<MNISTSW>()?;
    m.add_class::<StepResult>()?;
    m.add_class::<EvalResult>()?;
    m.add_function(wrap_pyfunction!(setup, &m)?)?;
    m.add_function(wrap_pyfunction!(step, &m)?)?;
    m.add_function(wrap_pyfunction!(eval, &m)?)?;
    pm.add_submodule(&m)?;
    Ok(())
}

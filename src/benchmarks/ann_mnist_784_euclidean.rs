use cpu_time::ProcessTime;
use std::{io::Write, time::{Duration, SystemTime}};

// search in serial mode i7-core @2.7Ghz for 10 fist neighbours
//  max_nb_conn   ef_cons    ef_search   scale_factor    extend  keep pruned  recall        req/s      last ratio
//
//     12           400         12           1              0          0        0.917        6486       1.005
//     24           400         24           1              1          0        0.9779       3456       1.001

// parallel mode 4 i7-core @2.7Ghz
//  max_nb_conn   ef_cons    ef_search   scale_factor    extend  keep pruned  recall        req/s      last ratio
//     24           400         24           1              0          0        0.977        12566       1.001
//     24           400         12           1              0          0        0.947        18425       1.003

// 8 hyperthreaded i7-core @ 2.3 Ghz
//     24           400         24           1              0          0        0.977        22197        1.001

// 24 core Core(TM) i9-13900HX simdeez
//     24           400         24           1              0          0        0.977        62000        1.001

use anndists::dist::*;
use hnsw_rs::prelude::*;

use super::utils::*;

pub fn run_hnsw(stdout: &mut impl Write, fname: String, parallel: bool) {
    writeln!(stdout,"test_load_hdf5 {:?}", fname).unwrap();
    // now recall that data are stored in row order.
    let anndata = annhdf5::AnnBenchmarkData::new(fname);
    let anndata = anndata.unwrap();
    let knbn_max = anndata.test_distances.dim().1;
    let nb_elem = anndata.train_data.len();
    writeln!(stdout,
        "Train size : {}, test size : {}",
        nb_elem,
        anndata.test_data.len()
    ).unwrap();
    writeln!(stdout,"Nb neighbours answers for test data : {}", knbn_max).unwrap();
    //
    let max_nb_connection = 24;
    let nb_layer = 16.min((nb_elem as f32).ln().trunc() as usize);
    let ef_c = 400;
    writeln!(stdout,
        " number of elements to insert {:?} , setting max nb layer to {:?} ef_construction {:?}",
        nb_elem, nb_layer, ef_c
    ).unwrap();
    writeln!(stdout,
        " ====================================================================================="
    ).unwrap();
    let nb_search = anndata.test_data.len();
    writeln!(stdout," number of search {:?}", nb_search).unwrap();

    let mut hnsw = Hnsw::<f32, DistL2>::new(max_nb_connection, nb_elem, nb_layer, ef_c, DistL2 {});
    hnsw.set_extend_candidates(false);

    // parallel insertion
    let mut start = ProcessTime::now();

    let mut now = SystemTime::now();
    let data_for_par_insertion = anndata
        .train_data
        .iter()
        .enumerate()
        .map(|(i, x)| (x.as_slice(), i))
        .collect();

    if parallel {
        writeln!(stdout," \n parallel insertion").unwrap();
        hnsw.parallel_insert_slice(&data_for_par_insertion);
    } else {
        writeln!(stdout," \n serial insertion").unwrap();
        for d in data_for_par_insertion {
            hnsw.insert_slice(d);
        }
    }
    let mut cpu_time: Duration = start.elapsed();
    //
    writeln!(stdout,
        "\n hnsw data insertion cpu time  {:?}  system time {:?} ",
        cpu_time,
        now.elapsed()
    ).unwrap();

    hnsw.dump_layer_info();
    writeln!(stdout," hnsw data nb point inserted {:?}", hnsw.get_nb_point()).unwrap();
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
    writeln!(stdout,"\n searching with ef : {:?}", ef_c).unwrap();
    start = ProcessTime::now();
    now = SystemTime::now();
    // search
    if parallel {
        writeln!(stdout," \n parallel search").unwrap();
        knn_neighbours_for_tests = hnsw.parallel_search(&anndata.test_data, knbn, ef_c);
    } else {
        writeln!(stdout," \n serial search").unwrap();
        for i in 0..anndata.test_data.len() {
            let knn_neighbours: Vec<Neighbour> = hnsw.search(&anndata.test_data[i], knbn, ef_c);
            knn_neighbours_for_tests.push(knn_neighbours);
        }
    }
    cpu_time = start.elapsed();
    let search_sys_time = now.elapsed().unwrap().as_micros() as f32;
    let search_cpu_time = cpu_time.as_micros() as f32;
    writeln!(stdout,
        "total cpu time for search requests {:?} , system time {:?} ",
        search_cpu_time, search_sys_time
    ).unwrap();
    // now compute recall rate
    for i in 0..anndata.test_data.len() {
        let true_distances = anndata.test_distances.row(i);
        let max_dist = true_distances[knbn - 1];
        let mut _knn_neighbours_id: Vec<usize> =
            knn_neighbours_for_tests[i].iter().map(|p| p.d_id).collect();
        let knn_neighbours_dist: Vec<f32> = knn_neighbours_for_tests[i]
            .iter()
            .map(|p| p.distance)
            .collect();
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
    let mean_fraction_returned = (nb_returned.iter().sum::<usize>() as f32) / ((nb_returned.len() * knbn) as f32);
    writeln!(stdout,"mean fraction returned by search {:?} ", mean_fraction_returned).unwrap();

    let last_distances_ratio = last_distances_ratio.iter().sum::<f32>() / last_distances_ratio.len() as f32;
    writeln!(stdout, "last distances ratio {:?}", last_distances_ratio).unwrap();

    let mean_recall = (recalls.iter().sum::<usize>() as f32) / ((knbn * recalls.len()) as f32);
    writeln!(stdout, "recall rate: {:?}", mean_recall).unwrap();

    let reqs = (nb_search as f32) * 1.0e+6_f32 / search_sys_time;
    writeln!(stdout, "req/s {:?}", reqs).unwrap();
}

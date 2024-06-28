//! This file provides hdf5 utilities to load ann-benchmarks hdf5 data files
//! As the libray does not depend on hdf5 nor on ndarray, it is nearly the same for both
//! ann benchmarks.  

use ndarray::Array2;

use hdf5::*;

// datasets
//   . distances (nbojects, dim)   f32 matrix    for tests objects
//   . neighbors (nbobjects, nbnearest) int32 matrix giving the num of nearest neighbors in train data
//   . test      (nbobjects, dim)   f32 matrix  test data
//   . train     (nbobjects, dim)   f32 matrix  train data

/// a structure to load  hdf5 data file benchmarks from https://github.com/erikbern/ann-benchmarks
pub struct AnnBenchmarkData {
    /// dataset source
    pub fname: String,
    /// distances from each test object to its nearest neighbours.
    pub test_distances: Array2<f32>,
    // for each test data , id of its nearest neighbours
    pub test_neighbours: Array2<i32>,
    /// list of vectors for which we will search ann.
    pub test_data: Vec<Vec<f32>>,
    /// list of data vectors
    pub train_data: Vec<Vec<f32>>,
}

impl AnnBenchmarkData {
    pub fn new(fname: String) -> Result<AnnBenchmarkData> {
        let file = hdf5::File::open(&fname)?;

        // load distance data
        let test_distances = file.dataset("distances")?
            .read_2d::<f32>()?;

        // load neighbours
        let test_neighbours = file.dataset("neighbors")?
            .read_2d::<i32>()?;

        // load test data
        let test_data: Vec<_> = file.dataset("test")?
            .read_2d::<f32>()?
            .rows().into_iter()
            .map(|row| row.to_vec())
            .collect();

        // load train data
        let train_data: Vec<_> = file.dataset("train")?
            .read_2d::<f32>()?
            .rows().into_iter()
            .map(|row| row.to_vec())
            .collect();

        Ok(AnnBenchmarkData {
            fname: fname.clone(),
            test_distances,
            test_neighbours,
            test_data,
            train_data
        })
    } // end new

    /// do l2 normalisation of test and train vector to use DistDot metrinc instead DistCosine to spare cpu
    #[allow(unused)]
    pub fn do_l2_normalization(&mut self) {
        self.test_data.iter_mut().for_each(|i| anndists::dist::l2_normalize(i));
        self.train_data.iter_mut().for_each(|i| anndists::dist::l2_normalize(i));
    } // end of do_l2_normalization
} // end of impl block

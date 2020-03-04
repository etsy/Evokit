//! Dataset
//! ---
//!
//! Dataset defines static and minibatch datasets
use hashbrown;
use rand;
use rand_xorshift;
use rayon;

use self::rayon::iter::{IndexedParallelIterator, IterBridge, ParallelBridge};
use self::rayon::prelude::*;

use std::ops::DerefMut;
use std::sync::Arc;

use super::dataset::types::{DataParse, Metadata};
use crate::load::parse_line;
use std::fs::File;
use std::io::{BufRead, BufReader};

use self::hashbrown::HashMap;
use self::rand::distributions::{Distribution, Uniform};
use self::rand::SeedableRng;
use self::rand_xorshift::XorShiftRng;
use std::fmt::Debug;

/// Defines sparse and dense vectors
pub mod types;

/// Specifies one request. A set of documents and the index in the full set of requests (for enumeration)
pub type DatasetItem<A> = (usize, Arc<Grouping<A>>);
/// Eager iterator that loads the entire dataset
pub type EagerIterator<A> =
    self::rayon::iter::Enumerate<self::rayon::vec::IntoIter<Arc<Grouping<A>>>>;

#[derive(Debug)]
/// A set of documents
pub struct Grouping<A> {
    /// X vectors for all the documents
    pub x: Vec<A>,
    /// Labels for all the documents
    pub y: Arc<Vec<f32>>,
    /// Metadata for all the documents
    pub metadata: Vec<Metadata>,
}

impl<A: Debug + Clone> Grouping<A> {
    /// Creates a new Grouping with no metadata
    pub fn new(x: Vec<A>, y: Vec<f32>) -> Self {
        let len = x.len();
        Grouping {
            x: x,
            y: Arc::new(y),
            metadata: vec![HashMap::with_capacity(0); len],
        }
    }
    /// Creates a new Grouping with metadata
    pub fn new_with_md(x: Vec<A>, y: Vec<f32>, md: Vec<Metadata>) -> Self {
        Grouping {
            x: x,
            y: Arc::new(y),
            metadata: md,
        }
    }
    /// Returns the number of documents in this set
    pub fn len(&self) -> usize {
        self.x.len()
    }
}

/// Trait defining methods for getting the current dataset
pub trait Dataset<A: Sized + Send + Sync> {
    /// Type of iterator for accessing the data
    type Iter: ParallelIterator<Item = DatasetItem<A>> + Sized;

    /// Shuffles the data if needed and returns whether a shuffle happened
    fn shuffle(&mut self) -> bool;

    /// Gets the current dataset
    fn data(&self) -> Self::Iter;

    /// Gets all the data
    fn all(&self) -> Self::Iter;
}

/// Dataset for training on the full data
pub struct StaticDataset<A> {
    /// List of requests
    qs: Vec<Arc<Grouping<A>>>,
}

impl<A> StaticDataset<A> {
    /// Returns a new StaticDataset
    pub fn new(qs: Vec<Grouping<A>>) -> Self {
        StaticDataset {
            qs: qs.into_iter().map(Arc::new).collect(),
        }
    }
}

impl<A: Sized + Debug + Send + Clone + Sync> Dataset<A> for StaticDataset<A> {
    /// This is an Eager Iterator as we have loaded the data already
    type Iter = EagerIterator<A>;

    /// This does not shuffle
    fn shuffle(&mut self) -> bool {
        false
    }

    /// Returns all the data
    fn data(&self) -> Self::Iter {
        let qs: Vec<_> = self.qs.par_iter().map(|x| x.clone()).collect();
        qs.into_par_iter().enumerate()
    }

    /// Returns all the data
    fn all(&self) -> Self::Iter {
        let qs: Vec<_> = self.qs.par_iter().map(|x| x.clone()).collect();
        qs.into_par_iter().enumerate()
    }
}

/// Dataset for only loading data as needed from disk
pub struct LazyDataset<P> {
    /// Method for parsing the file
    fmt: Box<P>,
    /// File to read
    fname: String,
}

/// Iterator for loading a section of a file at a time
pub struct LazyIterator<A, P> {
    /// Method for parsing the file
    fmt: Box<P>,
    /// Buffer reader for the file pointing to a particular position
    buff_reader: BufReader<File>,
    /// Buffer to read into
    buffer: String,
    /// last line read. This will be the start of the new Grouping
    remainder_line: Option<(f32, String, Metadata, A)>,
    /// Curent index for enumerating
    current_index: usize,
    /// Estimation of the number of docs per request
    capacity_estimate: usize,
}

impl<A: Debug + Send + Clone, P: DataParse<Out = A> + Clone> LazyDataset<P> {
    /// Creates a new LazyDataset
    pub fn new(fmt: P, fname: &str) -> Self {
        LazyDataset {
            fmt: Box::new(fmt),
            fname: fname.to_string(),
        }
    }

    /// Initializes the Lazy iterator
    pub fn get_iterator(&self) -> LazyIterator<A, P> {
        if let Ok(f) = File::open(&self.fname) {
            let br = BufReader::new(f);
            let buffer = String::new();
            LazyIterator {
                fmt: self.fmt.clone(),
                buff_reader: br,
                buffer: buffer,
                remainder_line: None,
                current_index: 0,
                capacity_estimate: 0,
            }
        } else {
            panic!("Was not able to open the dataset.");
        }
    }
}

impl<A: Debug + Send + Clone + Sync, P: DataParse<Out = A> + Clone> Iterator
    for LazyIterator<A, P>
{
    type Item = DatasetItem<A>;

    /// Parses the next request. This involves reading multiple lines
    fn next(&mut self) -> Option<Self::Item> {
        let mut x: Vec<A> = Vec::with_capacity(self.capacity_estimate);
        let mut y: Vec<f32> = Vec::with_capacity(self.capacity_estimate);
        let mut md: Vec<Metadata> = Vec::with_capacity(self.capacity_estimate);

        let remainder_line: Option<(f32, String, Metadata, A)> =
            std::mem::replace(&mut self.remainder_line, None);
        let mut cur_qid = match remainder_line {
            Some((score, previous_qid, mdi, v)) => {
                y.push(score);
                x.push(v);
                md.push(mdi);
                previous_qid
            }
            None => "".into(),
        };

        loop {
            self.buffer.clear();
            match self.buff_reader.read_line(&mut self.buffer) {
                Ok(buffer_size) if buffer_size > 0 => {
                    let res = parse_line(&*self.fmt, &self.buffer);
                    if let Some((score, qid, mdi, v)) = res {
                        if cur_qid.is_empty() {
                            y.push(score);
                            x.push(v);
                            md.push(mdi);
                            cur_qid.push_str(&qid);
                        } else if qid == cur_qid {
                            y.push(score);
                            x.push(v);
                            md.push(mdi);
                        } else {
                            self.remainder_line = Some((score, qid, mdi, v));
                            break;
                        }
                    } else {
                        break;
                    }
                }
                // end of the file or had a failure while parsing a line
                _ => break,
            }
            if self.remainder_line.is_some() {
                break;
            }
        }
        if y.len() > 0 {
            self.capacity_estimate = std::cmp::max(x.len(), self.capacity_estimate);
            x.shrink_to_fit();
            y.shrink_to_fit();
            md.shrink_to_fit();
            // get an index here to mimic enumerating
            let index = self.current_index;
            self.current_index += 1;
            Some((index, Arc::new(Grouping::new_with_md(x, y, md))))
        } else {
            None
        }
    }
}

impl<A: Debug + Send + Clone + Sync, P: DataParse<Out = A> + Clone + Send + Sync> Dataset<A>
    for LazyDataset<P>
{
    type Iter = IterBridge<LazyIterator<A, P>>;

    fn shuffle(&mut self) -> bool {
        false
    }

    fn data(&self) -> Self::Iter {
        self.get_iterator().par_bridge()
    }

    fn all(&self) -> Self::Iter {
        self.get_iterator().par_bridge()
    }
}

/// Dataset that selects a sample at a time for minibatch training
pub struct MinibatchDataset<A> {
    /// The full dataset
    full_set: Vec<Arc<Grouping<A>>>,
    /// The current minibatch
    batch: Vec<Arc<Grouping<A>>>,
    /// Whether to freeze the batch
    freeze: u32,
    /// Number of shuffles
    counter: u32,
    /// RNG for shuffling
    rng: XorShiftRng,
    /// Mini-batch percentage
    p: f32,
    /// Uniform distribution to sample from
    uniform: Uniform<f32>,
}

impl<A> MinibatchDataset<A> {
    /// New MinibatchDataset
    pub fn new(qs: Vec<Grouping<A>>, p: f32, freeze: u32, seed: u32) -> Self {
        MinibatchDataset {
            full_set: qs.into_iter().map(Arc::new).collect(),
            batch: Vec::new(),
            freeze: freeze,
            counter: 0,
            p: p,
            uniform: Uniform::new(0.0, 1.0),
            rng: XorShiftRng::seed_from_u64(seed as u64),
        }
    }
}

impl<A: Send + Sync> Dataset<A> for MinibatchDataset<A> {
    type Iter = EagerIterator<A>;

    fn shuffle(&mut self) -> bool {
        let updated = if self.counter % self.freeze == 0 {
            // Build a new minibatch
            self.batch.clear();
            for f in self.full_set.iter() {
                if self.uniform.sample(&mut self.rng) < self.p {
                    self.batch.push(f.clone());
                }
            }
            true
        } else {
            false
        };
        self.counter = self.counter.wrapping_add(1);
        updated
    }

    fn data(&self) -> Self::Iter {
        let batch: Vec<_> = self.batch.par_iter().map(|x| x.clone()).collect();
        batch.into_par_iter().enumerate()
    }

    fn all(&self) -> Self::Iter {
        let full_set: Vec<_> = self.full_set.par_iter().map(|x| x.clone()).collect();
        full_set.into_par_iter().enumerate()
    }
}

impl<A: Sync + Send> Dataset<A> for Box<dyn Dataset<A, Iter = EagerIterator<A>>> {
    type Iter = EagerIterator<A>;
    fn shuffle(&mut self) -> bool {
        self.deref_mut().shuffle()
    }

    fn data(&self) -> Self::Iter {
        self.as_ref().data()
    }

    fn all(&self) -> Self::Iter {
        self.as_ref().all()
    }
}

impl<A: Sync + Send> Dataset<A> for Box<dyn Dataset<A, Iter = EagerIterator<A>> + Send + Sync> {
    type Iter = EagerIterator<A>;
    fn shuffle(&mut self) -> bool {
        self.deref_mut().shuffle()
    }

    fn data(&self) -> Self::Iter {
        self.as_ref().data()
    }

    fn all(&self) -> Self::Iter {
        self.as_ref().all()
    }
}

/// Empty dataset for testing
pub struct EmptyDataset<A> {
    empty: Vec<Arc<Grouping<A>>>,
}

impl<A> EmptyDataset<A> {
    /// Create a new EmptyDataset
    pub fn new() -> Self {
        EmptyDataset {
            empty: Vec::with_capacity(0),
        }
    }
}

impl<A: Sync + Send> Dataset<A> for EmptyDataset<A> {
    type Iter = EagerIterator<A>;
    fn shuffle(&mut self) -> bool {
        false
    }

    fn data(&self) -> Self::Iter {
        self.empty.clone().into_par_iter().enumerate()
    }

    fn all(&self) -> Self::Iter {
        self.empty.clone().into_par_iter().enumerate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::types::SparseData;
    use std::collections::HashSet;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_lazy() {
        let test_data = vec![
            "1 qid:0 meta1:2 meta2:false 1:1.0 2:2.0",
            "1 qid:0 meta1:2 meta2:false 1:1.1 2:2.1",
            "1 qid:0 meta1:2 meta2:false 1:1.2 2:2.2",
            "1 qid:1 meta1:2 meta2:false 1:3.0 2:4.0",
            "1 qid:1 meta1:2 meta2:false 1:3.1 2:4.1",
        ];
        if let Ok(dir) = tempdir() {
            let data_path = dir.path().join("test_data");
            if let Ok(mut file) = File::create(data_path) {
                for line in test_data {
                    writeln!(file, "{}", line).expect("Must be able to write to file in test");
                }
            }
            let parser = SparseData(2);
            let dup_path = dir.path().join("test_data");
            let str_path = dup_path.to_str().unwrap();
            let dataset = LazyDataset::new(parser, str_path);
            {
                let data_iter = dataset.data();
                assert_eq!(data_iter.count(), 2);
            }
            {
                let data_iter = dataset.data();
                let num_rows_per_query: HashSet<_> =
                    data_iter.map(|(_idx, rs)| rs.x.len()).collect();
                let expected: HashSet<_> = vec![3, 2].into_iter().collect();
                assert_eq!(num_rows_per_query, expected);
            }
            {
                let data_iter = dataset.data();
                let indices: HashSet<_> = data_iter.map(|(idx, _rs)| idx).collect();
                let expected: HashSet<_> = vec![0, 1].into_iter().collect();
                assert_eq!(indices, expected);
            }
        }
    }
}

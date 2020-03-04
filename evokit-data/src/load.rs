//! Load
//! ---
//!
//! This defines the methods to read the libsvm data
use rayon;

use self::rayon::prelude::*;

use std::fmt::Debug;
use std::fs::File;
use std::io::{BufRead, BufReader, Error};

use super::dataset::types::DataParse;
use super::dataset::types::Metadata;
use super::dataset::Grouping;

/// Given a line from a libsvm file, outputs the label, metadata and vector
pub fn parse_line<'b, F: DataParse>(
    f: &'b F,
    line: &str,
) -> Option<(f32, String, Metadata, F::Out)> {
    // Remove comments
    let line = line.split('#').next().unwrap();
    let mut pieces = line.trim().split_whitespace();
    let score = pieces.next().and_then(|x| x.parse().ok());

    let qid = pieces.next();
    let data = f.parse(pieces);

    match (score, qid, data) {
        (Some(s), Some(q), Some((md, vals))) => Some((s, q.into(), md, vals)),
        _ => None,
    }
}

/// Hard code buffersize for now
static BUFFER_SIZE: usize = 1000;
/// Given a file path, loads it into a dataset
pub fn read_libsvm<D: Debug + Send + Clone, F: DataParse<Out = D>>(
    fmt: &F,
    fname: &str,
) -> Result<Vec<Grouping<F::Out>>, Error> {
    let f = File::open(fname)?;
    let br = BufReader::new(f);
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut md = Vec::new();
    let mut query_sets = Vec::new();
    let mut cur_qid: String = "".into();
    let mut buffer = Vec::with_capacity(BUFFER_SIZE);
    let mut tmp_results = Vec::with_capacity(BUFFER_SIZE);
    let mut it = br.lines();
    loop {
        buffer.clear();
        for _ in 0..BUFFER_SIZE {
            if let Some(line) = it.next() {
                let l = line?;
                buffer.push(l.clone());
            }
        }
        if buffer.len() == 0 {
            break;
        }

        tmp_results.clear();
        // Parse in parallel
        buffer
            .par_iter()
            .map(|l| parse_line(fmt, &l))
            .collect_into_vec(&mut tmp_results);

        for res in tmp_results.drain(..) {
            if let Some((score, qid, mdi, v)) = res {
                if qid == cur_qid {
                    y.push(score);
                    x.push(v);
                    md.push(mdi);
                } else {
                    if y.len() > 0 {
                        let len = x.len();
                        x.shrink_to_fit();
                        y.shrink_to_fit();
                        md.shrink_to_fit();
                        query_sets.push(Grouping::new_with_md(x, y, md));
                        x = Vec::with_capacity(len);
                        y = Vec::with_capacity(len);
                        md = Vec::with_capacity(len);
                    }
                    x.push(v);
                    y.push(score);
                    md.push(mdi);
                    cur_qid.clear();
                    cur_qid.push_str(&qid);
                }
            }
        }
    }
    if x.len() > 0 {
        x.shrink_to_fit();
        y.shrink_to_fit();
        md.shrink_to_fit();
        query_sets.push(Grouping::new_with_md(x, y, md));
    }
    query_sets.shrink_to_fit();
    Ok(query_sets)
}

extern crate es_core;
extern crate es_data;
extern crate rayon;

use self::es_data::dataset::types::{MetaType, Metadata};
use self::es_data::dataset::{DatasetItem, Grouping};
use self::rayon::prelude::*;

use std::borrow::Borrow;
use std::cmp::Ordering::Equal;
use std::collections::hash_map::DefaultHasher;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

use std::convert::TryInto;

use self::es_core::model::Evaluator;
use metrics::ndcg;

/// Evaluates a model by computing NDCG for the vecset. This is only for ES-Rank.
pub fn eval_rs<
    D: Debug + Send + Sync,
    M: Evaluator<D, f32>,
    I: ParallelIterator<Item = DatasetItem<D>>,
>(
    rs: I,
    state: &M,
    k: Option<usize>,
) -> f32 {
    let scorer = RankScorer::new(k);
    let (num_examples, cost) = rs
        .map(|(_idx, x)| (1., scorer.compute_ndcg(state, x.borrow())))
        .reduce(|| (0., 0.), |a, b| (a.0 + b.0, a.1 + b.1));
    cost / num_examples
}

/// Scorer for ES-Rank. Only computes NDCG.
pub struct RankScorer {
    /// K value for NDCG
    k: Option<usize>,
}

impl RankScorer {
    /// Returns new RankScorer
    pub fn new(k: Option<usize>) -> Self {
        RankScorer { k: k }
    }

    /// Orders the documents and computes NDCG
    pub fn compute_ndcg<D: Debug + Send + Sync, M: Evaluator<D, f32>>(
        &self,
        state: &M,
        rs: &Grouping<D>,
    ) -> f32 {
        let mut v: Vec<(f32, f32)> = (0..rs.x.len())
            .map(|i| {
                let yi = rs.y[i];
                let yi_hat = state.evaluate(&rs.x[i]);
                (yi_hat, yi)
            })
            .collect();

        // Sort descending by y_hat
        v.sort_by(|x, y| (y.0).partial_cmp(&x.0).unwrap_or(Equal));

        let mut scores: Vec<f32> = v.into_iter().map(|x| x.1).collect();

        ndcg(&mut scores, self.k) as f32
    }
}

/// Converts a MetaType to a number. If it's already a number, return as is. If it's a String, hash it.
pub fn metatype_to_num(data: &MetaType) -> f32 {
    match data {
        MetaType::Num(f) => *f,
        MetaType::Str(ref s) => {
            let mut hasher = DefaultHasher::new();
            s.hash(&mut hasher);
            hasher.finish() as f32
        }
    }
}

/// Extracts the float value from an item's Metadata
pub fn get_float_field(metadata: &Metadata, field_name: &String) -> f32 {
    if let Some(field) = metadata.get(field_name) {
        field.try_into().unwrap()
    } else {
        panic!(format!(
            "field_name '{}' is not defined in data",
            field_name
        ))
    }
}

/// Extracts the string value from an item's Metadata
pub fn get_string_field(metadata: &Metadata, field_name: &String, cast_to_string: bool) -> String {
    match metadata.get(field_name).expect(&format!(
        "field_name '{}' is not defined in data",
        field_name
    )) {
        MetaType::Str(s) => s.clone(),
        MetaType::Num(num) if cast_to_string => num.to_string(),
        _ => panic!(format!("Metatype is not string: {}", field_name)),
    }
}

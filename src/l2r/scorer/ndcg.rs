extern crate es_core;
extern crate es_data;

use self::es_core::optimizer::ScoreLogger;
use self::es_data::dataset::types::Metadata;

use crate::l2r::scorer::parameters::*;
use crate::l2r::scorer::utils::*;
use crate::metrics::ndcg;

#[derive(Clone, Debug)]
/// Scorer to compute NDCG using any field as the label
pub struct NdcgScorer {
    /// K value. If none is specified, all the docs are used
    pub k: Option<usize>,
    /// Field to use for the gain value
    pub field_name: Option<String>,
}

impl NdcgScorer {
    /// Creates a WeightedNdcgScorer
    pub fn new(parameters: &AtKScoringParameters) -> Self {
        NdcgScorer {
            k: parameters.k,
            field_name: Some(parameters.field_name.clone()),
        }
    }

    /// Returns a new WeightedNdcgScorer that uses the default relevance
    pub fn new_with_default_relevance(parameters: &NDCGScoringParameters) -> Self {
        NdcgScorer {
            k: parameters.k,
            field_name: None,
        }
    }
}

impl Scorer for NdcgScorer {
    /// Computes NDCG using the custom gain
    fn score(&self, scores: &[(ScoredInstance, &Metadata)]) -> (f32, Option<ScoreLogger>) {
        let mut rel: Vec<f32> = get_relevance_scores(scores, &self.field_name);
        let score = ndcg(&mut rel, self.k) as f32;
        (score, None)
    }
}

extern crate es_core;
extern crate es_data;

use self::es_core::optimizer::ScoreLogger;
use self::es_data::dataset::types::Metadata;

use crate::l2r::scorer::parameters::*;
use crate::l2r::scorer::utils::*;
use crate::metrics::get_err;

#[derive(Clone, Debug)]
/// Scorer for Err
pub struct ErrScorer {
    /// K value. If none is specified, all the docs are used
    pub k: Option<usize>,
    /// Field to use for the gain value
    pub field_name: String,
}

impl ErrScorer {
    /// Returns an ErrScorer
    pub fn new(parameters: &AtKScoringParameters) -> Self {
        ErrScorer {
            k: parameters.k,
            field_name: parameters.field_name.clone(),
        }
    }
}

impl Scorer for ErrScorer {
    /// Computes ERR
    fn score(&self, scores: &[(ScoredInstance, &Metadata)]) -> (f32, Option<ScoreLogger>) {
        let rel = get_float_fields(scores, &self.field_name);
        let score = get_err(&rel, self.k);
        (score, None)
    }
}

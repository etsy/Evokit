//! Scorer
//! ---
//!
//! This module contains the query level scorers. These are computed over a ranked list of documents
//! outputted by a policy. This are mainly ranking metrics.
//!
//! Types of query level scorers:
//! - NDCG: compute NDCG using the label as the gain
//! - WeightedNDCG: same as NDCG, except you can use any metadata field as the gain
//! - ERR:
//! - ReciprocalRank: useful for computing MRR
//! - Mean:
//! - Binary:
//! - DiscreteERRIA:
//! - FieldExtractor:
//! - Threshold:
//! - Min: minimum value.
//! - Max: maximum value
//! - Total: total value
//! - Distinct: number of distinct values
//! - Recall:
//!
//! You'll add them to the scoring config.
//!
//! ```json
//!    {
//!        "name": "avg-price",
//!        "group": "avg-price",
//!        "weight": 1.0,
//!        "scorer": {
//!            "Mean": {
//!                "k": 10,
//!                "field_name": "price"
//!            }
//!        }
//!    },
//! ```
//!
//! How to add new scorers:
//! - Define the struct for the new parameters in parameters.rs or use an existing parameter type
//! - add a new enum to ScoringParameters in mod.rs
//! - define the actual scorer struct and implement Scorer for that struct in a new file
//! - add a new statement to the match in From<&ScoringParameters>
extern crate es_core;
extern crate es_data;

use std::convert::From;

/// Definition of scorer and util methods
pub mod utils;

/// Parameters for the different scoring configs
pub mod parameters;

/// Scorer for NDCG
pub mod ndcg;

/// Scorer for grouped AUC
pub mod grouped_auc;

/// Scorer for recall
pub mod recall;

/// Scorer for distinct - diversity
pub mod distinct;

/// Scorer for total
pub mod total;

/// Scorer for max
pub mod max;

/// Scorer for min
pub mod min;

/// Scorer for mean
pub mod mean;

/// Scorer for reciprocal rank (to compute MRR)
pub mod reciprocal_rank;

/// Scorer to extract a field
pub mod field_extractor;

/// Scorer for err
pub mod err;

/// Scorer for discrete ERR-IA
pub mod discrete_erria;

/// Scorer for basic binary
pub mod basic_binary;

/// Scorer to apply a threshold
pub mod threshold;

use self::es_core::optimizer::ScoreLogger;
use self::es_data::dataset::types::Metadata;

use crate::l2r::scorer::basic_binary::BasicBinaryScorer;
use crate::l2r::scorer::discrete_erria::DiscreteERRIAScorer;
use crate::l2r::scorer::distinct::DistinctScorer;
use crate::l2r::scorer::err::ErrScorer;
use crate::l2r::scorer::field_extractor::FieldExtractor;
use crate::l2r::scorer::grouped_auc::GroupedAUCScorer;
use crate::l2r::scorer::max::MaxScorer;
use crate::l2r::scorer::mean::MeanScorer;
use crate::l2r::scorer::min::MinScorer;
use crate::l2r::scorer::ndcg::NdcgScorer;
use crate::l2r::scorer::parameters::*;
use crate::l2r::scorer::recall::RecallScorer;
use crate::l2r::scorer::reciprocal_rank::ReciprocalRankScorer;
use crate::l2r::scorer::threshold::ThresholdScorer;
use crate::l2r::scorer::total::TotalScorer;
use crate::l2r::scorer::utils::*;

#[derive(Deserialize, Debug)]
/// Parameters for the scorer
pub enum ScoringParameters {
    /// Config for NDCG
    NDCG(NDCGScoringParameters),
    /// Config for custom gain NDCG
    WeightedNDCG(AtKScoringParameters),
    /// Config for ERR
    ERR(AtKScoringParameters),
    /// Config for reciprocal rank
    ReciprocalRank(AtKScoringParameters),
    /// Config for mean
    Mean(AtKScoringParameters),
    /// Config for binary scorer
    Binary(BinaryScoringParameters),
    /// config for discrete-ERRIA
    DiscreteERRIA(DiscreteERRIAScoringParameters),
    /// config for field extractor
    FieldExtractor(FieldExtractorParameters),
    /// Config for threshold
    Threshold(ThresholdScoringParameters),
    /// Config for min
    Min(AtKScoringParameters),
    /// Config for max
    Max(AtKScoringParameters),
    /// Config for total
    Total(AtKScoringParameters),
    /// Config for distinct
    Distinct(AtKScoringParameters),
    /// Config for recall
    Recall(RecallParameters),
    /// Config for grouped AUC
    GroupedAUC(GroupedAUCScoringParameters),
}

impl From<&ScoringParameters> for Box<dyn Scorer> {
    /// Converts a scoring parameter to an actual scorer
    fn from(sc: &ScoringParameters) -> Box<dyn Scorer> {
        match sc {
            ScoringParameters::Binary(ref param) => Box::new(BasicBinaryScorer::new(&param)),
            ScoringParameters::DiscreteERRIA(ref param) => {
                Box::new(DiscreteERRIAScorer::new(&param))
            }
            ScoringParameters::Distinct(ref param) => Box::new(DistinctScorer::new(&param)),
            ScoringParameters::ERR(ref param) => Box::new(ErrScorer::new(&param)),
            ScoringParameters::FieldExtractor(ref param) => Box::new(FieldExtractor::new(&param)),
            ScoringParameters::GroupedAUC(ref param) => Box::new(GroupedAUCScorer::new(&param)),
            ScoringParameters::Max(ref param) => Box::new(MaxScorer::new(&param)),
            ScoringParameters::Mean(ref param) => Box::new(MeanScorer::new(&param)),
            ScoringParameters::Min(ref param) => Box::new(MinScorer::new(&param)),
            ScoringParameters::NDCG(ref param) => {
                Box::new(NdcgScorer::new_with_default_relevance(&param))
            }
            ScoringParameters::WeightedNDCG(ref param) => Box::new(NdcgScorer::new(&param)),
            ScoringParameters::Recall(ref param) => Box::new(RecallScorer::new(&param)),
            ScoringParameters::ReciprocalRank(ref param) => {
                Box::new(ReciprocalRankScorer::new(&param))
            }
            ScoringParameters::Threshold(ref param) => Box::new(ThresholdScorer::new(&param)),
            ScoringParameters::Total(ref param) => Box::new(TotalScorer::new(&param)),
        }
    }
}

#[derive(Debug)]
/// Scorer for combining the output of different scorers
pub struct WeightedAggregateScorer {
    scorers: Vec<(String, f32, Box<dyn Scorer>)>,
}

impl WeightedAggregateScorer {
    /// Initializes new WeightedAggregateScorer
    pub fn new() -> Self {
        WeightedAggregateScorer {
            scorers: Vec::new(),
        }
    }

    /// Adds another scorer along with the weight
    /// # Arguments
    ///
    /// * `name` Name of the scorer
    /// * `weight` weight for combining the scorers
    /// * `parameters` parameters for the scorer
    pub fn add_scorer(&mut self, name: String, weight: f32, parameters: &ScoringParameters) -> () {
        self.scorers.push((name, weight, parameters.into()));
    }

    /// Returns the number of scorers
    pub fn len(&self) -> usize {
        self.scorers.len()
    }
}

impl Scorer for WeightedAggregateScorer {
    /// Runs all the scorers and combines them using the predefined weights
    fn score(&self, scores: &[(ScoredInstance, &Metadata)]) -> (f32, Option<ScoreLogger>) {
        let mut score = 0f32;
        let mut v = Vec::new();
        let mut logger = ScoreLogger::new(None);
        let mut w = 0.;
        for (name, weight, scorer) in self.scorers.iter() {
            if *weight == 0.0 {
                continue;
            }
            v.truncate(0);
            v.extend_from_slice(scores);
            let (component, component_logger_opt) = scorer.score(&mut v);
            score += *weight * component;
            w += weight;

            logger.insert(name.clone(), component);
            match component_logger_opt {
                Some(component_logger) => {
                    for (key, val) in component_logger.iter() {
                        logger.insert(format!("{}-{}", name, key), *val);
                    }
                }
                None => (),
            };
        }
        (score / w, Some(logger))
    }
}

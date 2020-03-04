extern crate es_core;
extern crate es_data;

use self::es_core::optimizer::ScoreLogger;
use self::es_data::dataset::types::{MetaType, Metadata};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::l2r::scorer::parameters::*;
use crate::l2r::scorer::utils::*;
use crate::metrics::*;

#[derive(Debug, Clone)]
/// Scorer to compute Discrete-ERRIA
pub struct DiscreteERRIAScorer {
    /// Field containing the topic
    pub field_name: String,
    /// k value for ERR
    pub k: Option<usize>,
    /// Buckets to discretize the data by
    pub buckets: TopicBuckets,
    /// Default value when none exists
    pub default_topic: Option<u32>,
}

impl DiscreteERRIAScorer {
    /// Returns a new DiscreteERRIAScorer
    pub fn new(parameters: &DiscreteERRIAScoringParameters) -> DiscreteERRIAScorer {
        match parameters.buckets {
            TopicBuckets::StringBuckets(ref buckets) => {
                assert!(
                    (!buckets.is_empty() && parameters.default_topic.is_some())
                        || buckets.is_empty(),
                    "must provide default topic when providing buckets"
                );
            }
            TopicBuckets::NumericBuckets(ref buckets) => {
                assert!(
                    (!buckets.is_empty() && parameters.default_topic.is_some())
                        || buckets.is_empty(),
                    "must provide default topic when providing buckets"
                );
            }
        };
        DiscreteERRIAScorer {
            field_name: parameters.field_name.clone(),
            k: parameters.k,
            // TODO: confirm buckets are sorted and unique.
            buckets: parameters.buckets.clone(),
            default_topic: parameters.default_topic,
        }
    }

    /// Given the Metadata from an instance, this discretizes it into the appropriate bucket
    pub fn discretize(&self, data: Option<&MetaType>) -> u32 {
        match (data, &self.buckets) {
            (Some(MetaType::Num(val)), TopicBuckets::NumericBuckets(ref buckets)) => {
                (buckets
                    .iter()
                    .filter(|x| *x >= val)
                    .collect::<Vec<_>>()
                    .len() as u32)
            }
            (Some(MetaType::Str(val)), TopicBuckets::StringBuckets(ref buckets)) => {
                // if no buckets are provided we make each string its own bucket
                if buckets.is_empty() {
                    let mut hasher = DefaultHasher::new();
                    val.hash(&mut hasher);
                    hasher.finish() as u32
                } else if let Ok(index) = buckets.binary_search(val) {
                    (index as u32)
                } else {
                    self.default_topic
                        .expect("Default needs to be provided for remaining buckets")
                }
            }
            (None, _) => self
                .default_topic
                .expect("Default needs to be provided for remaining buckets"),
            _ => panic!("buckets do not match the type"),
        }
    }
}

impl Scorer for DiscreteERRIAScorer {
    /// Computes Discrete-ERRIA
    fn score(&self, scores: &[(ScoredInstance, &Metadata)]) -> (f32, Option<ScoreLogger>) {
        // TODO: should these weights be computed once for the entire data set?
        // If a candidate set doesn't contain a topic, the max score will be 0, and it won't affect the metric.
        let md: Vec<_> = scores.iter().map(|x| x.1).collect();
        let subtopics = get_subtopics(
            &md,
            &self.field_name,
            #[inline]
            |x| self.discretize(x),
        );
        let subtopic_weights = get_subtopic_weights(&subtopics);

        let rel: Vec<f32> = scores.iter().map(|x| x.0.label).collect();
        let score = get_err_ia(&rel, &subtopics, &subtopic_weights, self.k);
        (score, None)
    }
}

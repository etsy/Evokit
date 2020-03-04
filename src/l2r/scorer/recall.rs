extern crate es_core;
extern crate es_data;

use self::es_core::optimizer::ScoreLogger;
use self::es_data::dataset::types::{MetaType, Metadata};

use crate::l2r::scorer::parameters::*;
use crate::l2r::scorer::utils::*;

#[derive(Clone, Debug)]
/// Scorer to compute Recall
pub struct RecallScorer {
    /// Number of top docs to look at
    pub k: usize,
    /// Field to get the value for
    pub field_name: String,
    /// Expected value for positive example
    pub field_value: MetaType,
}

impl RecallScorer {
    /// New RecallScorer
    pub fn new(parameters: &RecallParameters) -> Self {
        RecallScorer {
            k: parameters.k,
            field_name: parameters.field_name.clone(),
            field_value: parameters.field_value.clone(),
        }
    }
}

impl Scorer for RecallScorer {
    /// Computes recall
    fn score(&self, scores: &[(ScoredInstance, &Metadata)]) -> (f32, Option<ScoreLogger>) {
        let score = if scores.len() <= self.k {
            1.0
        } else {
            let (num_in_class_at_k, num_in_class) = scores
                .iter()
                .map(|x| x.1.get(&self.field_name) == Some(&self.field_value))
                .enumerate()
                .fold(
                    (0usize, 0usize),
                    |(mut num_in_class_at_k, num_in_class), (idx, same_class)| {
                        if same_class {
                            num_in_class_at_k += if idx < self.k { 1 } else { 0 };
                            (num_in_class_at_k, num_in_class + 1)
                        } else {
                            (num_in_class_at_k, num_in_class)
                        }
                    },
                );
            if num_in_class == 0 {
                1.0
            } else {
                num_in_class_at_k as f32 / num_in_class as f32
            }
        };
        (score, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_scorer() {
        let metadata1: Metadata = [("sid".to_string(), MetaType::Str("s1".to_string()))]
            .iter()
            .cloned()
            .collect();
        let metadata2: Metadata = [("sid".to_string(), MetaType::Str("s2".to_string()))]
            .iter()
            .cloned()
            .collect();
        let si = ScoredInstance {
            label: 0.4,
            model_score: None,
        };
        let scores = [
            (si.clone(), &metadata1),
            (si.clone(), &metadata2),
            (si.clone(), &metadata1),
            (si.clone(), &metadata1),
        ];
        {
            let param = RecallParameters {
                field_name: "sid".to_string(),
                k: 1,
                field_value: MetaType::Str("s1".to_string()),
            };
            let scorer = RecallScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 0.33333333)
        }
        {
            let param = RecallParameters {
                field_name: "sid".to_string(),
                k: 2,
                field_value: MetaType::Str("s1".to_string()),
            };
            let scorer = RecallScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 0.33333333)
        }
        {
            let param = RecallParameters {
                field_name: "sid".to_string(),
                k: 3,
                field_value: MetaType::Str("s1".to_string()),
            };
            let scorer = RecallScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 0.6666667)
        }
        {
            let param = RecallParameters {
                field_name: "sid".to_string(),
                k: 4,
                field_value: MetaType::Str("s1".to_string()),
            };
            let scorer = RecallScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 1.0)
        }
    }
}

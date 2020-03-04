extern crate es_core;
extern crate es_data;
extern crate float_ord;

use self::float_ord::FloatOrd;

use self::es_core::optimizer::ScoreLogger;
use self::es_data::dataset::types::Metadata;

use crate::l2r::scorer::parameters::*;
use crate::l2r::scorer::utils::*;
use metrics::*;

#[derive(Clone, Debug)]
/// Scorer to compute the mean
pub struct MeanScorer {
    /// K value. Will only compute mean over top-K docs. Will use all if no value is provided
    pub k: Option<usize>,
    /// field containing the value to take an average of
    pub field_name: String,
    /// Whether to normalize the value between [0,1]
    pub normalize: bool,
}

impl MeanScorer {
    /// Returns a new MeanScorer
    pub fn new(parameters: &AtKScoringParameters) -> Self {
        MeanScorer {
            k: parameters.k,
            field_name: parameters.field_name.clone(),
            normalize: parameters.normalize.unwrap_or(false),
        }
    }
}

impl Scorer for MeanScorer {
    /// Computes the mean
    fn score(&self, scores: &[(ScoredInstance, &Metadata)]) -> (f32, Option<ScoreLogger>) {
        let mut field_values = get_float_fields(scores, &self.field_name);
        let score = get_mean(&field_values, self.k);

        // if normalize, get the max avg
        if self.normalize {
            // sorts in place
            field_values.sort_by_key(|v| FloatOrd(-*v));
            let max_score = get_mean(&field_values, self.k);
            if max_score == 0.0 {
                (1.0, None)
            } else {
                (score / max_score, None)
            }
        } else {
            (score, None)
        }
    }
}

#[cfg(test)]
mod tests {
    use self::es_data::dataset::types::MetaType;
    use super::*;

    #[test]
    fn test_mean_scorer() {
        let metadata1: Metadata = [("price".to_string(), MetaType::Num(5.0))]
            .iter()
            .cloned()
            .collect();
        let metadata2: Metadata = [("price".to_string(), MetaType::Num(4.0))]
            .iter()
            .cloned()
            .collect();
        let metadata3: Metadata = [("price".to_string(), MetaType::Num(10.0))]
            .iter()
            .cloned()
            .collect();
        let metadata4: Metadata = [("price".to_string(), MetaType::Num(15.0))]
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
            (si.clone(), &metadata3),
            (si.clone(), &metadata4),
        ];

        {
            let param = AtKScoringParameters {
                field_name: "price".to_string(),
                k: Some(3),
                normalize: None,
                opt_goal: None,
            };
            let scorer = MeanScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 6.333333333)
        }

        {
            let param = AtKScoringParameters {
                field_name: "price".to_string(),
                k: None,
                normalize: None,
                opt_goal: None,
            };
            let scorer = MeanScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 8.5)
        }

        {
            let param = AtKScoringParameters {
                field_name: "price".to_string(),
                k: Some(10),
                normalize: None,
                opt_goal: None,
            };
            let scorer = MeanScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 8.5)
        }

        {
            let param = AtKScoringParameters {
                field_name: "price".to_string(),
                k: Some(10),
                normalize: Some(true),
                opt_goal: None,
            };
            let scorer = MeanScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 1.0)
        }

        {
            let param = AtKScoringParameters {
                field_name: "price".to_string(),
                k: Some(2),
                normalize: Some(true),
                opt_goal: None,
            };
            let scorer = MeanScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 0.36)
        }
    }

    #[test]
    fn test_mean_scorer_large_value_normalize() {
        // previously we used i8 to sort. this test verifies that we don't cap them.
        let si1 = ScoredInstance {
            label: 0.4,
            model_score: None,
        };
        let si2 = ScoredInstance {
            label: 0.3,
            model_score: None,
        };
        let metadata1: Metadata = [("price".to_string(), MetaType::Num(1000.0))]
            .iter()
            .cloned()
            .collect();
        let metadata2: Metadata = [("price".to_string(), MetaType::Num(2000.0))]
            .iter()
            .cloned()
            .collect();
        {
            let scores = [(si1.clone(), &metadata1), (si2.clone(), &metadata2)];
            let param = AtKScoringParameters {
                field_name: "price".to_string(),
                k: Some(1),
                normalize: Some(true),
                opt_goal: None,
            };
            let scorer = MeanScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 0.5)
        }
        {
            let scores = [(si1.clone(), &metadata2), (si2.clone(), &metadata1)];
            let param = AtKScoringParameters {
                field_name: "price".to_string(),
                k: Some(1),
                normalize: Some(true),
                opt_goal: None,
            };
            let scorer = MeanScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 1.0)
        }
    }
}

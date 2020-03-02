extern crate es_core;
extern crate es_data;

use self::es_core::optimizer::ScoreLogger;
use self::es_data::dataset::types::Metadata;

use crate::l2r::scorer::parameters::*;
use crate::l2r::scorer::utils::*;

#[derive(Clone, Debug)]
/// Scorer to compute reciprocal rank
pub struct ReciprocalRankScorer {
    /// Field name for the label
    pub field_name: String,
}

impl ReciprocalRankScorer {
    /// Returns a new ReciprocalRankScorer
    pub fn new(parameters: &AtKScoringParameters) -> Self {
        ReciprocalRankScorer {
            field_name: parameters.field_name.clone(),
        }
    }
}

impl Scorer for ReciprocalRankScorer {
    /// Computes reciprocal rank
    fn score(&self, scores: &[(ScoredInstance, &Metadata)]) -> (f32, Option<ScoreLogger>) {
        let index_opt: Option<usize> = get_float_fields(scores, &self.field_name)
            .iter()
            .position(|x: &f32| *x > 0.0);

        let score = match index_opt {
            Some(index) => 1.0 / ((index + 1) as f32),
            None => 0.0,
        };
        (score, None)
    }
}

#[cfg(test)]
mod tests {
    use self::es_data::dataset::types::MetaType;
    use super::*;

    #[test]
    fn test_reciprocal_rank_scorer() {
        let param = AtKScoringParameters {
            field_name: "is_free_shipping".to_string(),
            k: None,
            normalize: None,
            opt_goal: None,
        };
        let scorer = ReciprocalRankScorer::new(&param);
        {
            let metadata: Metadata = [("is_free_shipping".to_string(), MetaType::Num(1.0))]
                .iter()
                .cloned()
                .collect();
            let scored_instance = ScoredInstance {
                label: 0.4,
                model_score: None,
            };
            let scores = [(scored_instance, &metadata)];
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 1.0)
        }

        // takes reciprocal of rank of first occurrence only
        {
            let metadata_valid: Metadata = [("is_free_shipping".to_string(), MetaType::Num(1.0))]
                .iter()
                .cloned()
                .collect();
            let metadata_invalid: Metadata = [("is_free_shipping".to_string(), MetaType::Num(0.0))]
                .iter()
                .cloned()
                .collect();
            let si1 = ScoredInstance {
                label: 0.4,
                model_score: None,
            };
            let si2 = ScoredInstance {
                label: 0.1,
                model_score: None,
            };
            let si3 = ScoredInstance {
                label: 0.3,
                model_score: None,
            };
            let scores = [
                (si1, &metadata_invalid),
                (si2, &metadata_valid),
                (si3, &metadata_valid),
            ];
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 0.5)
        }

        // Return 0 if no matches
        {
            let scores = [];
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 0.0)
        }
    }
}

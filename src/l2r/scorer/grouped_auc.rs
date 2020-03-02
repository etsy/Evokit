extern crate classifier_measures;
extern crate es_core;
extern crate es_data;

use self::classifier_measures::roc_auc;

use self::es_core::optimizer::ScoreLogger;
use self::es_data::dataset::types::Metadata;

use crate::l2r::scorer::parameters::*;
use crate::l2r::scorer::utils::*;
use crate::l2r::utils::get_float_field;

#[derive(Clone, Debug)]
/// Scorer to compute grouped AUC
pub struct GroupedAUCScorer {
    /// Field containing the label
    pub field_name: Option<String>,
}

impl GroupedAUCScorer {
    /// Returns a new GroupedAUCScorer
    pub fn new(parameters: &GroupedAUCScoringParameters) -> Self {
        GroupedAUCScorer {
            field_name: parameters.field_name.clone(),
        }
    }
}

impl Scorer for GroupedAUCScorer {
    /// Computes grouped-AUC over the ranked list of documents
    fn score(&self, scores: &[(ScoredInstance, &Metadata)]) -> (f32, Option<ScoreLogger>) {
        let score_result = roc_auc(scores, |x| {
            let y_true = match &self.field_name {
                Some(ref fname) => get_float_field(x.1, fname) > 0.,
                None => x.0.label > 0.,
            };
            let y_hat = x.0.model_score;
            (
                y_true,
                y_hat.expect("Must use a policy that outputs a score"),
            )
        });
        match score_result {
            Some(v) => (v, None),
            _ => (1.0, None),
        }
    }
}

#[cfg(test)]
mod tests {
    use self::es_data::dataset::types::MetaType;
    use super::*;

    #[test]
    fn test_grouped_auc_scorer() {
        {
            let si1 = ScoredInstance {
                label: 0.0,
                model_score: Some(0.5),
            };
            let si2 = ScoredInstance {
                label: 1.0,
                model_score: Some(0.2),
            };
            let si3 = ScoredInstance {
                label: 1.0,
                model_score: Some(0.1),
            };
            let metadata_neg: Metadata = [("label".to_string(), MetaType::Num(0.0))]
                .iter()
                .cloned()
                .collect();
            let metadata_pos: Metadata = [("label".to_string(), MetaType::Num(1.0))]
                .iter()
                .cloned()
                .collect();
            let scores = [
                (si1.clone(), &metadata_neg),
                (si2.clone(), &metadata_pos),
                (si3.clone(), &metadata_pos),
            ];
            let param = GroupedAUCScoringParameters {
                field_name: Some("label".to_string()),
            };
            let scorer = GroupedAUCScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 0.0)
        }
        {
            let si1 = ScoredInstance {
                label: 0.0,
                model_score: Some(0.5),
            };
            let si2 = ScoredInstance {
                label: 1.0,
                model_score: Some(0.2),
            };
            let si3 = ScoredInstance {
                label: 1.0,
                model_score: Some(0.8),
            };
            let metadata_neg: Metadata = [("label".to_string(), MetaType::Num(0.0))]
                .iter()
                .cloned()
                .collect();
            let metadata_pos: Metadata = [("label".to_string(), MetaType::Num(1.0))]
                .iter()
                .cloned()
                .collect();
            let scores = [
                (si3.clone(), &metadata_pos),
                (si1.clone(), &metadata_neg),
                (si2.clone(), &metadata_pos),
            ];
            let param = GroupedAUCScoringParameters {
                field_name: Some("label".to_string()),
            };
            let scorer = GroupedAUCScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 0.5)
        }
        {
            let si1 = ScoredInstance {
                label: 0.0,
                model_score: Some(0.1),
            };
            let si2 = ScoredInstance {
                label: 1.0,
                model_score: Some(0.2),
            };
            let si3 = ScoredInstance {
                label: 1.0,
                model_score: Some(0.8),
            };
            let metadata_neg: Metadata = [("label".to_string(), MetaType::Num(0.0))]
                .iter()
                .cloned()
                .collect();
            let metadata_pos: Metadata = [("label".to_string(), MetaType::Num(1.0))]
                .iter()
                .cloned()
                .collect();
            let scores = [
                (si3.clone(), &metadata_pos),
                (si2.clone(), &metadata_pos),
                (si1.clone(), &metadata_neg),
            ];
            let param = GroupedAUCScoringParameters {
                field_name: Some("label".to_string()),
            };
            let scorer = GroupedAUCScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 1.0)
        }
    }
}

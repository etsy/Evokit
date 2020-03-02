extern crate es_core;
extern crate es_data;
extern crate float_ord;

use self::float_ord::FloatOrd;

use self::es_core::optimizer::ScoreLogger;
use self::es_data::dataset::types::Metadata;

use crate::l2r::scorer::parameters::*;
use crate::l2r::scorer::utils::*;

#[derive(Clone, Debug)]
/// Scorer to get the max value
pub struct MaxScorer {
    /// Number of top docs to look at
    pub k: Option<usize>,
    /// Field to get the value for
    pub field_name: String,
    /// Type of optimization goal
    pub opt_goal: Option<OptimizationGoal>,
}

impl MaxScorer {
    /// Returns a MaxScorer
    pub fn new(parameters: &AtKScoringParameters) -> Self {
        MaxScorer {
            k: parameters.k,
            field_name: parameters.field_name.clone(),
            opt_goal: if parameters.normalize.unwrap_or(false) {
                Some(
                    parameters
                        .opt_goal
                        .clone()
                        .expect("Must provide optimization goal when normalizing"),
                )
            } else {
                parameters.opt_goal.clone()
            },
        }
    }
}

impl Scorer for MaxScorer {
    /// Gets the max value across the docs
    fn score(&self, scores: &[(ScoredInstance, &Metadata)]) -> (f32, Option<ScoreLogger>) {
        let field_values = get_float_fields(scores, &self.field_name);
        let k = self.k.unwrap_or(field_values.len()).min(field_values.len());
        let max_val_at_k = *field_values[..k]
            .iter()
            .max_by_key(|x| FloatOrd(**x))
            .expect("must have at least one document");
        let (min_val, max_val) = get_min_and_max(field_values);
        let score = normalize_optimization(self.opt_goal, min_val, max_val, max_val_at_k);
        (score, None)
    }
}

#[cfg(test)]
mod tests {
    use self::es_data::dataset::types::MetaType;
    use super::*;

    #[test]
    fn test_max_scorer() {
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
            let scorer = MaxScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 10.0)
        }
        {
            let param = AtKScoringParameters {
                field_name: "price".to_string(),
                k: None,
                normalize: None,
                opt_goal: None,
            };
            let scorer = MaxScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 15.0)
        }
        {
            let param = AtKScoringParameters {
                field_name: "price".to_string(),
                k: Some(1),
                normalize: None,
                opt_goal: None,
            };
            let scorer = MaxScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 5.0)
        }
    }

    #[test]
    fn test_max_scorer_normalize() {
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
                normalize: Some(true),
                opt_goal: Some(OptimizationGoal::Maximize),
            };
            let scorer = MaxScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 0.54545456);
        }
        {
            let param = AtKScoringParameters {
                field_name: "price".to_string(),
                k: Some(4),
                normalize: Some(true),
                opt_goal: Some(OptimizationGoal::Maximize),
            };
            let scorer = MaxScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 1.0);
        }
    }
}

extern crate counter;
extern crate es_core;
extern crate es_data;

use self::counter::Counter;

use self::es_core::optimizer::ScoreLogger;
use self::es_data::dataset::types::Metadata;

use crate::l2r::scorer::parameters::*;
use crate::l2r::scorer::utils::*;

#[derive(Clone, Debug)]
/// Scorer to get the number of distinct values
pub struct DistinctScorer {
    /// Number of top docs to look at
    pub k: Option<usize>,
    /// Field to get the value for
    pub field_name: String,
    /// Type of optimization goal
    pub opt_goal: Option<OptimizationGoal>,
}

impl DistinctScorer {
    /// Returns new DistinctScorer
    pub fn new(parameters: &AtKScoringParameters) -> Self {
        DistinctScorer {
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

impl Scorer for DistinctScorer {
    /// Computes number of distinct documents
    fn score(&self, scores: &[(ScoredInstance, &Metadata)]) -> (f32, Option<ScoreLogger>) {
        let field_values = get_string_fields(scores, &self.field_name, true);
        let k = self.k.unwrap_or(field_values.len()).min(field_values.len());
        // Get the number of unique classes in the top K items (for the current ordering)
        let num_unique_at_k = field_values[..k].iter().collect::<Counter<_>>().len();

        // Get min and max number of distinct values we could have across all possible orderings to
        // normalize the score [0, 1].
        // Count the number of times each class occurs. Determine the number of classes that would
        // be in the top K for the worst case ordering (all items of the largest class are in the
        // top positions, and so forth in decreasing order)
        let counter = field_values.iter().collect::<Counter<_>>();
        let mut min_val = 0;
        let mut num_seen = 0;
        for (_key, count) in counter.most_common_ordered() {
            min_val += 1;
            num_seen += count;
            if num_seen >= k {
                break;
            }
        }

        // Get the maximum number of classes we could have in the top K if we had the best ordering,
        // one of each possible class.
        let max_val = counter.len().min(k);
        let score = normalize_optimization(
            self.opt_goal,
            min_val as f32,
            max_val as f32,
            num_unique_at_k as f32,
        );
        (score, None)
    }
}

#[cfg(test)]
mod tests {
    use self::es_data::dataset::types::MetaType;
    use super::*;

    #[test]
    fn test_distinct_scorer() {
        let metadata1: Metadata = [("sid".to_string(), MetaType::Str("s1".to_string()))]
            .iter()
            .cloned()
            .collect();
        let metadata2: Metadata = [("sid".to_string(), MetaType::Str("s1".to_string()))]
            .iter()
            .cloned()
            .collect();
        let metadata3: Metadata = [("sid".to_string(), MetaType::Str("s2".to_string()))]
            .iter()
            .cloned()
            .collect();
        let metadata4: Metadata = [("sid".to_string(), MetaType::Str("s3".to_string()))]
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
                field_name: "sid".to_string(),
                k: Some(3),
                normalize: None,
                opt_goal: None,
            };
            let scorer = DistinctScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 2.0)
        }
        {
            let param = AtKScoringParameters {
                field_name: "sid".to_string(),
                k: None,
                normalize: None,
                opt_goal: None,
            };
            let scorer = DistinctScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 3.0)
        }
        {
            let param = AtKScoringParameters {
                field_name: "sid".to_string(),
                k: Some(1),
                normalize: None,
                opt_goal: None,
            };
            let scorer = DistinctScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 1.0)
        }
    }

    #[test]
    fn test_distinct_scorer_normalize() {
        let metadata1: Metadata = [("sid".to_string(), MetaType::Str("s1".to_string()))]
            .iter()
            .cloned()
            .collect();
        let metadata2: Metadata = [("sid".to_string(), MetaType::Str("s2".to_string()))]
            .iter()
            .cloned()
            .collect();
        let metadata3: Metadata = [("sid".to_string(), MetaType::Str("s3".to_string()))]
            .iter()
            .cloned()
            .collect();
        let si = ScoredInstance {
            label: 0.4,
            model_score: None,
        };
        let scores = [
            (si.clone(), &metadata1),
            (si.clone(), &metadata1),
            (si.clone(), &metadata2),
            (si.clone(), &metadata3),
        ];

        {
            let param = AtKScoringParameters {
                field_name: "sid".to_string(),
                k: Some(3),
                normalize: Some(true),
                opt_goal: Some(OptimizationGoal::Maximize),
            };
            let scorer = DistinctScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 0.0)
        }
        {
            let param = AtKScoringParameters {
                field_name: "sid".to_string(),
                k: None,
                normalize: Some(true),
                opt_goal: Some(OptimizationGoal::Maximize),
            };
            let scorer = DistinctScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 1.0)
        }
        {
            let param = AtKScoringParameters {
                field_name: "sid".to_string(),
                k: Some(1),
                normalize: Some(true),
                opt_goal: Some(OptimizationGoal::Maximize),
            };
            let scorer = DistinctScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 1.0)
        }
    }
}

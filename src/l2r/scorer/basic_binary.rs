extern crate es_core;
extern crate es_data;

use self::es_core::optimizer::ScoreLogger;
use self::es_data::dataset::types::{MetaType, Metadata};

use crate::l2r::scorer::parameters::*;
use crate::l2r::scorer::utils::*;

/// Scorer for basic binary indicator
/// Can be used to bury or boost based on a field
#[derive(Debug, Clone)]
pub struct BasicBinaryScorer {
    /// Field name containing the value to compare with
    pub field_name: String,
    /// Value to compare with
    pub field_value: MetaType,
    /// Only matters for numerical MetaTypes. Defaults to val > threshold
    pub flip_comparator: bool,
    /// 0-indexed goal to optimize for
    pub goal_index: f32,
}

impl BasicBinaryScorer {
    /// Returns new BasicBinaryScorer
    pub fn new(parameters: &BinaryScoringParameters) -> BasicBinaryScorer {
        BasicBinaryScorer {
            field_name: parameters.field_name.clone(),
            field_value: parameters.field_value.clone(),
            flip_comparator: parameters.flip_comparator,
            goal_index: parameters.goal_index,
        }
    }

    /// Gets the best average we could have with an optimal order
    fn get_optimal_average(&self, num_valid: usize, num_docs: usize, goal_index: f32) -> f32 {
        // if goal_index is beyond the number of docs for this query, it is never achievable.
        // The goal is beyond the number of docs. Closest we can do is moving everything to the bottom
        if goal_index >= (num_docs as f32) {
            let mut total = 0.;
            for index in (num_docs - num_valid)..num_docs {
                total += index as f32;
            }
            return total / (num_valid as f32);
        }
        // (num_docs_above + num_docs_below) >= num_valid since we grabbed the valid docs from that list
        let num_docs_above = goal_index;
        let num_docs_below = (num_docs as f32) - goal_index - 1.;

        // if one doc is at the goal, we can focus on the remaining docs.
        let num_docs_remaining = num_valid - 1;
        let half_num_remaining = (num_docs_remaining / 2) as f32;

        // Can perfectly center around the goal index
        if half_num_remaining <= num_docs_above && half_num_remaining <= num_docs_below {
            let average = if num_docs_remaining % 2 == 0 {
                goal_index
            } else {
                // have an extra doc. can be added above or below.
                // Place below for simplicity. This will not affect the optimal distance, but allows us to avoid negatives.
                let total = (num_docs_remaining as f32) * goal_index
                    + (goal_index + half_num_remaining + 1.);
                total / (num_valid as f32)
            };
            return average;
        }
        // Either above or below doesn't have enough room for half the docs.
        if half_num_remaining >= num_docs_above {
            // There aren't enough slots above, so include all.
            let mut total = 0.;
            for index in 0..num_valid {
                total += index as f32;
            }
            total / (num_valid as f32)
        } else {
            // There aren't enough slots below, so include all.
            let mut total = 0.;
            for index in (num_docs - num_valid)..num_docs {
                total += index as f32;
            }
            total / (num_valid as f32)
        }
    }
}

impl Scorer for BasicBinaryScorer {
    /// Computes the distance from the goal
    fn score(&self, scores: &[(ScoredInstance, &Metadata)]) -> (f32, Option<ScoreLogger>) {
        // computes the average index of the feature in the Metadata
        let valid_indices = scores
            .iter()
            .enumerate()
            .map(|(idx, score)| {
                if !score.1.contains_key(&self.field_name) {
                    None
                } else if validate_metatype(
                    &score.1[&self.field_name],
                    &self.field_value,
                    self.flip_comparator,
                ) {
                    Some(idx as f32)
                } else {
                    None
                }
            })
            .filter(|idx| idx.is_some())
            .map(|idx| idx.expect("index should not be None"))
            .collect::<Vec<f32>>();

        // This candidate set does not contain documents that match the expected topic.
        // TODO: should we penalize or ignore these cases?
        if valid_indices.is_empty() {
            return (1., None);
        }

        let index_sum: f32 = valid_indices.iter().map(|x| x).sum();
        let average = index_sum / (valid_indices.len() as f32);
        // computes the distance between that average and the provided index
        let dist = (self.goal_index - average).powf(2.);

        let optimal_average =
            self.get_optimal_average(valid_indices.len(), scores.len(), self.goal_index);
        let optimal_dist = (self.goal_index - optimal_average).powf(2.);

        // get farthest position
        let farthest_position = if ((scores.len() as f32) - self.goal_index - 1.) > self.goal_index
        {
            (scores.len() as f32) - self.goal_index - 1.
        } else {
            0.
        };
        // get the optimal average for the farthest position
        let worst_case_average =
            self.get_optimal_average(valid_indices.len(), scores.len(), farthest_position);
        // Can't improve
        if (worst_case_average - optimal_average).abs() <= 1e-6 {
            return (1., None);
        }

        let worst_case_dist = (self.goal_index - worst_case_average).powf(2.);

        // Scale dist and worst_case_dist by optimal_dist as we may never be able to reach the goal
        // This allows the score to be 1 if we reach the optimal arrangement.
        let dist_scaled = dist - optimal_dist;
        let worst_case_dist_scaled = worst_case_dist - optimal_dist;

        // Normalize by maximum distance from the goal possible
        // if avg = goal_index, score = 1. if avg ~ worst_case_dist, score ~ 0
        (
            (worst_case_dist_scaled - dist_scaled) / worst_case_dist_scaled,
            None,
        )
    }
}

#[cfg(test)]
mod scorer_tests {
    use super::*;

    #[test]
    fn test_binary_optimal_average() {
        let param = BinaryScoringParameters {
            field_name: "unused".to_string(),
            field_value: MetaType::Str("unused".to_string()),
            flip_comparator: false,
            goal_index: 0.,
        };
        let scorer = BasicBinaryScorer::new(&param);
        assert_eq!(scorer.get_optimal_average(1, 10, 0.), 0.);
        assert_eq!(scorer.get_optimal_average(1, 10, 9.), 9.);
        assert_eq!(scorer.get_optimal_average(1, 10, 15.), 9.);
        assert_eq!(scorer.get_optimal_average(2, 13, 0.), 0.5);
        assert_eq!(scorer.get_optimal_average(4, 13, 1.), 1.5);
        assert_eq!(scorer.get_optimal_average(5, 13, 1.), 2.);
        assert_eq!(scorer.get_optimal_average(5, 13, 11.), 10.);
        assert_eq!(scorer.get_optimal_average(2, 13, 1.), 1.5);
        assert_eq!(scorer.get_optimal_average(4, 13, 11.), 11.5);
    }
}

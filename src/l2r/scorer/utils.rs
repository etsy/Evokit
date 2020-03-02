extern crate es_core;
extern crate es_data;

use std::fmt::Debug;

use self::es_core::optimizer::ScoreLogger;
use self::es_data::dataset::types::{MetaType, Metadata};
use self::es_data::dataset::Grouping;
use self::es_data::datatypes::Sparse;

use crate::l2r::scorer::parameters::OptimizationGoal;
use crate::l2r::utils::{get_float_field, get_string_field};

/// Given a set of documents, get the metadata values for `field_name`, assuming they are float values
/// # Arguments
///
/// * `field_name` String indicating the name of the Metadata
pub fn get_float_fields(scores: &[(ScoredInstance, &Metadata)], field_name: &String) -> Vec<f32> {
    scores
        .iter()
        .map(|x| get_float_field(x.1, field_name))
        .collect()
}

/// Given a set of documents, get the metadata values for `field_name`, assuming they are string values
/// # Arguments
///
/// * `field_name` String indicating the name of the Metadata
pub fn get_string_fields(
    scores: &[(ScoredInstance, &Metadata)],
    field_name: &String,
    cast_to_string: bool,
) -> Vec<String> {
    scores
        .iter()
        .map(|x| get_string_field(x.1, field_name, cast_to_string))
        .collect()
}

/// Given a set of documents, get the metadata value for `field_name` if provided, otherwise return the relevance score
/// # Arguments
///
/// * `field_name` String indicating the name of the Metadata
pub fn get_relevance_scores(
    scores: &[(ScoredInstance, &Metadata)],
    field_name: &Option<String>,
) -> Vec<f32> {
    match field_name {
        Some(ref fname) => get_float_fields(scores, fname),
        None => scores.iter().map(|x| x.0.label).collect(),
    }
}

/// Given an optimization goal, this normalizes the score so when aiming to:
///  - minimize: lower values -> higher scores
///  - maximize: higher values -> higher scores
/// # Arguments
///
/// * opt_goal: whether to minimize or maximize. If not provided, the actual value is returned
/// * min_val: minimum possible value. Should be pre-computed
/// * max_val: maximum possible value. Should be pre-computed
/// * actual_val: The value to normalize
pub fn normalize_optimization(
    opt_goal: Option<OptimizationGoal>,
    min_val: f32,
    max_val: f32,
    actual_val: f32,
) -> f32 {
    opt_goal
        .map(|opt| {
            let max_range = max_val - min_val;
            match opt {
                _ if max_range == 0. => 1.0,
                OptimizationGoal::Maximize => (actual_val - min_val) / max_range,
                OptimizationGoal::Minimize => (max_val - actual_val) / max_range,
            }
        })
        .unwrap_or(actual_val)
}

/// Given a list of values, this gets the minimum and maximum values
/// # Arguments
///
/// * field_values: list of values
pub fn get_min_and_max(field_values: Vec<f32>) -> (f32, f32) {
    field_values
        .iter()
        .fold((std::f32::MAX, std::f32::MIN), |init, x| {
            (x.min(init.0), x.max(init.1))
        })
}

/// Validates a value
/// # Arguments
///
/// * actual: value to validate
/// * goal: value to compare it with
/// * flip_comparator: used if Num to change from > to <=
///
/// Given a metatype:
/// - if Metatype is Num and flip_comparator = False: x > goal
/// - if Metatype is Num and flip_comparator = True: x <= goal
/// - if Metatype is Str: x == goal
pub fn validate_metatype(actual: &MetaType, goal: &MetaType, flip_comparator: bool) -> bool {
    match (actual, goal) {
        (MetaType::Num(ref val), MetaType::Num(ref threshold)) => {
            let greater_than_check = !flip_comparator && (val > threshold);
            let less_than_or_equal_check = flip_comparator && (val <= threshold);
            greater_than_check || less_than_or_equal_check
        }
        (MetaType::Str(ref val), MetaType::Str(ref expected)) if val == expected => true,
        // Error is raised during the expect below.
        _ => false,
    }
}

#[derive(Debug, Clone)]
/// Representation of a document
pub struct ScoredInstance {
    /// True label, normally relevance
    pub label: f32,
    /// If the pointwise policy was used, we will store the score from the underlying model here
    pub model_score: Option<f32>,
}

/// Trait defining the methods for all scorers. You will need to implement this for any new scorers
pub trait Scorer: Send + Sync + Debug {
    /// Given a set of documents outputted by a policy, this outputs a score
    ///
    /// There is also an optional logger to output stats
    /// # Arguments
    ///
    /// * `scores` List of docs
    fn score(&self, scores: &[(ScoredInstance, &Metadata)]) -> (f32, Option<ScoreLogger>);
}

#[derive(Debug)]
/// Abstraction to box different types of Scorers
pub struct BScorer(pub Box<dyn Scorer>);

impl Scorer for BScorer {
    /// Calls the underlying boxed scorer method
    fn score(&self, scores: &[(ScoredInstance, &Metadata)]) -> (f32, Option<ScoreLogger>) {
        self.0.score(scores)
    }
}

#[derive(Debug)]
/// Wraps a scorer along with a name
pub struct NamedScorer<S: Debug>(String, S);

impl<S: Scorer> NamedScorer<S> {
    /// Initializes a new NamedScorer
    pub fn new(name: String, scorer: S) -> Self {
        NamedScorer(name, scorer)
    }

    /// Calls the underlying score method
    pub fn score(
        &self,
        rs: &Grouping<Sparse>,
        idxs: &[usize],
        scores_opt: &Option<Vec<f32>>,
    ) -> (&str, f32, Option<ScoreLogger>) {
        let v: Vec<_> = match scores_opt {
            Some(scores) => idxs
                .iter()
                .zip(scores)
                .map(|(idx, score)| {
                    (
                        ScoredInstance {
                            label: rs.y[*idx],
                            model_score: Some(*score),
                        },
                        &rs.metadata[*idx],
                    )
                })
                .collect(),
            None => idxs
                .iter()
                .map(|idx| {
                    (
                        ScoredInstance {
                            label: rs.y[*idx],
                            model_score: None,
                        },
                        &rs.metadata[*idx],
                    )
                })
                .collect(),
        };

        let (s, l) = self.1.score(&v);
        (&self.0, s, l)
    }
}

#[cfg(test)]
mod tests {
    use self::es_data::dataset::types::MetaType;
    use super::*;

    #[test]
    fn test_min_max() {
        let (min_val, max_val) = get_min_and_max(vec![2.0, 3.0, 1.0, -2.0, 10.0]);
        assert_eq!(min_val, -2.0);
        assert_eq!(max_val, 10.0);
    }

    #[test]
    fn test_validate_metatype_num() {
        {
            let actual1 = MetaType::Num(10.0);
            let threshold1 = MetaType::Num(10.0);
            // 10 > 10 is false
            assert!(!validate_metatype(&actual1, &threshold1, false));
        }
        {
            let actual2 = MetaType::Num(10.0);
            let threshold2 = MetaType::Num(10.0);
            // 10 <= 10 is true
            assert!(validate_metatype(&actual2, &threshold2, true));
        }
        {
            let actual3 = MetaType::Num(10.0);
            let threshold3 = MetaType::Num(1.0);
            // 10 > 0 is true
            assert!(validate_metatype(&actual3, &threshold3, false));
        }
    }

    #[test]
    fn test_validate_metatype_str() {
        {
            let actual1 = MetaType::Str("test".to_string());
            let threshold1 = MetaType::Str("test".to_string());
            assert!(validate_metatype(&actual1, &threshold1, false));
        }
        {
            let actual1 = MetaType::Str("test".to_string());
            let threshold1 = MetaType::Str("invalid".to_string());
            assert!(!validate_metatype(&actual1, &threshold1, false));
        }
    }

    #[test]
    fn test_normalize_optimization() {
        assert_eq!(
            normalize_optimization(Some(OptimizationGoal::Maximize), 0.0, 10.0, 7.0),
            0.7
        );
        assert_eq!(
            normalize_optimization(Some(OptimizationGoal::Minimize), 0.0, 10.0, 7.0),
            0.3
        );
        assert_eq!(
            normalize_optimization(Some(OptimizationGoal::Maximize), 1.0, 11.0, 8.0),
            0.7
        );
        assert_eq!(
            normalize_optimization(Some(OptimizationGoal::Minimize), 1.0, 11.0, 8.0),
            0.3
        );
        assert_eq!(
            normalize_optimization(Some(OptimizationGoal::Minimize), 11.0, 11.0, 8.0),
            1.0
        );
        assert_eq!(normalize_optimization(None, 1.0, 11.0, 8.0), 8.0);
    }
}

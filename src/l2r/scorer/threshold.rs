extern crate es_core;
extern crate es_data;

use self::es_core::optimizer::ScoreLogger;
use self::es_data::dataset::types::{MetaType, Metadata};

use crate::l2r::scorer::parameters::*;
use crate::l2r::scorer::utils::*;

#[derive(Debug, Clone)]
/// Scorer to compare a value with a threshold
pub struct ThresholdScorer {
    /// Field containing the value
    pub field_name: String,
    /// Value to compare with
    pub field_value: MetaType,
    /// Whether to flip from > to <=
    /// Only matters for numerical MetaTypes. Defaults to val > threshold
    pub flip_comparator: bool,
    /// Which document in the list to look at. 0-indexed
    pub pos: usize,
}

impl ThresholdScorer {
    /// Returns a new ThresholdScorer
    pub fn new(parameters: &ThresholdScoringParameters) -> Self {
        ThresholdScorer {
            field_name: parameters.field_name.clone(),
            field_value: parameters.field_value.clone(),
            flip_comparator: parameters.flip_comparator,
            pos: parameters.pos,
        }
    }
}

impl Scorer for ThresholdScorer {
    /// Compares the value with the threshold
    fn score(&self, scores: &[(ScoredInstance, &Metadata)]) -> (f32, Option<ScoreLogger>) {
        let class = if scores.len() <= self.pos {
            0.
        } else if !scores[self.pos].1.contains_key(&self.field_name) {
            panic!(format!(
                "field_name '{}' is not defined in data",
                &self.field_name
            ))
        } else if validate_metatype(
            &scores[self.pos].1[&self.field_name],
            &self.field_value,
            self.flip_comparator,
        ) {
            1.
        } else {
            0.
        };
        (class, None)
    }
}

#[cfg(test)]
mod tests {
    use self::es_data::dataset::types::MetaType;
    use super::*;

    #[test]
    fn test_threshold_num() {
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
            let param = ThresholdScoringParameters {
                field_name: "price".to_string(),
                field_value: MetaType::Num(5.0),
                flip_comparator: false,
                pos: 0,
            };
            let scorer = ThresholdScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 0.0)
        }
        {
            let param = ThresholdScoringParameters {
                field_name: "price".to_string(),
                field_value: MetaType::Num(5.0),
                flip_comparator: true,
                pos: 0,
            };
            let scorer = ThresholdScorer::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 1.0)
        }
    }
}

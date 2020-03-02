extern crate es_core;
extern crate es_data;

use self::es_core::optimizer::ScoreLogger;
use self::es_data::dataset::types::Metadata;

use crate::l2r::scorer::parameters::*;
use crate::l2r::scorer::utils::*;
use crate::l2r::utils::metatype_to_num;

#[derive(Debug, Clone)]
/// Scorer to extract a value
/// Can be used to bury or boost based on a field or to extract a value for gini or filters
pub struct FieldExtractor {
    /// Field to extract
    field_name: String,
    /// Which document to extract it from
    k: usize,
    /// Value to use if none exists
    sentinel: Option<f32>,
}

impl FieldExtractor {
    /// Returns a FieldExtractor
    pub fn new(params: &FieldExtractorParameters) -> Self {
        FieldExtractor {
            field_name: params.field_name.clone(),
            k: params.k,
            sentinel: params.sentinel,
        }
    }
}

impl Scorer for FieldExtractor {
    /// Gets the field for the specified document
    fn score(&self, scores: &[(ScoredInstance, &Metadata)]) -> (f32, Option<ScoreLogger>) {
        let class = if scores.len() <= self.k {
            self.sentinel.unwrap_or(std::f32::MAX)
        } else if !scores[self.k].1.contains_key(&self.field_name) {
            panic!(format!(
                "field_name '{}' is not defined in data",
                &self.field_name
            ))
        } else {
            metatype_to_num(&scores[self.k].1[&self.field_name])
        };
        (class, None)
    }
}

#[cfg(test)]
mod tests {
    use self::es_data::dataset::types::MetaType;
    use super::*;

    #[test]
    fn test_field_extractor() {
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
            let param = FieldExtractorParameters {
                field_name: "price".to_string(),
                k: 0,
                sentinel: Some(-1.),
            };
            let scorer = FieldExtractor::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 5.0)
        }

        {
            let param = FieldExtractorParameters {
                field_name: "price".to_string(),
                k: 1,
                sentinel: Some(-1.),
            };
            let scorer = FieldExtractor::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, 4.0)
        }

        {
            let param = FieldExtractorParameters {
                field_name: "price".to_string(),
                k: 10,
                sentinel: Some(-1.),
            };
            let scorer = FieldExtractor::new(&param);
            let score_and_logger = scorer.score(&scores);
            assert_eq!(score_and_logger.0, -1.0)
        }
    }
}

extern crate classifier_measures;
use self::classifier_measures::roc_auc;
use crate::l2r::market::utils::*;

/// Indicator for computing AUC
pub struct AUCIndicator {
    /// name of the indicator
    name: String,
}

impl AUCIndicator {
    /// Returns a new AUCIndicator
    pub fn new(name: &str) -> Self {
        AUCIndicator { name: name.into() }
    }
}

impl Indicator for AUCIndicator {
    /// Computes the AUC across all the documents. A classification metric
    fn evaluate(&self, metrics: &Vec<&Metrics>) -> f32 {
        // TODO: remove this clone
        let flattened_scores = metrics
            .iter()
            .flat_map(|x| match (&x.labels, &x.scores) {
                (Some(labels), Some(scores)) => labels.iter().zip(scores.iter()),
                _ => panic!("Must use a policy that outputs a score for AUC"),
            })
            .map(|(&label, &score)| (label, score));
        let score_result = roc_auc(flattened_scores, |x| (x.0 > 0., x.1));
        match score_result {
            Some(v) => v,
            _ => 1.0,
        }
    }

    /// Name of the indicator
    fn name(&self) -> &str {
        &self.name
    }
}

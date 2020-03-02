use crate::l2r::market::utils::*;
use std::collections::HashMap;

/// Indicator for computing Chi-Squared
pub struct ChiSquaredIndicator {
    /// Which query-level scorer to compute this for
    field: String,
    /// Pre-defined number of classes to use
    n_classes: usize,
}

impl ChiSquaredIndicator {
    /// Returns a new ChiSquaredIndicator
    pub fn new(field: &str, n_classes: usize) -> Self {
        ChiSquaredIndicator {
            field: field.into(),
            n_classes,
        }
    }
}

impl Indicator for ChiSquaredIndicator {
    /// Computes Chi-Squared
    fn evaluate(&self, metrics: &Vec<&Metrics>) -> f32 {
        let mut counts = HashMap::new();
        for metric in metrics.iter() {
            let class = metric.read_num(&self.field) as usize;
            let entry = counts.entry(class).or_insert(0usize);
            *entry += 1;
        }

        let expected = metrics.len() as f32 / self.n_classes as f32;

        // Handle the zero case
        let mut sum = (self.n_classes - counts.len()) as f32 * expected.powi(2) / expected;
        for (_class, count) in counts.into_iter() {
            sum += (count as f32 - expected).powi(2) / expected;
        }

        1. / (1. + sum)
    }

    /// Gets the name of the indicator
    fn name(&self) -> &str {
        &self.field
    }
}

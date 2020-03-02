use metrics::*;

use crate::l2r::market::utils::*;

/// Indicator to compute the percentiles
pub struct HistogramIndicator {
    /// Name of query-level scorer to get the percentile for
    field: String,
    /// Which percentiles to compute
    percentiles: Vec<usize>,
}

impl HistogramIndicator {
    /// Returns a HistogramIndicator
    pub fn new(field: &str, percentiles: &[usize]) -> Self {
        HistogramIndicator {
            field: field.into(),
            percentiles: percentiles.to_vec(),
        }
    }
}

impl Indicator for HistogramIndicator {
    /// Computes the percentiles
    fn evaluate(&self, metrics: &Vec<&Metrics>) -> f32 {
        let mut vals: Vec<f32> = metrics.iter().map(|x| x.read_num(&self.field)).collect();
        get_percentiles(&mut vals, &self.percentiles, None)
    }

    /// Gets the name
    fn name(&self) -> &str {
        &self.field
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_histogram() {
        let mapping1: HashMap<String, f32> = [("avg-price-1".to_string(), 1000.0)]
            .iter()
            .cloned()
            .collect();
        let metric1: Metrics = mapping1.into();
        let mapping2: HashMap<String, f32> = [("avg-price-1".to_string(), 20.0)]
            .iter()
            .cloned()
            .collect();
        let metric2: Metrics = mapping2.into();
        let mapping3: HashMap<String, f32> = [("avg-price-1".to_string(), 100.0)]
            .iter()
            .cloned()
            .collect();
        let metric3: Metrics = mapping3.into();
        let metrics: Vec<&Metrics> = vec![&metric1, &metric2, &metric3];
        let quantiles = vec![50];
        let indicator = HistogramIndicator::new("avg-price-1", &quantiles);
        {
            assert_eq!(indicator.evaluate(&metrics), 100.0);
        }
    }
}

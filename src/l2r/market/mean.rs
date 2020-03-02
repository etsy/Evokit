use crate::l2r::market::utils::*;

/// Indicator for computing the mean
pub struct MeanIndicator {
    /// name of the indicator. Also the query-level scorer to use
    name: String,
    /// Value to scale the score by
    scale: Option<f32>,
}

impl MeanIndicator {
    /// Returns a new MeanIndicator
    pub fn new(name: &str, scale: Option<f32>) -> Self {
        MeanIndicator {
            name: name.into(),
            scale,
        }
    }
}

impl Indicator for MeanIndicator {
    /// Computes the mean across all the requests. Scales if asked.
    fn evaluate(&self, metrics: &Vec<&Metrics>) -> f32 {
        let mut s = 0.0;
        for m in metrics.iter() {
            s += m.read_num(&self.name);
        }

        // Optional scale. Allows the score to be between [0,1]
        s / (metrics.len() as f32) * self.scale.unwrap_or(1.)
    }

    /// Name of indicator
    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_mean() {
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
        let metrics: Vec<&Metrics> = vec![&metric1, &metric2];
        let empty_vector: Vec<&Metrics> = vec![];

        let indicator = MeanIndicator::new("avg-price-1", None);
        assert_eq!(indicator.evaluate(&metrics), 510.0);
        assert!(indicator.evaluate(&empty_vector).is_nan());
    }
}

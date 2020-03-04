use std::f32::{INFINITY, NEG_INFINITY};

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Definition of a histogram
pub(super) struct Histogram(Vec<(f32, f32)>);

impl Histogram {
    /// Initialize a new histogram from a set of points given the number of features
    pub(super) fn new(n_features: usize, data: impl Iterator<Item = (usize, f32)>) -> Self {
        let mut min = vec![INFINITY; n_features];
        let mut max = vec![NEG_INFINITY; n_features];

        for (i, value) in data {
            min[i] = value.min(min[i]);
            max[i] = value.max(max[i]);
        }

        // Take care of missing data.  If we don't have it, we apply no scaling.
        // This will let us handle cases where data is augmented with, say, stochastic
        // features
        for i in 0..n_features {
            if min[i] == INFINITY {
                min[i] = 0.;
                max[i] = 1.
            }
        }

        Histogram(min.into_iter().zip(max.into_iter()).collect())
    }

    #[inline]
    pub(super) fn scale(&self, index: usize, value: f32) -> f32 {
        let (low, high) = self.0[index];
        value * (high - low) + low
    }
}

#[cfg(test)]
mod tree_histogram {
    use super::*;

    fn build_hist() -> Histogram {
        let data = vec![vec![0.3, -1.0], vec![7., 0.9], vec![10., 1.0]];
        let it = data
            .iter()
            .flat_map(|v| v.iter().enumerate().map(|(idx, f)| (idx, *f)));
        let histogram = Histogram::new(2, it);
        histogram
    }

    #[test]
    fn test_histogram() {
        let histogram = build_hist();
        assert_eq!(histogram.0.len(), 2);
        assert_eq!(histogram.0[0], (0.3, 10.));
        assert_eq!(histogram.0[1], (-1., 1.));

        // Test scaling
        let value = 0.2;
        let res = histogram.scale(0, value);
        assert_eq!(0.2 * (10. - 0.3) + 0.3, res);
    }

    #[test]
    fn test_histogram_inf() {
        let data = vec![vec![0.3, -1.0], vec![7., 0.9], vec![10., 1.0]];
        let it = data
            .iter()
            .flat_map(|v| v.iter().enumerate().map(|(idx, f)| (idx, *f)));
        let histogram = Histogram::new(3, it);
        assert_eq!(histogram.0[2], (0., 1.));
    }
}

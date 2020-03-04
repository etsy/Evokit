extern crate es_data;

use self::es_data::datatypes::Sparse;
use crate::l2r::aggregator::utils::*;

#[derive(Clone)]
/// Exponential Moving Average (EMA)
pub struct EMAAgg {
    /// gamma parameter
    gamma: f32,
    /// Number of docs seen
    seen: usize,
    /// The current aggregation
    agg: Sparse,
}

impl EMAAgg {
    /// New EMAAgg
    pub fn new(gamma: f32, dims: usize) -> Self {
        EMAAgg {
            gamma,
            seen: 0,
            agg: Sparse(dims, vec![], vec![]),
        }
    }
}

impl Aggregator for EMAAgg {
    /// updates the moving average
    fn update(&mut self, item: &Sparse) -> () {
        if self.seen == 0 {
            self.agg = item.clone();
        } else {
            let decay = 1. - self.gamma;
            self.agg *= self.gamma;
            self.agg += &(item * decay);
        }

        self.seen += 1;
    }

    /// Gets the current aggregator
    fn read(&self) -> &Sparse {
        &self.agg
    }
}

#[derive(Clone, Copy)]
/// Builder to get a EMAAgg. Requires the Sparse vector size
pub struct EMABuilder(pub f32, pub usize);

impl AggBuilder for EMABuilder {
    type Agg = EMAAgg;

    /// Initialize a EMAAgg
    fn start(&self) -> Self::Agg {
        EMAAgg::new(self.0, self.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_agg() {
        let dims: usize = 5;
        let gamma = 0.3_f32;
        let builder = EMABuilder(gamma, dims);
        let mut aggregator = builder.start();
        // First round should be just the value
        // Any new items should be scaled even if that index hasn't occurred before
        // We still scale the items even if we don't see that index in the new data.
        let data = vec![
            Sparse(dims, vec![1, 4], vec![3., 1.]),
            Sparse(dims, vec![2, 4], vec![3., 5.]),
        ];

        let expected = vec![
            Sparse(dims, vec![1, 4], vec![3., 1.]),
            Sparse(
                dims,
                vec![1, 2, 4],
                vec![
                    gamma * 3.,
                    (1. - gamma) * 3.,
                    gamma * 1. + (1. - gamma) * 5.,
                ],
            ),
        ];

        validate_agg(&mut aggregator, &data, &expected);
    }
}

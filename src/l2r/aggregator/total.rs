extern crate es_data;

use self::es_data::datatypes::Sparse;
use crate::l2r::aggregator::utils::*;

#[derive(Clone)]
/// Sum up the feature
pub struct TotalAgg {
    /// The current aggregation
    agg: Sparse,
}

impl TotalAgg {
    /// Creates a new TotalAgg
    pub fn new(dims: usize) -> Self {
        TotalAgg {
            agg: Sparse(dims, vec![], vec![]),
        }
    }
}

impl Aggregator for TotalAgg {
    /// Increment the total
    fn update(&mut self, item: &Sparse) -> () {
        self.agg += item;
    }

    /// Gets the current aggregator
    fn read(&self) -> &Sparse {
        &self.agg
    }
}

#[derive(Clone, Copy)]
/// Builder to get a TotalAgg. Requires the Sparse vector size
pub struct TotalAggBuilder(pub usize);

impl AggBuilder for TotalAggBuilder {
    type Agg = TotalAgg;

    /// Initialize a TotalAgg
    fn start(&self) -> Self::Agg {
        Self::Agg::new(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_total_agg() {
        let dims: usize = 5;
        let builder = TotalAggBuilder(dims);
        let mut aggregator = builder.start();
        let data = vec![
            Sparse(dims, vec![1, 4], vec![3., 1.]),
            Sparse(dims, vec![2, 4], vec![3., 5.]),
        ];

        let expected = vec![
            Sparse(dims, vec![1, 4], vec![3., 1.]),
            Sparse(dims, vec![1, 2, 4], vec![3., 3., 6.]),
        ];

        validate_agg(&mut aggregator, &data, &expected);
    }
}

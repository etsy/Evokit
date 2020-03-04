extern crate es_data;

use self::es_data::datatypes::Sparse;
use crate::l2r::aggregator::utils::*;

#[derive(Clone)]
/// Counts the number of times the feature was seen
pub struct CountAgg {
    /// The current aggregation
    agg: Sparse,
}

impl CountAgg {
    /// Creates a new CountAgg
    pub fn new(dims: usize) -> Self {
        CountAgg {
            agg: Sparse(dims, vec![], vec![]),
        }
    }
}

impl Aggregator for CountAgg {
    /// Increment the count
    fn update(&mut self, item: &Sparse) -> () {
        self.agg = self.agg.combine(
            &item,
            #[inline]
            |l, r| l.unwrap_or(0.) + r.map(|_v| 1.).unwrap_or(0.),
        );
    }

    /// Gets the current aggregator
    fn read(&self) -> &Sparse {
        &self.agg
    }
}

#[derive(Clone, Copy)]
/// Builder to get a CountAgg. Requires the Sparse vector size
pub struct CountAggBuilder(pub usize);

impl AggBuilder for CountAggBuilder {
    type Agg = CountAgg;

    /// Initialize a CountAgg
    fn start(&self) -> Self::Agg {
        Self::Agg::new(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_agg() {
        let dims: usize = 5;
        let builder = CountAggBuilder(dims);
        let mut aggregator = builder.start();
        let data = vec![
            Sparse(dims, vec![1, 4], vec![3., 1.]),
            Sparse(dims, vec![2, 4], vec![3., 5.]),
        ];

        let expected = vec![
            Sparse(dims, vec![1, 4], vec![1., 1.]),
            Sparse(dims, vec![1, 2, 4], vec![1., 1., 2.]),
        ];

        validate_agg(&mut aggregator, &data, &expected);
    }
}

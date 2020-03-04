extern crate es_data;

use self::es_data::datatypes::Sparse;
use crate::l2r::aggregator::count::CountAgg;
use crate::l2r::aggregator::total::TotalAgg;
use crate::l2r::aggregator::utils::*;

#[derive(Clone)]
/// Aggregator to compute the mean across the docs
pub struct MeanAgg {
    /// The total seen
    total: TotalAgg,
    /// The number seen
    count: CountAgg,
    /// The current aggregation
    agg: Sparse,
}

impl MeanAgg {
    /// Creates a new MeanAgg
    pub fn new(dims: usize) -> Self {
        MeanAgg {
            total: TotalAgg::new(dims),
            count: CountAgg::new(dims),
            agg: Sparse(dims, vec![], vec![]),
        }
    }
}

impl Aggregator for MeanAgg {
    /// Increments the mean by incrementing the total and count
    /// empty docs are excluded in mean calculation
    fn update(&mut self, item: &Sparse) -> () {
        self.total.update(item);
        self.count.update(item);
        self.agg = self.total.read() / self.count.read();
    }

    /// Gets the current aggregator
    fn read(&self) -> &Sparse {
        &self.agg
    }
}

#[derive(Clone, Copy)]
/// Builder to get a MeanAgg. Requires the Sparse vector size
pub struct MeanAggBuilder(pub usize);

impl AggBuilder for MeanAggBuilder {
    type Agg = MeanAgg;

    /// Initialize a MeanAgg
    fn start(&self) -> Self::Agg {
        Self::Agg::new(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrics::*;

    #[test]
    fn test_mean_agg() {
        let dims: usize = 5;
        let builder = MeanAggBuilder(dims);
        let mut aggregator = builder.start();
        // Test initialization of mean (n = 1) works
        // Test incremental update of mean works
        // Test empty doc ignored in count
        let data = vec![
            Sparse(dims, vec![1, 4], vec![3., 1.]),
            Sparse(dims, vec![2, 4], vec![3., 5.]),
            Sparse(dims, vec![1, 4], vec![4., 7.]),
            Sparse(dims, vec![2, 4], vec![9., 2.]),
            Sparse(dims, vec![], vec![]),
        ];

        let expected = vec![
            Sparse(dims, vec![1, 4], vec![3., 1.]),
            Sparse(
                dims,
                vec![1, 2, 4],
                vec![3., 3., get_mean(&vec![1_f32, 5_f32], None)],
            ),
            Sparse(
                dims,
                vec![1, 2, 4],
                vec![
                    get_mean(&vec![3_f32, 4_f32], None),
                    3.,
                    get_mean(&vec![1_f32, 5_f32, 7_f32], None),
                ],
            ),
            // empty doc so don't update
            Sparse(
                dims,
                vec![1, 2, 4],
                vec![
                    get_mean(&vec![3_f32, 4_f32], None),
                    get_mean(&vec![3_f32, 9_f32], None),
                    get_mean(&vec![1_f32, 5_f32, 7_f32, 2_f32], None),
                ],
            ),
        ];

        validate_agg(&mut aggregator, &data, &expected);
    }
}

extern crate es_data;

use self::es_data::datatypes::Sparse;
use crate::l2r::aggregator::utils::*;

#[derive(Clone)]
/// Aggregator to compute variance
pub struct VarianceAgg {
    /// Number of docs seen
    num_items: Sparse,
    /// Current mean
    mean: Sparse,
    /// First values seen
    first_time: Sparse,
    /// The current aggregation
    agg: Sparse,
}

impl VarianceAgg {
    /// Returns a new VarianceAgg
    pub fn new(dims: usize) -> Self {
        VarianceAgg {
            num_items: Sparse(dims, vec![], vec![]),
            mean: Sparse(dims, vec![], vec![]),
            first_time: Sparse(dims, vec![], vec![]),
            agg: Sparse(dims, vec![], vec![]),
        }
    }
}

impl Aggregator for VarianceAgg {
    /// Computes the running variance
    /// empty docs are excluded in mean calculation
    fn update(&mut self, item: &Sparse) -> () {
        // 0) Clone the items we need immutable copies of in the closure as we will modify self.
        let num_items_clone = self.num_items.2.clone();
        let mean_clone = self.mean.2.clone();

        // 1) Handle indices we already have variance for
        // 1a) Compute new mean for existing indices
        let mut incremental_mean = Sparse(self.mean.0, vec![], vec![]);
        let local_mean_incrementer_fn =
            |idx: usize, b: f32, a: f32| mean_incrementer_fn(idx, b, a, &num_items_clone);
        // if the idx isn't in self.mean, ignore it for now because it won't be used in self.agg
        combine_vectors_with_init_and_combiner(
            &self.mean,
            item,
            &init_none_fn,
            &local_mean_incrementer_fn,
            &mut incremental_mean,
        );

        // 1b) Compute new variance for existing indices
        let mut incremental_variance = Sparse(self.agg.0, vec![], vec![]);
        let variance_incrementer_fn = |idx: usize, b: f32, a: f32| {
            // idx is the same for self.num_items, self.mean, self.agg
            // idx is the same for incremental_mean because we only included items in self.mean
            b + ((a - mean_clone[idx]) * (a - incremental_mean.2[idx]) - b)
                / (num_items_clone[idx] + 1.)
        };
        combine_vectors_with_init_and_combiner(
            &self.agg,
            item,
            &init_none_fn,
            &variance_incrementer_fn,
            &mut incremental_variance,
        );

        // 2) Handle indices we don't have variance for yet
        // 2a) Split indices we don't have variance for into ones we have never seen before and
        // ones we have seen already
        let mut remaining_indices = Sparse(self.agg.0, vec![], vec![]);
        set_difference(item, &self.agg, &mut remaining_indices);
        let mut seen_once = Sparse(self.agg.0, vec![], vec![]);
        set_difference(&remaining_indices, &self.first_time, &mut seen_once);
        let mut seen_twice = Sparse(self.agg.0, vec![], vec![]);
        set_difference(&remaining_indices, &seen_once, &mut seen_twice);

        // 2b) Handle items we have seen once by merging it with first_time
        // These are brand new indices that need to be saved for the next update call.
        // Get the indices from last time we still haven't handled.
        let mut seen_once_previously = Sparse(self.agg.0, vec![], vec![]);
        set_difference(&self.first_time, &seen_twice, &mut seen_once_previously);
        let mut updated_first_time = Sparse(self.agg.0, vec![], vec![]);
        // can use add_vector as these sets are disjoint
        add_vector(&seen_once, &seen_once_previously, &mut updated_first_time);

        // 2c) Handle the seen twice case by computing mean and variance.
        let mut remaining_mean = Sparse(self.agg.0, vec![], vec![]);
        let mean_fn = |_idx: usize, b: f32, a: f32| (a + b) / 2.;
        combine_vectors_with_init_and_combiner(
            &seen_twice,
            &self.first_time,
            &init_none_fn,
            &mean_fn,
            &mut remaining_mean,
        );
        // variance
        let mut remaining_variance = Sparse(self.agg.0, vec![], vec![]);
        let variance_fn = |idx: usize, b: f32, a: f32| {
            (b.powf(2.) + a.powf(2.)) / 2. - remaining_mean.2[idx].powf(2.)
        };
        combine_vectors_with_init_and_combiner(
            &seen_twice,
            &self.first_time,
            &init_none_fn,
            &variance_fn,
            &mut remaining_variance,
        );

        // 3) increment num_items accordingly
        // increment num_items by 1 for the existing cases
        let mut incremental_num_items = Sparse(self.agg.0, vec![], vec![]);
        combine_vectors_with_init_and_combiner(
            &self.num_items,
            item,
            &init_none_fn,
            &|_idx: usize, b: f32, _a: f32| b + 1.,
            &mut incremental_num_items,
        );
        // Also set num_items to 2 for the new cases because we included the item saved last time
        let mut updated_num_items = Sparse(self.agg.0, vec![], vec![]);
        combine_vectors_with_init_and_combiner(
            &incremental_num_items,
            &seen_twice,
            &|_x: f32| Some(2.),
            &|_idx: usize, b: f32, _a: f32| b,
            &mut updated_num_items,
        );

        // 4) Merge the two partial lists of mean and variance.
        // We can use add_vector here because the lists are disjoint.
        let mut updated_mean = Sparse(self.agg.0, vec![], vec![]);
        add_vector(&incremental_mean, &remaining_mean, &mut updated_mean);
        let mut updated_var = Sparse(self.agg.0, vec![], vec![]);
        add_vector(&incremental_variance, &remaining_variance, &mut updated_var);

        // 5) actually update the state
        copy_anchor(&updated_num_items, &mut self.num_items);
        copy_anchor(&updated_first_time, &mut self.first_time);
        copy_anchor(&updated_mean, &mut self.mean);
        copy_anchor(&updated_var, &mut self.agg);
    }

    /// Gets the current aggregator
    fn read(&self) -> &Sparse {
        &self.agg
    }
}

#[derive(Clone, Copy)]
/// Builder to get a VarianceAgg. Requires the Sparse vector size
pub struct VarianceAggBuilder(pub usize);

impl AggBuilder for VarianceAggBuilder {
    type Agg = VarianceAgg;

    /// Initialize a VarianceAgg
    fn start(&self) -> Self::Agg {
        Self::Agg::new(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrics::*;

    /// Computes traditional variance (for tests)
    /// assumes data has at least two items as this is a helper function for tests
    fn compute_traditional_variance(data: &Vec<f32>) -> f32 {
        let mean = get_mean(data, None);
        let mut result: f32 = 0.;

        for val in data {
            let delta: f32 = val - mean;
            result += delta.powf(2.);
        }

        result / (data.len() as f32)
    }

    #[test]
    fn test_variance_agg() {
        let dims: usize = 5;
        let builder = VarianceAggBuilder(dims);
        let mut aggregator = builder.start();
        // Test variance isn't updated when only one instance of an index exists
        // Test variance is initialized when n = 2
        // Test incremental update of variance works for n = 3
        // Test incremental update of variance works for n = 4 (to make sure n was updated correctly)
        // Test empty doc ignored in count
        let data = vec![
            Sparse(dims, vec![1, 4], vec![3., 1.]),
            Sparse(dims, vec![2, 4], vec![3., 5.]),
            Sparse(dims, vec![1, 4], vec![4., 7.]),
            Sparse(dims, vec![2, 4], vec![9., 2.]),
            Sparse(dims, vec![], vec![]),
        ];

        let expected = vec![
            // only have one instance of each index
            Sparse(dims, vec![], vec![]),
            Sparse(
                dims,
                vec![4],
                vec![compute_traditional_variance(&vec![1_f32, 5_f32])],
            ),
            Sparse(
                dims,
                vec![1, 4],
                vec![
                    compute_traditional_variance(&vec![3_f32, 4_f32]),
                    compute_traditional_variance(&vec![1_f32, 5_f32, 7_f32]),
                ],
            ),
            Sparse(
                dims,
                vec![1, 2, 4],
                vec![
                    compute_traditional_variance(&vec![3_f32, 4_f32]),
                    compute_traditional_variance(&vec![3_f32, 9_f32]),
                    compute_traditional_variance(&vec![1_f32, 5_f32, 7_f32, 2_f32]),
                ],
            ),
            Sparse(
                dims,
                vec![1, 2, 4],
                vec![
                    compute_traditional_variance(&vec![3_f32, 4_f32]),
                    compute_traditional_variance(&vec![3_f32, 9_f32]),
                    compute_traditional_variance(&vec![1_f32, 5_f32, 7_f32, 2_f32]),
                ],
            ),
        ];

        validate_agg(&mut aggregator, &data, &expected);
    }
}

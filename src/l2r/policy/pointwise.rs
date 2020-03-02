extern crate es_core;
extern crate es_data;
extern crate float_ord;
extern crate rand;
extern crate rand_xorshift;

use self::float_ord::FloatOrd;

use self::rand::distributions::{Distribution, Uniform};
use self::rand::SeedableRng;
use self::rand_xorshift::XorShiftRng;

use self::es_core::model::Evaluator;
use self::es_data::dataset::Grouping;
use self::es_data::datatypes::Sparse;

use crate::l2r::policy::utils::*;

/// Policy to score each doc individually
pub struct PointwisePolicy {
    stochastic: bool,
}

impl PointwisePolicy {
    /// New PointwisePolicy
    pub fn new(stochastic: bool) -> Self {
        PointwisePolicy { stochastic }
    }
}

impl Policy for PointwisePolicy {
    /// Lets the upstream environment know that the evaluation is stochastic
    fn is_stochastic(&self) -> bool {
        self.stochastic
    }

    /// Scores each doc individually, then sorts based on these values. Also outputs the scores
    /// from the underlying model
    fn evaluate<M: Evaluator<Sparse, f32>>(
        &self,
        state: &M,
        rs: &Grouping<Sparse>,
        seed: u32,
        idx: usize,
    ) -> (Vec<usize>, Option<Vec<f32>>) {
        let mut rng = XorShiftRng::seed_from_u64(seed as u64 + idx as u64);
        let dist = Uniform::new_inclusive(-1f32, 1.);
        let bit = dist.sample(&mut rng) as f32;
        let mut v: Vec<(f32, usize)> = (0..rs.x.len())
            .map(|i| {
                let yi_hat = if self.stochastic {
                    let mut sparse = rs.x[i].clone();
                    sparse.1.push(sparse.0 - 1);
                    sparse.2.push(bit);
                    state.evaluate(&sparse)
                } else {
                    state.evaluate(&rs.x[i])
                };
                (yi_hat, i)
            })
            .collect();

        // Sort descending by y_hat
        v.sort_by_key(|x| FloatOrd(x.0));
        v.reverse();
        let (scores, indices) = v.into_iter().unzip();
        (indices, Some(scores))
    }
}

#[cfg(test)]
mod tests {
    extern crate es_models;

    use self::es_data::datatypes::Dense;
    use self::es_models::linear::DenseWrapper;
    use super::*;

    #[test]
    fn test_pointwise_evaluate() {
        // verify we sort in descending order
        let base_policy = PointwisePolicy::new(false);

        // Identity on a single value
        let state = DenseWrapper { w: Dense(vec![1.]) };
        let x = vec![
            Sparse(1, vec![0], vec![-1.]),
            Sparse(1, vec![0], vec![0.]),
            Sparse(1, vec![0], vec![2.]),
        ];
        let y = vec![1.0; 3];
        let rs = Grouping::new(x, y);

        let (sorted_ids, _scores) = base_policy.evaluate(&state, &rs, 1234, 0);
        assert_eq!(sorted_ids, vec![2, 1, 0]);
    }
}

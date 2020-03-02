extern crate es_core;
extern crate es_data;
extern crate rand;
extern crate rand_xorshift;

use self::rand::SeedableRng;

use self::rand::seq::SliceRandom;
use self::rand_xorshift::XorShiftRng;

use self::es_core::model::Evaluator;
use self::es_data::dataset::Grouping;
use self::es_data::datatypes::Sparse;

use crate::l2r::policy::utils::*;

/// Policy for randomly ordering the documents
pub struct RandomPolicy {
    /// Seed
    seed: Option<u32>,
}

impl RandomPolicy {
    /// Returns a new RandomPolicy
    pub fn new(seed: Option<u32>) -> Self {
        RandomPolicy { seed }
    }
}

impl Policy for RandomPolicy {
    /// Lets the upstream environment know that the evaluation is stochastic
    fn is_stochastic(&self) -> bool {
        self.seed.is_none()
    }

    /// Randomly shuffles the documents
    fn evaluate<M: Evaluator<Sparse, f32>>(
        &self,
        _state: &M,
        rs: &Grouping<Sparse>,
        seed: u32,
        _idx: usize,
    ) -> (Vec<usize>, Option<Vec<f32>>) {
        let mut scores: Vec<_> = (0..rs.len()).collect();
        let seed = self.seed.unwrap_or(seed << 2 + 1) as u64;
        let mut rng = XorShiftRng::seed_from_u64(seed);
        scores.as_mut_slice().shuffle(&mut rng);
        (scores, None)
    }
}

//! Defines interfaces for learning to rank functions

extern crate es_core;
extern crate es_data;
extern crate es_models;

/// Functions for loading scoring configs
pub mod load;

/// Functions for Mulberry environment
pub mod anchor;

/// Structs for aggregating information across documents
pub mod aggregator;

/// Structs for defining policies
pub mod policy;

/// Defines different query-level scorers
pub mod scorer;

/// Defines different market-level indicators
pub mod market;

/// Defines filters on requests
pub mod filters;

/// Defines query-level metadata
pub mod metadata_extractors;

/// Defines ES-Rank and helper methods:w
mod utils;

use std::fmt::Debug;
use std::marker::PhantomData;

use self::utils::eval_rs;

use self::es_core::model::Evaluator;
use self::es_core::optimizer::{Environment, ScoreLogger};

use self::es_models::regularizer::WeightDecay;

use self::es_data::dataset::{Dataset, EagerIterator};

/// Environment for basic ES-Rank. Without the bells & whistles of Mulberry
pub struct LtrEnv<TS, VS, M, D> {
    /// Training set
    pub train: Vec<(f32, TS)>,
    /// Validation set
    pub valid: Option<VS>,
    /// Weight decay
    pub weight_decay: f32,
    /// K value for train NDCG
    pub k: Option<usize>,
    /// K value for validation NDCG
    pub valid_k: Option<usize>,
    /// Allows the struct to act like it owns something of type M (even though it doesn't). This is used for the evaluator
    pub pd: PhantomData<M>,
    /// Allows the struct to act like it owns a policy even though it doesn't
    pub d: PhantomData<D>,
}

impl<TS, VS, M, D> LtrEnv<TS, VS, M, D> {
    /// Returns new LtrEnv
    pub fn new(
        train: Vec<(f32, TS)>,
        v: Option<VS>,
        weight_decay: f32,
        k: Option<usize>,
        valid_k: Option<usize>,
    ) -> Self {
        LtrEnv {
            train,
            valid: v,
            weight_decay,
            k,
            valid_k,
            pd: PhantomData,
            d: PhantomData,
        }
    }
}

// ES-Rank Environment
impl<TS, VS, M, D> Environment<M> for LtrEnv<TS, VS, M, D>
where
    TS: Dataset<D, Iter = EagerIterator<D>> + Send + Sync,
    VS: Dataset<D, Iter = EagerIterator<D>> + Send + Sync,
    D: Debug + Sync + Send,
    M: Evaluator<D, f32> + WeightDecay + Send + Sync,
{
    /// Updates the environment with the provided
    /// `true` indicates the environment is stochastic
    fn step(&mut self) -> bool {
        let mut shuffle = false;
        for &mut (_, ref mut d) in self.train.iter_mut() {
            shuffle |= d.shuffle();
        }
        shuffle
    }

    /// Evaluates a given state, returning the fitness.
    /// Higher is better.
    fn eval(&self, state: &M) -> (f32, Option<ScoreLogger>) {
        // Compute weight decay
        let wd = if self.weight_decay > 0.0 {
            self.weight_decay * state.l2norm()
        } else {
            0.0
        };

        let mut score = 0.0;
        let mut weight = 0.0;
        for (i, &(w, ref dataset)) in self.train.iter().enumerate() {
            let k = if i == 0 { self.k } else { None };
            score += w * eval_rs(dataset.data(), state, k);
            weight += w;
        }
        ((score / weight) - wd, None)
    }

    /// Evaluates a given state, potentially against
    /// Validation data.  If there is no validation,
    /// returns None
    fn validate(&self, state: &M) -> (Option<f32>, Option<ScoreLogger>) {
        match &self.valid {
            &Some(ref rs) => (Some(eval_rs(rs.data(), state, self.valid_k)), None),
            &None => (None, None),
        }
    }
}

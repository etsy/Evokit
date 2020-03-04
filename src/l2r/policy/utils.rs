extern crate es_core;
extern crate es_data;

use self::es_core::model::Evaluator;
use self::es_data::dataset::Grouping;
use self::es_data::datatypes::Sparse;

/// Trait defining the methods for all policies
pub trait Policy: Send + Sync {
    /// Lets the upstream environment know that the evaluation is stochastic
    fn is_stochastic(&self) -> bool;

    /// Runs a policy to order the documents
    fn evaluate<'a, M: Evaluator<Sparse, f32>>(
        &self,
        state: &M,
        rs: &Grouping<Sparse>,
        seed: u32,
        idx: usize,
    ) -> (Vec<usize>, Option<Vec<f32>>);
}

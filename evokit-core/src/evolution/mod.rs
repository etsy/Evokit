mod de;

use crate::optimizer::{Environment, State};

pub use self::de::{
    CrossoverType, DifferentialEvolution, DiscardStrategy, Mutation, StochasticUpdate,
};

/// Implements a phenotype with a given gene type.
pub trait Genes<GeneType> {
    fn num_genes(&self) -> usize;

    fn get_gene(&self, index: usize) -> GeneType;

    fn set_gene(&mut self, index: usize, new_gene: GeneType);
}

/// Implements a continuous genetic optimizer, which maximizes continuous gene models.
pub trait ContinuousGeneticOptimizer {
    fn run<S, E>(&self, init: S, env: &mut E) -> State<S>
    where
        E: Environment<S>,
        S: Clone + Genes<f32>;
}

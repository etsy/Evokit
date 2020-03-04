pub mod sampler;

use std::io::{Read, Write};

/// Initializes an empty model
pub trait Initializer {
    type Model: Clone + Send + Sync;

    fn zero(&self) -> Self::Model;
}

/// Evaluates a model with a given payload to a given output
pub trait Evaluator<Payload: ?Sized, Output: ?Sized>: Sync {
    fn evaluate(&self, payload: &Payload) -> Output;
}

/// Updates a given model's weights
pub trait WeightUpdater {
    #[inline]
    fn num_weights(&self) -> usize;

    #[inline]
    fn update_gradients<F>(&mut self, f: &mut F) -> ()
    where
        F: FnMut() -> f32;

    #[inline]
    fn scale_gradients(&mut self, f: f32) -> ();

    #[inline]
    fn copy_gradients(&self, other: &mut Self) -> ();

    #[inline]
    fn add_gradients(&mut self, other: &Self) -> ();
}

/// Updates a model with a gradient
pub trait GradientFuser<A> {
    fn update(&self, grad: &A, into: &mut Self) -> ();
}

/// Serialization for models
pub trait SerDe: Sized {
    /// Error conditions due to writing
    type Error;

    /// Writes out a model to writer
    fn save<A: Write>(&self, writer: &mut A) -> Result<(), Self::Error>;

    /// Loads a model from a reader.  All necessary metadata should be
    /// stored within the model
    fn load<A: Read>(reader: &mut A) -> Result<Self, Self::Error>;
}

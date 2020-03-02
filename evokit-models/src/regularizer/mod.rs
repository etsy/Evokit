/// Creates a weight decay model
pub trait WeightDecay {
    /// Computes the l2 norm for the model
    fn l2norm(&self) -> f32;
}

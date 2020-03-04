extern crate es_models;

use self::es_models::nn::NonLinearity;

/// Capture the model parameters. Right now just neural network
pub struct ModelParams<'a> {
    /// hidden nodes
    pub hidden_nodes: Option<Vec<usize>>,
    /// Type of non-linearity between layers
    pub act: NonLinearity,
    /// Optional path to load a model
    pub load_model_path: Option<&'a str>,
    /// Optional path to save the model
    pub save_model_path: Option<&'a str>,
}

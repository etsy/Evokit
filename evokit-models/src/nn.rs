//! Defines a fully connected neural network for use with Evokit
use es_core;
use es_data;
use serde_json;

use std::f32::consts::E;
use std::io::{Read, Write};
use std::mem;

use self::es_core::evolution::Genes;
use self::es_core::model::SerDe;
use self::es_core::model::{Evaluator, GradientFuser, Initializer, WeightUpdater};

use self::es_data::datatypes::Sparse;
use self::es_data::intrinsics::{dot, inplace_sum, l2norm, scale, sparse_dot, sum};

use crate::regularizer::WeightDecay;

use super::*;

#[derive(Serialize, Deserialize, Clone, Debug, Copy)]
/// Defines the different types of functions between layers
pub enum NonLinearity {
    /// ReLu
    ReLu,

    /// Tanh
    Tanh,

    /// Linear. Note this isn't non-linear
    Linear,

    /// Sigmoid
    Sigmoid,

    /// ELU
    ELU,
}

impl NonLinearity {
    #[inline]
    /// Applies the specified function
    fn eval(&self, f: f32) -> f32 {
        use self::NonLinearity::*;
        match self {
            &ReLu => f.max(0f32),
            &Tanh => f.tanh(),
            &Sigmoid => 1. / (1. + E.powf(-f)),
            &Linear => f,
            &ELU => {
                if f > 0. {
                    f
                } else {
                    f.exp() - 1.
                }
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
/// Represents a layer of a neural network
struct Layer {
    /// The weights of the layer
    w: Vec<Vec<f32>>,
    /// The bias
    bias: Vec<f32>,
    /// The type of non-linearity to apply
    nl: NonLinearity,
}

impl Layer {
    /// Creates a new layer
    pub fn new(input_dim: usize, hidden_dims: usize, nl: NonLinearity) -> Self {
        let w = (0..hidden_dims)
            .map({ |_x| vec![0f32; input_dim] })
            .collect();

        Layer {
            w: w,
            bias: vec![0f32; hidden_dims],
            nl: nl,
        }
    }

    /// Gets the dimensions of the layer
    fn dims(&self) -> (usize, usize) {
        (self.w[0].len(), self.bias.len())
    }

    /// Given a dense payload, applies the weights, bias, and non-linearity to it
    fn eval(&self, payload: &[f32], output: &mut [f32]) -> () {
        for i in 0..self.w.len() {
            output[i] = dot(&self.w[i], payload);
        }

        inplace_sum(&mut output[0..self.bias.len()], &self.bias);
        for i in 0..self.bias.len() {
            output[i] = self.nl.eval(output[i]);
        }
    }

    /// Given a sparse payload, applies the weights, bias, and non-linearity to it
    fn sparse_eval(&self, payload: &Sparse, output: &mut [f32]) -> () {
        for i in 0..self.w.len() {
            output[i] = sparse_dot(&payload.1, &payload.2, &self.w[i]);
        }

        inplace_sum(&mut output[0..self.bias.len()], &self.bias);
        for i in 0..self.bias.len() {
            output[i] = self.nl.eval(output[i]);
        }
    }

    /// Computes the l2 norm of the weights + bias
    fn l2norm(&self) -> f32 {
        let norm: f32 = self.w.iter().map(|r| l2norm(&r)).sum();
        norm + l2norm(&self.bias)
    }
}

impl WeightUpdater for Layer {
    #[inline]
    fn num_weights(&self) -> usize {
        self.w.len() * (self.w[0].len() + 1)
    }

    fn update_gradients<F>(&mut self, f: &mut F) -> ()
    where
        F: FnMut() -> f32,
    {
        for mut r in self.w.iter_mut() {
            update_vec(&mut r, f);
        }

        update_vec(&mut self.bias, f);
    }

    fn scale_gradients(&mut self, s: f32) -> () {
        for mut r in self.w.iter_mut() {
            scale(&mut r, s);
        }

        scale(&mut self.bias, s);
    }

    fn copy_gradients(&self, other: &mut Self) {
        assert_eq!(self.num_weights(), other.num_weights());
        for i in 0..self.w.len() {
            copy_vec(&self.w[i], &mut other.w[i]);
        }
        copy_vec(&self.bias, &mut other.bias);
    }

    fn add_gradients(&mut self, other: &Self) {
        assert_eq!(self.num_weights(), other.num_weights());
        for i in 0..self.w.len() {
            add_vec(&mut self.w[i], &other.w[i]);
        }
        add_vec(&mut self.bias, &other.bias);
    }
}

impl GradientFuser<Layer> for Layer {
    fn update(&self, grad: &Layer, into: &mut Self) -> () {
        assert_eq!(self.w.len(), grad.w.len());
        assert_eq!(self.bias.len(), grad.bias.len());

        // Copy weights
        for i in 0..self.w.len() {
            sum(&self.w[i], &grad.w[i], &mut into.w[i]);
        }

        sum(&self.bias, &grad.bias, &mut into.bias);
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
/// Representation of a full network
pub struct Network {
    /// All the layers
    layers: Vec<Layer>,
    /// Number of parameters. Used for memory optimizations
    max_dim: usize,
}

impl Network {
    /// Initializes a new network with the specified layers
    pub fn new(input_dim: usize, sizes: &[usize], nl: NonLinearity) -> Self {
        assert!(sizes.len() > 0);
        let mut input = input_dim;
        let mut layers = Vec::with_capacity(sizes.len());
        for i in 0..sizes.len() {
            let lnl = if i + 1 == sizes.len() {
                NonLinearity::Linear
            } else {
                nl
            };
            let s = sizes[i];
            layers.push(Layer::new(input, s, lnl));
            input = s;
        }

        let max_dim = layers.iter().map(|l| l.bias.len()).max().unwrap();

        Network {
            layers: layers,
            max_dim: max_dim,
        }
    }
}

impl WeightDecay for Network {
    #[inline]
    fn l2norm(&self) -> f32 {
        self.layers.iter().map(|l| l.l2norm()).sum()
    }
}

// Evaluates the neural net, layer by layer, writing it into the output array
fn evaluate<P, F>(nn: &Network, payload: &P, input: &mut [f32], output: &mut [f32], f: F) -> f32
where
    F: Fn(&Layer, &P, &mut [f32]) -> (),
{
    for i in 0..nn.layers.len() {
        let (input_dim, output_dim) = nn.layers[i].dims();
        if i == 0 {
            f(&nn.layers[i], payload, &mut input[0..output_dim]);
        } else if i % 2 == 1 {
            nn.layers[i].eval(&input[0..input_dim], &mut output[0..output_dim]);
        } else {
            nn.layers[i].eval(&output[0..input_dim], &mut input[0..output_dim]);
        }
    }

    if nn.layers.len() % 2 == 1 {
        input[0]
    } else {
        output[0]
    }
}

// Evaluates a dense layer
fn dense_eval(l: &Layer, p: &Vec<f32>, out: &mut [f32]) -> () {
    l.eval(&p, out)
}

impl Evaluator<Vec<f32>, f32> for Network {
    fn evaluate(&self, payload: &Vec<f32>) -> f32 {
        if self.max_dim < 100 {
            let mut input: [f32; 100] = unsafe { mem::MaybeUninit::uninit().assume_init() };
            let mut output: [f32; 100] = unsafe { mem::MaybeUninit::uninit().assume_init() };
            evaluate(&self, payload, &mut input, &mut output, dense_eval)
        } else {
            let mut input = vec![0f32; self.max_dim];
            let mut output = vec![0f32; self.max_dim];
            evaluate(&self, payload, &mut input, &mut output, dense_eval)
        }
    }
}

// Evaluates a sparse layer
fn sparse_eval(l: &Layer, p: &Sparse, out: &mut [f32]) -> () {
    l.sparse_eval(&p, out)
}

impl Evaluator<Sparse, f32> for Network {
    fn evaluate(&self, payload: &Sparse) -> f32 {
        if self.max_dim < 100 {
            let mut input: [f32; 100] = unsafe { mem::MaybeUninit::uninit().assume_init() };
            let mut output: [f32; 100] = unsafe { mem::MaybeUninit::uninit().assume_init() };
            evaluate(&self, payload, &mut input, &mut output, sparse_eval)
        } else {
            let mut input = vec![0f32; self.max_dim];
            let mut output = vec![0f32; self.max_dim];
            evaluate(&self, payload, &mut input, &mut output, sparse_eval)
        }
    }
}

impl WeightUpdater for Network {
    fn num_weights(&self) -> usize {
        self.layers.iter().map(|l| l.num_weights()).sum()
    }

    fn update_gradients<F>(&mut self, f: &mut F) -> ()
    where
        F: FnMut() -> f32,
    {
        for l in self.layers.iter_mut() {
            l.update_gradients(f);
        }
    }

    fn scale_gradients(&mut self, s: f32) -> () {
        for l in self.layers.iter_mut() {
            l.scale_gradients(s);
        }
    }

    fn copy_gradients(&self, other: &mut Self) {
        assert_eq!(self.num_weights(), other.num_weights());
        for i in 0..self.layers.len() {
            self.layers[i].copy_gradients(&mut other.layers[i]);
        }
    }

    fn add_gradients(&mut self, other: &Self) {
        assert_eq!(self.num_weights(), other.num_weights());
        for i in 0..self.layers.len() {
            self.layers[i].add_gradients(&other.layers[i]);
        }
    }
}

impl GradientFuser<Network> for Network {
    fn update(&self, grad: &Network, into: &mut Self) -> () {
        assert_eq!(self.num_weights(), grad.num_weights());
        assert_eq!(self.num_weights(), into.num_weights());
        for i in 0..self.layers.len() {
            self.layers[i].update(&grad.layers[i], &mut into.layers[i]);
        }
    }
}

impl Genes<f32> for Network {
    fn num_genes(&self) -> usize {
        self.num_weights()
    }

    fn get_gene(&self, mut idx: usize) -> f32 {
        for layer in self.layers.iter() {
            if idx >= layer.num_weights() {
                idx -= layer.num_weights();
            } else {
                // go through w first, then bias
                for w_i in layer.w.iter() {
                    if idx >= w_i.len() {
                        idx -= w_i.len();
                    } else {
                        return w_i[idx];
                    }
                }
                // gotta be in the bias
                return layer.bias[idx];
            }
        }
        panic!("Out of bounds for weights!");
    }

    fn set_gene(&mut self, mut idx: usize, value: f32) {
        // duplicate of the above code, sadly
        for layer in self.layers.iter_mut() {
            if idx >= layer.num_weights() {
                idx -= layer.num_weights();
            } else {
                // go through w first, then bias
                for w_i in layer.w.iter_mut() {
                    if idx >= w_i.len() {
                        idx -= w_i.len();
                    } else {
                        w_i[idx] = value;
                        return ();
                    }
                }
                // gotta be in the bias
                layer.bias[idx] = value;
                return ();
            }
        }
        panic!("Out of bounds for weights!");
    }
}

/// Representation of a neural network model
pub struct NNModel {
    /// Size of the imput dimensions
    input_dims: usize,
    /// Hidden layer dimensions
    hidden_dims: Vec<usize>,
    /// Type of non-linearity to apply
    nl: NonLinearity,
}

impl NNModel {
    /// Initializes a new neural network model. The underlying model is a Network
    pub fn new(input_dims: usize, hidden_dims: &[usize], nl: NonLinearity) -> Self {
        assert!(input_dims > 0);
        hidden_dims.iter().for_each(|d| assert!(*d > 0));
        NNModel {
            input_dims: input_dims,
            hidden_dims: hidden_dims.iter().map(|d| *d).collect(),
            nl: nl,
        }
    }

    /// Loads a network from a file
    pub fn load_from_network<A: Read>(reader: &mut A) -> Result<Self, SerDeErr> {
        let network = Network::load(reader)?;
        let input_dims = network.layers[0].w[0].len();
        let hidden_dims: Vec<_> = network.layers.iter().map(|layer| layer.w.len()).collect();
        let activation = network.layers[0].nl.clone();
        Ok(NNModel::new(input_dims, &hidden_dims, activation))
    }
}

impl Initializer for NNModel {
    type Model = Network;

    fn zero(&self) -> Self::Model {
        Network::new(self.input_dims, &self.hidden_dims, self.nl)
    }
}

#[derive(Debug)]
/// Error conditions due to writing the neural network
pub enum SerDeErr {
    /// Error when writing the json
    SerDeError(serde_json::Error),
}

impl SerDe for Network {
    type Error = SerDeErr;

    fn save<A: Write>(&self, writer: &mut A) -> Result<(), Self::Error> {
        serde_json::to_writer(writer, &self).map_err(|e| SerDeErr::SerDeError(e))
    }

    fn load<A: Read>(reader: &mut A) -> Result<Self, Self::Error> {
        serde_json::from_reader(reader).map_err(|e| SerDeErr::SerDeError(e))
    }
}

#[cfg(test)]
mod test_nn {
    use super::*;

    #[test]
    fn test_nn_genes() {
        let mut network = Network::new(5, &[2, 3], NonLinearity::ReLu);
        for idx in 0..network.num_genes() {
            network.set_gene(idx, idx as f32);
            assert_eq!(network.get_gene(idx), idx as f32);
        }

        let mut network = Network::new(4, &[1], NonLinearity::ReLu);
        for idx in 0..network.num_genes() {
            network.set_gene(idx, idx as f32);
            assert_eq!(network.get_gene(idx), idx as f32);
        }
    }
}

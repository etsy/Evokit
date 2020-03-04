//! Defines a linear model for use within Evokit
use es_core;
use es_data;

use self::es_core::evolution::Genes;
use self::es_core::model::{Evaluator, GradientFuser, Initializer, WeightUpdater};

use self::es_data::datatypes::{Dense, Sparse};
use self::es_data::intrinsics::{dot, l2norm, scale, sparse_dot, sum};

use super::*;
use crate::regularizer::WeightDecay;

#[derive(Serialize, Deserialize, Clone, Debug)]
/// Wrapper around type from external crate to define weight updates
pub struct DenseWrapper {
    /// Dense vector
    pub w: Dense,
}

// Linear Model prediction
impl Evaluator<Vec<f32>, f32> for DenseWrapper {
    fn evaluate(&self, payload: &Vec<f32>) -> f32 {
        dot(&self.w.0, payload)
    }
}

// Linear Model prediction
impl Evaluator<Sparse, f32> for DenseWrapper {
    fn evaluate(&self, payload: &Sparse) -> f32 {
        sparse_dot(&payload.1, &payload.2, &self.w.0)
    }
}

impl GradientFuser<Vec<f32>> for DenseWrapper {
    fn update(&self, grad: &Vec<f32>, into: &mut Self) -> () {
        sum(&self.w.0, grad, &mut into.w.0);
    }
}

impl WeightUpdater for DenseWrapper {
    #[inline]
    fn num_weights(&self) -> usize {
        self.w.0.len()
    }

    #[inline]
    fn update_gradients<F>(&mut self, f: &mut F) -> ()
    where
        F: FnMut() -> f32,
    {
        update_vec(&mut self.w.0, f);
    }

    fn scale_gradients(&mut self, f: f32) -> () {
        scale(&mut self.w.0, f);
    }

    fn copy_gradients(&self, other: &mut Self) {
        assert_eq!(self.w.0.len(), other.w.0.len());
        copy_vec(&self.w.0, &mut other.w.0);
    }

    fn add_gradients(&mut self, other: &Self) {
        add_vec(&mut self.w.0, &other.w.0);
    }
}

impl WeightDecay for DenseWrapper {
    #[inline]
    fn l2norm(&self) -> f32 {
        l2norm(&self.w.0)
    }
}

impl Genes<f32> for DenseWrapper {
    fn num_genes(&self) -> usize {
        self.num_weights()
    }

    fn get_gene(&self, idx: usize) -> f32 {
        self.w.0[idx]
    }

    fn set_gene(&mut self, idx: usize, value: f32) {
        self.w.0[idx] = value;
    }
}

/// Struct to define the model parameters for linear model
pub struct LinearModel {
    /// Size of model
    size: usize,
}

impl LinearModel {
    /// Returns a new LinearModel
    pub fn new(size: usize) -> Self {
        LinearModel { size: size }
    }
}

impl Initializer for LinearModel {
    type Model = DenseWrapper;

    fn zero(&self) -> DenseWrapper {
        DenseWrapper {
            w: Dense(vec![0f32; self.size]),
        }
    }
}

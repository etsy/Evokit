//! Library defining different types of models
#![warn(missing_docs, unused)]

#[macro_use]
extern crate serde_derive;
use es_data;

/// Defines the weight decay methods
pub mod regularizer;

/// Defines linear models
pub mod linear;
/// Defines neural networks
pub mod nn;
/// Defines trees
pub mod trees;

use self::es_data::intrinsics::inplace_sum;

// We make proxy versions of standard updates since we can't directly
// implement the traits since they exist in another library
// This takes in a noise function which generates new values into a vector;
// we use it for the stochastic hill clibining within nn.rs and linear.rs
/// Method to update a vector in place
fn update_vec<F>(x: &mut [f32], f: &mut F)
where
    F: FnMut() -> f32,
{
    for e in x.iter_mut() {
        *e = f();
    }
}

/// Method to copy from one vector to another
fn copy_vec(from: &[f32], other: &mut [f32]) {
    assert_eq!(from.len(), other.len());
    for i in 0..from.len() {
        other[i] = from[i];
    }
}

/// Method to add two vectors in place
fn add_vec(into: &mut [f32], other: &[f32]) {
    inplace_sum(into, other);
}

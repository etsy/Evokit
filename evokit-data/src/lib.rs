//! Library for dataset methods
#![warn(missing_docs, unused)]

#[macro_use]
extern crate serde_derive;

/// Definitions of static, minibatch, and lazy
pub mod dataset;
/// Definitions of sparse and dense
pub mod datatypes;
/// Helper methods for intrinsics
pub mod intrinsics;
/// Defines methods for loading data from a file
pub mod load;

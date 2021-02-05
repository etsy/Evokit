//! Evokit
//!
//! Evokit is a library and an executable to train models using evolutionary strategies.
#![warn(missing_docs, unused)]

#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate clap;

/// Tools for binaries
pub mod bin_utils;
/// Example environment
pub mod example;
/// Defines interfaces for learning to rank functions
pub mod l2r;
/// Contains a set of optimization metrics
pub mod metrics;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

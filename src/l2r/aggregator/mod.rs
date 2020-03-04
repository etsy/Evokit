//! Aggregator
//! ---
//!
//! This module contains the structs for aggregating two vectors. This is used during BeamSearch
//! to summarize the previous listings seen
//!
//! Types of Aggregators:
//! - CountAgg
//! - TotalAgg
//! - MeanAgg
//! - VarianceAgg
//! - EMAAgg
//!
//! Currently, this is hard-coded in mulberry and mulberry-test. Please change the aggregator there.
//! Eventually we will support this through the policy config.

/// Util functions for aggregators
pub mod utils;

/// Aggregator for counts
pub mod count;

/// Aggregator for total
pub mod total;

/// Aggregator for mean
pub mod mean;

/// Aggregator for variance
pub mod variance;

/// Aggregator for EMA
pub mod ema;

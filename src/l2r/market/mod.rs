//! Market
//! ---
//!
//! This module contains the Market level indicators. These are computed across all requests unless
//! you specify a term to filter on.
//!
//! Types of indicators:
//! - Mean(String, Option<f32>): takes a string specifying the query metric and an optional f32
//!   to scale the final indicator by.
//! - Histogram(String, Vec<usize>): takes a string specifying the query metric and a list of usize
//!   specifying percentiles to compute.
//! - ChiSquared(String, usize): takes a string specifying the query metric and a usize specifying
//!   the number of classes
//! - Gini(String, String, String): takes the path to a file containing a population map and the
//!   names of the fields for wealth and category.
//! - AUC(String): takes the name of the indicator
//!
//! You'll add them to the scoring config after specifying query level metrics
//!
//! E.g.
//! ```json
//! {
//!     "name": "avg-price-q1",
//!     "weight": 0.0,
//!     "indicator": {
//!         "Histogram": ["avg-price-10", [25]]
//!     },
//!     "filter_config": {
//!         "MetricGreaterThan": ["is_purchase_request", 0.0]
//!     }
//! }
//! ```
//!
//! This config is for a Histogram indicator to compute the 25th percentile of "avg-price-10" across
//! queries where "is_purchase_request" is non-zero.
//!
//! How to add new indicators:
//! - Add it's required parameters to IndicatorParameters
//! - Update the From<&IndicatorParameters>
//! - Create the actual struct and implement the trait Indicator in a new file
use std::convert::From;

/// Base Indicators
pub mod utils;

/// Indicator for mean
pub mod mean;

/// Indicator for AUC
pub mod auc;

/// Indicator for histograms
pub mod histogram;

/// Indicator for gini
pub mod gini;

/// Indicate for chi squared
pub mod chi_squared;

use crate::l2r::market::auc::AUCIndicator;
use crate::l2r::market::chi_squared::ChiSquaredIndicator;
use crate::l2r::market::gini::GiniIndicator;
use crate::l2r::market::histogram::HistogramIndicator;
use crate::l2r::market::mean::MeanIndicator;
use crate::l2r::market::utils::*;

#[derive(Deserialize, Debug)]
/// Parameters for Indicators
pub enum IndicatorParameters {
    /// takes a string specifying the query metric and an optional f32 to scale the final indicator by.
    Mean(String, Option<f32>),
    /// takes a string specifying the query metric and a list of usize specifying percentiles to compute.
    Histogram(String, Vec<usize>),
    /// takes a string specifying the query metric and a usize specifying the number of classes
    ChiSquared(String, usize),
    /// takes the path to a file containing a population map and the names of the fields for wealth and category.
    Gini(String, String, String),
    /// takes the name of the indicator
    AUC(String),
}

impl From<&IndicatorParameters> for Box<dyn Indicator> {
    /// Converts an indicator parameter to an actual indicator
    fn from(ic: &IndicatorParameters) -> Box<dyn Indicator> {
        match ic {
            IndicatorParameters::Mean(f, scale) => Box::new(MeanIndicator::new(f, scale.clone())),
            IndicatorParameters::Histogram(f, p) => Box::new(HistogramIndicator::new(f, p)),
            IndicatorParameters::ChiSquared(f, c) => Box::new(ChiSquaredIndicator::new(f, *c)),
            IndicatorParameters::Gini(f_name, w, cat) => {
                Box::new(GiniIndicator::new(f_name, w, cat))
            }
            IndicatorParameters::AUC(f) => Box::new(AUCIndicator::new(f)),
        }
    }
}

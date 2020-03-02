//! Filters
//! ---
//!
//! This module contains the structs for filtering requests before computing market level indicators
//!
//! Types of filters:
//! - MetricGreaterThan(String, f32): takes a string for the query level metric to filter on and
//!   the value it needs to be greater than
//! - MetricEqualTo(String, f32): takes a string for the query level metric to filter on and the
//!   value it needs to be equal to
//! - MetricStringContains(String, String): takes a string for the query level metric to filter on
//!   and a string for the value it needs to contain
//!
//! These are provided when defining the scoring config.
//!
//! With filter:
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
//! Without filter:
//! ```json
//! {
//!     "name": "avg-price-q1",
//!     "weight": 0.0,
//!     "indicator": {
//!         "Histogram": ["avg-price-10", [25]]
//!     },
//!     "filter_config": null,
//! }
//! ```
use l2r::market::utils::{MetricType, Metrics};

/// Parameters for filtering requests
#[derive(Deserialize, Debug)]
pub enum FilterParameters {
    // TODO(DSCI-960): Consider merging these two filters
    /// Parameters for metric greater than
    MetricGreaterThan(GreaterThanFilterParameters),
    /// Parameters for metric equal to
    MetricEqualTo(EqualToFilterParameters),
    /// Parameters for metric string contains
    MetricStringContains(StringContainsFilterParameters),
}

impl From<&FilterParameters> for Box<dyn MetricsFilter> {
    /// Converts the parameters into actual structs we can use to filter requests
    fn from(fp: &FilterParameters) -> Box<dyn MetricsFilter> {
        match fp {
            FilterParameters::MetricGreaterThan(params) => Box::new(GreaterThanFilter::new(params)),
            FilterParameters::MetricEqualTo(params) => Box::new(EqualToFilter::new(params)),
            FilterParameters::MetricStringContains(params) => {
                Box::new(StringContainsFilter::new(params))
            }
        }
    }
}

/// Trait defining methods for filtering
pub trait MetricsFilter: Send + Sync {
    /// Filters the requests
    /// # Arguments
    ///
    /// * metrics: requests to filter
    fn filter<'a>(&'a self, metrics: &'a Vec<Metrics>) -> Vec<&'a Metrics>;
}

/// Wrapper to box a MetricsFilter
pub struct WrappedFilter(pub Box<dyn MetricsFilter>);

impl MetricsFilter for WrappedFilter {
    /// Calls the wrapped filter method
    fn filter<'a>(&'a self, metrics: &'a Vec<Metrics>) -> Vec<&'a Metrics> {
        self.0.filter(metrics)
    }
}

/// Parameters for defining a GreaterThanFilter
#[derive(Deserialize, Debug)]
pub struct GreaterThanFilterParameters {
    /// Name of the scorer
    pub scorer_name: String,
    /// Threshold to compare with
    pub threshold: f32,
}

/// Filter to exclude any requests with a value <= the provided value
#[derive(Clone, Debug)]
pub struct GreaterThanFilter {
    /// Name of the scorer
    pub scorer_name: String,
    /// Threshold to compare with
    pub threshold: f32,
}

impl GreaterThanFilter {
    /// Returns new GreaterThanFilter
    pub fn new(params: &GreaterThanFilterParameters) -> Self {
        GreaterThanFilter {
            scorer_name: params.scorer_name.to_string(),
            threshold: params.threshold,
        }
    }
}

impl MetricsFilter for GreaterThanFilter {
    /// Filters all requests to include just ones where x > value
    /// # Arguments
    ///
    /// * metrics: requests to filter
    fn filter<'a>(&'a self, metrics: &'a Vec<Metrics>) -> Vec<&'a Metrics> {
        metrics
            .into_iter()
            .filter(|m| m.read_num(&self.scorer_name) > self.threshold)
            .collect()
    }
}

/// Parameters for defining a EqualToFilter
#[derive(Deserialize, Debug)]
pub struct EqualToFilterParameters {
    /// Name of the scorer
    pub scorer_name: String,
    /// value to compare with
    pub value: MetricType,
}

/// Filter to exclude any requests with a value != the provided value
#[derive(Clone, Debug)]
pub struct EqualToFilter {
    /// Name of the scorer
    pub scorer_name: String,
    /// value to compare with
    pub value: MetricType,
}

impl EqualToFilter {
    /// Returns new EqualToFilter
    pub fn new(params: &EqualToFilterParameters) -> Self {
        EqualToFilter {
            scorer_name: params.scorer_name.to_string(),
            value: params.value.clone(),
        }
    }
}

impl MetricsFilter for EqualToFilter {
    /// Filters all requests to include just ones where x == value
    /// # Arguments
    ///
    /// * metrics: requests to filter
    fn filter<'a>(&'a self, metrics: &'a Vec<Metrics>) -> Vec<&'a Metrics> {
        metrics
            .into_iter()
            .filter(|m| *(m.read(&self.scorer_name)) == self.value)
            .collect()
    }
}

/// Parameters for defining a StringContainsFilter
#[derive(Deserialize, Debug)]
pub struct StringContainsFilterParameters {
    /// Name of the scorer field
    pub scorer_name: String,
    /// Value which we want to test if the string contains
    pub value: String,
}

/// Filter to exclude any requests that don't contain the given `value`
/// Assumes that the field type will be MetricType::Str
#[derive(Clone, Debug)]
pub struct StringContainsFilter {
    /// Name of the scorer field
    pub scorer_name: String,
    /// Value which we want to test if the string contains
    pub value: String,
}

impl StringContainsFilter {
    /// Returns new StringContainsFilter
    pub fn new(params: &StringContainsFilterParameters) -> Self {
        StringContainsFilter {
            scorer_name: params.scorer_name.to_string(),
            value: params.value.to_string(),
        }
    }
}

impl MetricsFilter for StringContainsFilter {
    /// Filters all requests to include just ones where the metric (assumed to be a String)
    /// contains `value`
    /// # Arguments
    ///
    /// * metrics: requests to filter
    fn filter<'a>(&'a self, metrics: &'a Vec<Metrics>) -> Vec<&'a Metrics> {
        metrics
            .into_iter()
            .filter(|m| m.read_str(&self.scorer_name).contains(&self.value))
            .collect()
    }
}

/// Filter to return everything
#[derive(Clone, Debug)]
pub struct NoopFilter;

impl NoopFilter {
    /// Returns a NoopFilter
    pub fn new() -> Self {
        NoopFilter
    }
}

impl MetricsFilter for NoopFilter {
    /// This filter doesn't do anything
    fn filter<'a>(&'a self, metrics: &'a Vec<Metrics>) -> Vec<&'a Metrics> {
        metrics.into_iter().collect()
    }
}

#[cfg(test)]
mod test_filters {
    use super::*;
    use std::collections::HashMap;

    fn make_single_map<T>(key: String, value: T) -> HashMap<String, T>
    where
        T: Clone,
    {
        // Helper fn that makes a HashMap with a single map
        // corresponding to the passed key-value pair
        let map: HashMap<String, T> = [(key, value)].iter().cloned().collect();
        return map;
    }

    fn make_metrics_vec<I>(key: &str, values: I) -> Vec<Metrics>
    where
        I: IntoIterator<Item = MetricType>,
    {
        // Helper fn that makes vector of metrics
        // Each metric element has same key, populated with passed values
        let mut metrics: Vec<Metrics> = Vec::new();
        for x in values {
            let mut metric = Metrics::new();
            metric.add_metric(key, x);
            metrics.push(metric);
        }
        return metrics;
    }

    #[test]
    fn test_equal_to_filter() {
        let filter_params = EqualToFilterParameters {
            scorer_name: "test".to_string(),
            value: MetricType::from(5.0),
        };
        let filter = EqualToFilter::new(&filter_params);
        let metrics = make_metrics_vec("test", (1..10).map(|x| MetricType::from(x as f32)));

        let filter_result = filter.filter(&metrics);
        let expected_result: Vec<Metrics> =
            vec![Metrics::from(make_single_map(String::from("test"), 5.0))];
        for (expected_metric, result_metric) in expected_result.iter().zip(filter_result.iter()) {
            assert_eq!(expected_metric.mapping, result_metric.mapping);
        }
    }

    #[test]
    fn test_greater_than_filter() {
        let filter_params = GreaterThanFilterParameters {
            scorer_name: "test".to_string(),
            threshold: 8.0,
        };
        let filter = GreaterThanFilter::new(&filter_params);
        let metrics = make_metrics_vec("test", (1..10).map(|x| MetricType::from(x as f32)));

        let filter_result = filter.filter(&metrics);
        let expected_result: Vec<Metrics> = vec![
            Metrics::from(make_single_map(String::from("test"), 9.0)),
            Metrics::from(make_single_map(String::from("test"), 10.0)),
        ];
        for (expected_metric, result_metric) in expected_result.iter().zip(filter_result.iter()) {
            assert_eq!(expected_metric.mapping, result_metric.mapping);
        }
    }

    #[test]
    fn test_string_contains_filter() {
        let filter_params = StringContainsFilterParameters {
            scorer_name: "test".to_string(),
            value: "GB".to_string(),
        };
        let filter = StringContainsFilter::new(&filter_params);
        let metrics = make_metrics_vec(
            "test",
            vec![
                MetricType::from("US"),
                MetricType::from("US,GB"),
                MetricType::from("GB"),
            ],
        );

        let filter_result = filter.filter(&metrics);
        let expected_result: Vec<Metrics> = vec![
            Metrics::from(make_single_map(String::from("test"), String::from("US,GB"))),
            Metrics::from(make_single_map(String::from("test"), String::from("GB"))),
        ];
        for (expected_metric, result_metric) in expected_result.iter().zip(filter_result.iter()) {
            assert_eq!(expected_metric.mapping, result_metric.mapping);
        }
    }
}

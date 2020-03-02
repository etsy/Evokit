extern crate es_data;

use self::es_data::dataset::types::MetaType;

use std::collections::HashMap;
use std::convert::From;
use std::sync::Arc;

#[derive(Deserialize, Clone, Debug, PartialEq)]
/// Types of metrics
pub enum MetricType {
    /// Numeric
    Num(f32),
    /// String
    Str(String),
}

impl From<f32> for MetricType {
    fn from(value: f32) -> Self {
        MetricType::Num(value)
    }
}

impl From<String> for MetricType {
    fn from(value: String) -> Self {
        MetricType::Str(value)
    }
}

impl From<&str> for MetricType {
    fn from(value: &str) -> Self {
        MetricType::Str(value.to_string())
    }
}

impl From<MetaType> for MetricType {
    fn from(value: MetaType) -> Self {
        match value {
            MetaType::Str(v) => MetricType::Str(v),
            MetaType::Num(v) => MetricType::Num(v),
        }
    }
}

/// Contains on the query-level scores for that request
pub struct Metrics {
    /// query-level scores
    pub mapping: HashMap<String, MetricType>,
    /// labels for all the documents in the set
    pub labels: Option<Arc<Vec<f32>>>,
    /// if provided, scores from the underlying model from the policy that was used
    pub scores: Option<Vec<f32>>,
}

impl Metrics {
    /// Create a new Metric
    pub fn new() -> Self {
        Metrics {
            mapping: HashMap::new(),
            labels: None,
            scores: None,
        }
    }

    /// Adds a new query level metric
    pub fn add_metric(&mut self, name: &str, value: MetricType) -> () {
        self.mapping.insert(name.into(), value);
    }

    /// Returns the requested query-level metric
    pub fn read(&self, name: &str) -> &MetricType {
        self.mapping
            .get(name)
            .expect(&format!("Missing metric: {}", name))
    }

    /// Returns the requested query-level metric if it is a number, else panics
    pub fn read_num(&self, name: &str) -> f32 {
        let raw_value = self
            .mapping
            .get(name)
            .expect(&format!("Missing metric: {}", name));
        match raw_value {
            MetricType::Num(num) => *num,
            _ => panic!(format!(
                "Tried to read from {} but was not a MetricType::Num",
                name
            )),
        }
    }

    /// Returns the requested query-level metric if it is a string, else panics
    pub fn read_str(&self, name: &str) -> String {
        let raw_value = self
            .mapping
            .get(name)
            .expect(&format!("Missing metric: {}", name));
        match raw_value {
            MetricType::Str(string) => string.to_string(),
            _ => panic!(format!(
                "Tried to read from {} but was not a MetricType::Str",
                name
            )),
        }
    }
}

impl From<HashMap<String, MetricType>> for Metrics {
    /// Converts a HashMap to Metrics
    fn from(item: HashMap<String, MetricType>) -> Self {
        Metrics {
            mapping: item,
            labels: None,
            scores: None,
        }
    }
}

impl From<HashMap<String, f32>> for Metrics {
    /// Converts a HashMap to Metrics
    fn from(item: HashMap<String, f32>) -> Self {
        // Convert HashMap values from f32 to MetricType
        let mut new_item: HashMap<String, MetricType> = HashMap::new();
        for (key, value) in item {
            new_item.insert(key, MetricType::Num(value));
        }
        return Metrics {
            mapping: new_item,
            labels: None,
            scores: None,
        };
    }
}

impl From<HashMap<String, String>> for Metrics {
    /// Converts a HashMap to Metrics
    fn from(item: HashMap<String, String>) -> Self {
        // Convert HashMap values from String to MetricType
        let mut new_item: HashMap<String, MetricType> = HashMap::new();
        for (key, value) in item {
            new_item.insert(key, MetricType::Str(value));
        }
        return Metrics {
            mapping: new_item,
            labels: None,
            scores: None,
        };
    }
}

/// Trait to define all indicator methods
pub trait Indicator: Send + Sync {
    /// Computes a market level indicator
    /// # Arguments
    ///
    /// * scores: query level scores across all the queries. This set should already be filtered.
    fn evaluate(&self, scores: &Vec<&Metrics>) -> f32;

    /// Returns the name of this indicator. Should be unique as we use it for logging.
    fn name(&self) -> &str;
}

/// Struct to wrap any Indicator in a box
pub struct WrappedIndicator(pub Box<dyn Indicator>, pub String);

impl Indicator for WrappedIndicator {
    /// Computes the wrapped market level indicator
    fn evaluate(&self, scores: &Vec<&Metrics>) -> f32 {
        self.0.evaluate(scores)
    }

    /// Returns the name of the wrapped indicator
    fn name(&self) -> &str {
        &self.1
    }
}

/// Struct to wrap multiple indicators and weight them
pub struct WeightedIndicators {
    indicators: Vec<(f32, Box<dyn Indicator>)>,
}

impl WeightedIndicators {
    /// Returns an instance of WeightedIndicators
    pub fn new() -> Self {
        WeightedIndicators {
            indicators: Vec::new(),
        }
    }

    /// Add a new indicator to the set
    pub fn add_indicator<I: 'static + Indicator>(&mut self, weight: f32, indicator: I) -> () {
        self.indicators.push((weight, Box::new(indicator)));
    }
}

impl Indicator for WeightedIndicators {
    /// Evaluated all the wrapped indicators and aggregate them
    fn evaluate(&self, scores: &Vec<&Metrics>) -> f32 {
        let (mut agg_s, mut agg_w) = (0., 0.);
        for (w, i) in self.indicators.iter() {
            agg_s += w * i.evaluate(scores);
            agg_w += w;
        }

        agg_s / agg_w
    }

    /// Return the name
    fn name(&self) -> &str {
        "weighted"
    }
}

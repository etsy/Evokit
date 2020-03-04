extern crate es_data;
use std::collections::HashMap;
use std::fs;

use self::es_data::dataset::types::MetaType;

use l2r::utils::metatype_to_num;
use metrics::*;

use crate::l2r::market::utils::*;

/// Indicator to compute gini
pub struct GiniIndicator {
    /// Population map. Class -> Population
    population_map: HashMap<u32, f32>,
    /// Name of query level scorer with wealth information
    wealth_field: String,
    /// Name of query level scorer with category information
    category_field: String,
}

impl GiniIndicator {
    /// Returns a GiniIndicator
    ///
    /// # Arguments
    /// * population_file_path: path to file containing a population map
    /// * wealth_field: name of the query level scorer with the wealth information
    /// * category_field: name of the query level scorer with the category information
    pub fn new(population_file_path: &str, wealth_field: &str, category_field: &str) -> Self {
        let file_contents =
            fs::read_to_string(population_file_path).expect("Failure to read config file");
        let config: Vec<(MetaType, f32)> =
            serde_json::from_str(&file_contents).expect("JSON was not well-formatted");
        let mut population_map = HashMap::new();
        for (key, val) in config.iter() {
            let key_u32 = metatype_to_num(key) as u32;
            population_map.insert(key_u32, *val);
        }

        GiniIndicator {
            population_map: population_map,
            wealth_field: wealth_field.into(),
            category_field: category_field.into(),
        }
    }
}

impl Indicator for GiniIndicator {
    /// Computes Gini indicator
    fn evaluate(&self, metrics: &Vec<&Metrics>) -> f32 {
        let mut wealth_counts = HashMap::new();
        for metric in metrics.iter() {
            let category = metric.read_num(&self.category_field) as u32;

            // TODO: handle std::f32::MAX
            let wealth = metric.read_num(&self.wealth_field);
            let wealth_entry = wealth_counts.entry(category).or_insert(0.);
            *wealth_entry += wealth;
        }

        let mut income_and_population = Vec::new();
        for (category, population) in self.population_map.iter() {
            let wealth = wealth_counts.get(category).map(|x| *x).unwrap_or(0.);
            income_and_population.push((wealth, *population));
        }
        // TODO: what about items that aren't in the population_map (validation set)? Should we set a value for the population?

        get_gini_coefficient(&mut income_and_population)
    }

    /// Gets the name of the indicator
    fn name(&self) -> &str {
        &self.category_field
    }
}

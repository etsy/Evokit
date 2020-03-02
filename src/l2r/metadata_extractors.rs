//! Metadata Extractors
//! ---
//!
//! This module contains query level metadata extractors. These are distinct from scorers, as
//! scorers only generate floats, while these may generate any MetaType. Because of this, the
//! metadata is intended for use with Filters which can take MetricType input, not just floats.
//!
//! Types of metadata extractors:
//! - MetadataFieldExtractor(String, int): Takes a string for the field name of the document from
//!   which to extract a value, and an int for the index of the document list for which document to
//!   extract from
//!
//! These are provided when defining the scoring config. The metadata field of the scoring config
//! is optional.
//!
//! ```json
//! {
//!     "name": "is_international_user_request",
//!     "extractor": {
//!         "MetadataFieldExtractor": {
//!             "field_name": "is_international_user_request",
//!             "k": 0
//!         }
//!     }
//! }
//! ```

extern crate es_data;

use self::es_data::dataset::types::{MetaType, Metadata};
use self::es_data::dataset::Grouping;
use self::es_data::datatypes::Sparse;

/// Parameters for extracting metadata
#[derive(Deserialize, Debug)]
pub enum MetadataExtractorParameters {
    /// Parameters for metadata field extractor
    MetadataFieldExtractor(MetadataFieldExtractorParameters),
}

impl From<&MetadataExtractorParameters> for Box<dyn MetadataExtractor> {
    /// Converts the parameters into actual struct
    fn from(params: &MetadataExtractorParameters) -> Box<dyn MetadataExtractor> {
        match params {
            MetadataExtractorParameters::MetadataFieldExtractor(params) => {
                Box::new(MetadataFieldExtractor::new(params))
            }
        }
    }
}

/// Trait defining methods for metadata extraction
pub trait MetadataExtractor: Send + Sync {
    /// Extracts metadata from the data
    /// # Arguments
    ///
    /// * document_set: The data from which to extract metadata
    ///                 a struct with 'metadata' field containing a Vec<Metadata>
    fn extract<'a>(&'a self, document_set: &Grouping<Sparse>) -> (&str, MetaType);
}

/// Parameters for defining a MetadataFieldExtractor
#[derive(Deserialize, Debug)]
pub struct MetadataFieldExtractorParameters {
    /// Name of the field in document from which to extract a value
    field_name: String,
    /// Which document in the list of documents to extract the value from
    k: usize,
    /// A default value to output if the given document field doesn't exist, rather than panicing
    default_value: Option<MetaType>,
}

///
#[derive(Clone, Debug)]
pub struct MetadataFieldExtractor {
    /// Name of the field in document from which to extract a value
    field_name: String,
    /// Which document in the list of documents to extract the value from
    k: usize,
    /// A default value to output if the given document field doesn't exist, rather than panicing
    default_value: Option<MetaType>,
}

impl MetadataFieldExtractor {
    /// Create a new MetadataFieldExtractor
    pub fn new(params: &MetadataFieldExtractorParameters) -> Self {
        MetadataFieldExtractor {
            field_name: params.field_name.to_string(),
            k: params.k,
            default_value: params.default_value.clone(),
        }
    }
}

impl MetadataExtractor for MetadataFieldExtractor {
    /// Extracts metadata directly from a given field in a given document in the list
    /// # Arguments
    ///
    /// * document_set: The data from which to extract metadata
    ///                 a struct with 'metadata' field containing a Vec<Metadata>
    fn extract<'a>(&'a self, document_set: &Grouping<Sparse>) -> (&str, MetaType) {
        // Get metadata out of passed data
        let metadata: &Vec<Metadata> = &document_set.metadata;

        // Check the inputs
        if self.k >= metadata.len() {
            panic!(format!(
                "There are fewer than k={} documents in the passed data. \
                Cannot extract from kth document.",
                self.k
            ))
        }
        let extracted_value: Option<&MetaType> = metadata[self.k].get(&self.field_name);
        if extracted_value.is_none() && self.default_value.is_none() {
            panic!(format!(
                "The field_name '{}' is not defined in passed data, and no default value was specified.",
                self.field_name
            ))
        }
        let return_value: &MetaType = match extracted_value {
            Some(value) => value,
            _ => self.default_value.as_ref().unwrap(),
        };

        (&self.field_name, return_value.clone())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_field_extractor() {
        let params: MetadataFieldExtractorParameters = MetadataFieldExtractorParameters {
            field_name: "test_field".to_string(),
            k: 0,
            default_value: None,
        };
        let extractor: MetadataFieldExtractor = MetadataFieldExtractor::new(&params);

        let x: Vec<Sparse> = vec![];
        let y: Vec<f32> = vec![];
        let metadata: Vec<Metadata> = vec![[
            ("test_field".to_string(), MetaType::Num(42.0)),
            (
                "other_field".to_string(),
                MetaType::Str("stuff".to_string()),
            ),
        ]
        .iter()
        .cloned()
        .collect()];
        let data: Grouping<Sparse> = Grouping::new_with_md(x, y, metadata);

        let (_, result) = extractor.extract(&data);
        assert_eq!(result, MetaType::Num(42.0));
    }
}

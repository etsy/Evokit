extern crate es_data;

use self::es_data::dataset::types::MetaType;

#[derive(Deserialize, Debug)]
/// Scorer to compute the NDCG of the ranked set. This uses the relevance label provided
pub struct NDCGScoringParameters {
    /// K value to use. If none is provided, we will use the number of documents
    pub k: Option<usize>,
}

#[derive(Clone, Deserialize, Debug, Copy)]
/// Enum to indicate the desired optimization goal
pub enum OptimizationGoal {
    /// Aim to minimize the value. Smaller values -> higher scores
    Minimize,
    /// Aim to maximize the value. Larger values -> higher scores
    Maximize,
}

#[derive(Deserialize, Debug)]
/// Parameters for scorers that compute a value over the top K documents
pub struct AtKScoringParameters {
    /// K value to use. If none is provided, we will use the number of documents
    pub k: Option<usize>,
    /// Name of field containing the "label" to use
    pub field_name: String,
    /// Whether we should normalize the score to be [0,1]
    pub normalize: Option<bool>,
    /// Desired optimization goal
    pub opt_goal: Option<OptimizationGoal>,
}

#[derive(Clone, Deserialize, Debug)]
/// Buckets for Discrete ERRIA scorer
pub enum TopicBuckets {
    /// Buckets for numeric values
    NumericBuckets(Vec<f32>),
    /// Buckets for string values
    StringBuckets(Vec<String>),
}

#[derive(Deserialize, Debug)]
/// Parameters for discrete ERRIA
pub struct DiscreteERRIAScoringParameters {
    /// Field name containing the topic
    pub field_name: String,
    /// k value for ERR
    pub k: Option<usize>,
    /// Way to bucketize the topics
    pub buckets: TopicBuckets,
    /// Value to use when the topic is missing
    pub default_topic: Option<u32>,
}

#[derive(Deserialize, Debug)]
/// Parameters for the binary scorer
pub struct BinaryScoringParameters {
    /// Field name containing the value to compare with
    pub field_name: String,
    /// Value for comparison
    pub field_value: MetaType,
    /// Whether to switch from x > field_value to x <= field_value
    pub flip_comparator: bool,
    /// Used to push the doc to the top or bottom of the list
    pub goal_index: f32,
}

#[derive(Deserialize, Debug)]
/// Parameters for a threshold scorer
pub struct ThresholdScoringParameters {
    /// Field name containing the value to compare with
    pub field_name: String,
    /// Value for comparision
    pub field_value: MetaType,
    /// Whether to switch from x > field_value to x <= field_value. Only matters for Num MetaType.
    pub flip_comparator: bool,
    /// Which document in the list to look at
    pub pos: usize,
}

#[derive(Deserialize, Debug)]
/// Parameters for field extractor
pub struct FieldExtractorParameters {
    /// Which field to extract
    pub field_name: String,
    /// Which doc to get the field from
    pub k: usize,
    /// Value to use if None exists
    pub sentinel: Option<f32>,
}

#[derive(Deserialize, Debug)]
/// Parameters for computing recall
pub struct RecallParameters {
    /// Field indicating the label
    pub field_name: String,
    /// @K threshold
    pub k: usize,
    /// Expected label value
    pub field_value: MetaType,
}

#[derive(Deserialize, Debug)]
/// Parameters for grouped AUC
pub struct GroupedAUCScoringParameters {
    /// Field containing the label
    pub field_name: Option<String>,
}

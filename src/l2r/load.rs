//! Load
//! ---
//!
//! This defines the methods to load the scoring config
extern crate hashbrown;
extern crate serde_json;

use std::fs;

use self::hashbrown::HashMap;

use super::scorer::{ScoringParameters, WeightedAggregateScorer};

use super::market::utils::{Indicator, WrappedIndicator};
use super::market::IndicatorParameters;

use super::anchor::DatasetEvaluator;
use super::filters::{FilterParameters, MetricsFilter, NoopFilter, WrappedFilter};
use super::metadata_extractors::{MetadataExtractor, MetadataExtractorParameters};

use super::aggregator::utils::AggBuilder;
use super::policy::beam::BeamSearchPolicy;
use super::policy::pointwise::PointwisePolicy;
use super::policy::random::RandomPolicy;
use super::policy::{PolicySet, PolicyWrapper};

#[derive(Deserialize, Debug)]
/// Metadata extractor config for a single request
pub struct MetadataExtractorConfig {
    /// Name of the extractor
    name: String,
    /// Specific extractor parameters
    extractor: MetadataExtractorParameters,
}

#[derive(Deserialize, Debug)]
/// Scoring config for a single request
pub struct ComponentScoringConfig {
    /// Name of the scorer
    name: String,
    /// Group if combining multiple scorers.
    group: String,
    /// Weight if combining multiple scorers
    weight: f32,
    /// Specific scoring parameters
    scorer: ScoringParameters,
}

#[derive(Deserialize, Debug)]
/// Config for indicators
struct IndicatorConfig {
    /// Name of the indicator
    name: String,
    /// Weight to be used when computing the fitness function
    weight: f32,
    /// Indicator specific parameters
    indicator: IndicatorParameters,
    /// Any filters to be applied to the requests before computing the indicator
    // TODO: make this a vector so we can support multiple filters
    filter_config: Option<FilterParameters>,
}

#[derive(Deserialize, Debug)]
/// Config for beam policy
struct BeamConfig {
    /// How many docs to subset before running the search
    subset: Option<usize>,
    /// Seed
    seed: Option<u32>,
    /// Number of candidates to explore
    num_candidates: Option<usize>,
}

#[derive(Deserialize, Debug)]
/// Config for random policy
struct RandomConfig {
    /// Seed
    seed: Option<u32>,
}

#[derive(Deserialize, Debug)]
/// Enum for different types of policies
enum PolicyConfig {
    /// Beam config
    Beam(BeamConfig),
    /// Pointwise, no args
    Pointwise,
    /// Random
    Random(RandomConfig),
}

#[derive(Deserialize, Debug, Clone)]
/// Config for last past boost
pub struct BoostConfig {
    /// Which field to boost using
    pub field_name: String,
    /// Threshold to compare with
    pub greater_than_threshold: f32,
    /// Number of docs to move to the top of the list
    pub max_number_to_boost: usize,
}

#[derive(Deserialize, Debug, Clone)]
/// Enum indicating which last past to use
pub enum LastPassConfig {
    /// Boost
    Boost(BoostConfig),
}

#[derive(Deserialize, Debug)]
/// Full config to compute across all requests
struct Config {
    /// Configs for the query level metadata extractors
    metadata: Option<Vec<MetadataExtractorConfig>>,
    /// Configs for the query level scorers
    query: Vec<ComponentScoringConfig>,
    /// Configs for the market level indicators
    indicators: Vec<IndicatorConfig>,
    /// Config for the policy
    policy: PolicyConfig,
    /// Optional config for last pass
    // TODO: should this be a vector
    last_pass: Option<LastPassConfig>,
}

#[derive(Deserialize, Debug)]
/// Config for a train, valid, & test
pub struct OverallScoringConfig {
    /// Train config
    train: Config,
    /// Valid config
    validation: Config,
    /// Optional test config. If none is provided, test is skipped
    test: Option<Config>,
}

/// Output of config loader. Defines the setup of a run
pub struct AnchorSetup<B> {
    /// Which policy to use
    pub policy: PolicySet<B>,
    /// How to evaluate a dataset. Captures the query and market level scorers
    pub evaluator: DatasetEvaluator,
}

/// Set of scorers & names
type GroupedScorers = HashMap<String, WeightedAggregateScorer>;

/// Converts scoring configs to actual scorers
/// # Arguments
///
/// * component_configs: List of configs to convert
fn create_weighted_scorer(component_configs: &[ComponentScoringConfig]) -> GroupedScorers {
    let mut groups = HashMap::new();
    for config in component_configs {
        let name = config.name.clone();
        let group_name = config.group.clone();
        let group = groups
            .entry(group_name)
            .or_insert_with(|| WeightedAggregateScorer::new());

        group.add_scorer(name, config.weight, &config.scorer);
    }
    println!(
        "{}",
        format!("Num scorers added: {:?}", component_configs.len())
    );
    groups
}

/// Adds metadata config to an evaluator, if there is one provided
/// # Arguments
///
/// * evaluator: Current dataset evaluator
/// * metadata: Option for configs for the metadata extractors
fn add_metadata(
    evaluator: &mut DatasetEvaluator,
    metadata_confs: &Option<Vec<MetadataExtractorConfig>>,
) -> () {
    if let Some(ref confs) = metadata_confs {
        for extractor in confs.iter() {
            let params: &MetadataExtractorParameters = &extractor.extractor;
            let boxed_extractor: Box<dyn MetadataExtractor> = params.into();
            evaluator.add_metadata_extractor(boxed_extractor);
        }
    }
}

/// Given an evaluator, adds the necessary market indicators
/// # Arguments
///
/// * evaluator: Current dataset evaluator
/// * indicators: configs for the indicators
fn add_indicators(evaluator: &mut DatasetEvaluator, indicators: &[IndicatorConfig]) -> () {
    for ind in indicators.iter() {
        let ip: &IndicatorParameters = &ind.indicator;
        let bi: Box<dyn Indicator> = ip.into();
        let metrics_filter: Box<dyn MetricsFilter> = if let Some(filter_config) = &ind.filter_config
        {
            filter_config.into()
        } else {
            Box::new(NoopFilter::new())
        };
        evaluator.add_market_indicator(
            ind.weight,
            WrappedIndicator(bi, ind.name.clone()),
            WrappedFilter(metrics_filter),
        );
    }
    println!(
        "{}",
        format!("Num indicators added: {:?}", indicators.len())
    );
}

/// Given a policy config, actually initializes the necessary policy
/// # Arguments
///
/// * config: defines the policy parameters
/// * builder: Builder defining how to aggregate information for beam policy
/// * stochastic: whether to use a stochastic policy
/// * last_pass: config for any last pass to run
fn create_policy<B: AggBuilder + Clone>(
    config: &PolicyConfig,
    builder: &B,
    stochastic: bool,
    last_pass: &Option<LastPassConfig>,
) -> PolicySet<B> {
    match config {
        PolicyConfig::Beam(ref settings) => {
            let policy = BeamSearchPolicy::new(
                builder.clone(),
                settings.subset,
                settings.seed,
                stochastic,
                settings.num_candidates,
            );
            PolicySet {
                policy_wrapper: PolicyWrapper::BeamSearch(policy),
                last_pass: last_pass.clone(),
            }
        }
        PolicyConfig::Pointwise => {
            let policy = PointwisePolicy::new(stochastic);
            PolicySet {
                policy_wrapper: PolicyWrapper::Pointwise(policy),
                last_pass: last_pass.clone(),
            }
        }
        PolicyConfig::Random(ref settings) => {
            let policy = RandomPolicy::new(settings.seed);
            PolicySet {
                policy_wrapper: PolicyWrapper::Random(policy),
                last_pass: last_pass.clone(),
            }
        }
    }
}

/// Given a file path, this outputs the setup for all runs (train, valid, test)
///
/// # Arguments
/// * builder: Builder defining how to aggregate information for beam policy, if used
/// * fname: name of the file
/// * stochastic: whether to use a stochastic policy
pub fn read_config<B: AggBuilder + Clone>(
    builder: B,
    fname: String,
    stochastic: bool,
) -> Result<(AnchorSetup<B>, AnchorSetup<B>, AnchorSetup<B>), String> {
    let file_contents =
        fs::read_to_string(fname.clone()).expect(&format!("Error reading config file: {}", fname));
    let config: OverallScoringConfig =
        serde_json::from_str(&file_contents).expect("scoring config JSON was not well-formatted");

    // Need to add assertion that fields aren't missing!
    let mut configs = vec![&config.train, &config.validation];
    if let Some(ref conf) = config.test {
        configs.push(&conf);
    } else {
        configs.push(&config.validation);
    }
    let mut out: Vec<_> = configs
        .into_iter()
        .map(|conf| {
            let scorers = create_weighted_scorer(&conf.query);
            let mut evaluator = DatasetEvaluator::new();
            // Add groups into dataset
            for (group_name, was) in scorers.into_iter() {
                evaluator.add_query_scorer(&group_name, was);
            }
            // Add metadata
            add_metadata(&mut evaluator, &conf.metadata);
            // Add indicators
            add_indicators(&mut evaluator, &conf.indicators);

            // Build the policy
            let policy = create_policy(&conf.policy, &builder, stochastic, &conf.last_pass);
            AnchorSetup { evaluator, policy }
        })
        .collect();

    let (test, valid, train) = (out.pop(), out.pop(), out.pop());

    // Guaranteed to be there
    Ok((train.unwrap(), valid.unwrap(), test.unwrap()))
}

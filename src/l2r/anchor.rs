//! Defines interfaces for learning to rank functions
extern crate es_core;
extern crate es_data;
extern crate es_models;
extern crate rayon;

use self::es_core::model::Evaluator;
use self::es_core::optimizer::{Environment, ScoreLogger};

use self::es_models::regularizer::WeightDecay;

use self::es_data::dataset::types::MetaType;
use self::es_data::dataset::{Dataset, DatasetItem, EagerIterator};
use self::es_data::datatypes::Sparse;

use std::borrow::Borrow;

use std::fs::File;
use std::io::Write;
use std::marker::PhantomData;

use self::rayon::prelude::*;

use l2r::filters::MetricsFilter;
use l2r::market::utils::{Indicator, MetricType, Metrics};
use l2r::metadata_extractors::MetadataExtractor;
use l2r::policy::utils::Policy;
use l2r::scorer::utils::{BScorer, NamedScorer, Scorer};

/// Struct to define the query and market level scorers
pub struct DatasetEvaluator {
    metadata: Vec<Box<dyn MetadataExtractor>>,
    query: Vec<NamedScorer<BScorer>>,
    indicator: Vec<(f32, Box<dyn Indicator>, Box<dyn MetricsFilter>)>,
}

impl DatasetEvaluator {
    /// returns a new DatasetEvaluator
    pub fn new() -> Self {
        DatasetEvaluator {
            metadata: Vec::new(),
            query: Vec::new(),
            indicator: Vec::new(),
        }
    }

    /// Adds a new query level metadata extractor
    pub fn add_metadata_extractor(&mut self, extractor: Box<dyn MetadataExtractor>) {
        self.metadata.push(extractor);
    }

    /// Adds a new query level scorer
    pub fn add_query_scorer<S: 'static + Scorer>(&mut self, name: &str, scorer: S) {
        self.query
            .push(NamedScorer::new(name.into(), BScorer(Box::new(scorer))));
    }

    /// Adds a new market level indicator
    pub fn add_market_indicator<I: 'static + Indicator, F: 'static + MetricsFilter>(
        &mut self,
        weight: f32,
        indicator: I,
        filter: F,
    ) {
        self.indicator
            .push((weight, Box::new(indicator), Box::new(filter)));
    }
}

/// Environment for Mulberry ES
pub struct AnchorLtrEnv<TS, VS, M, P> {
    /// Training data
    pub train: Vec<(f32, TS)>,
    /// Optional validation data
    pub valid: Option<VS>,
    /// Weight decay
    pub weight_decay: f32,
    /// Number of steps
    pub count: usize,
    /// Policy to order training instances
    pub train_policy: P,
    /// Policy to order validation instances
    pub valid_policy: P,
    /// Evaluator for training set
    pub train_ev: DatasetEvaluator,
    /// Evaluator for validation set
    pub valid_ev: DatasetEvaluator,
    /// Allows the struct to act like it owns something of type M (even though it doesn't). This is used for the evaluator
    pub pd: PhantomData<M>,
}

impl<TS, VS, M, P> AnchorLtrEnv<TS, VS, M, P> {
    /// Returns a new AnchorLtrEnv
    pub fn new(
        train: Vec<(f32, TS)>,
        v: Option<VS>,
        weight_decay: f32,
        train_policy: P,
        valid_policy: P,
        train_ev: DatasetEvaluator,
        valid_ev: DatasetEvaluator,
    ) -> Self {
        AnchorLtrEnv {
            train,
            valid: v,
            weight_decay,
            count: 1,
            train_policy,
            valid_policy,
            train_ev,
            valid_ev,
            pd: PhantomData,
        }
    }
}

// Ranking Environment
impl<TS, VS, M, P> Environment<M> for AnchorLtrEnv<TS, VS, M, P>
where
    TS: Dataset<Sparse, Iter = EagerIterator<Sparse>> + Send + Sync,
    VS: Dataset<Sparse, Iter = EagerIterator<Sparse>> + Send + Sync,
    M: Evaluator<Sparse, f32> + WeightDecay + Send + Sync,
    P: Policy,
{
    /// Updates the environment with the provided
    /// `true` indicates the environment is stochastic
    fn step(&mut self) -> bool {
        let mut shuffle = false;
        for &mut (_, ref mut d) in self.train.iter_mut() {
            shuffle |= d.shuffle();
        }
        self.count += 1;
        shuffle | self.train_policy.is_stochastic() | self.valid_policy.is_stochastic()
    }

    /// Evaluates a given state, returning the fitness.
    /// Higher is better.
    fn eval(&self, state: &M) -> (f32, Option<ScoreLogger>) {
        // Compute weight decay
        let wd = if self.weight_decay > 0.0 {
            self.weight_decay * state.l2norm()
        } else {
            0.0
        };

        let mut score = 0.0;
        let mut weight = 0.0;
        let mut logger: ScoreLogger = ScoreLogger::new(None);
        for (_i, &(w, ref dataset)) in self.train.iter().enumerate() {
            let (s, l) = eval_rs(
                dataset.data(),
                state,
                &self.train_policy,
                &self.train_ev,
                self.count,
                None,
            );
            score += w * s;
            logger.update(l, Some(w));
            weight += w;
        }
        let scaling_fn = |x: f32| x / weight - wd;
        logger.apply(&scaling_fn);
        (scaling_fn(score), Some(logger))
    }

    /// Evaluates a given state, potentially against
    /// Validation data.  If there is no validation,
    /// returns None
    fn validate(&self, state: &M) -> (Option<f32>, Option<ScoreLogger>) {
        if let Some(ref rs) = self.valid {
            let (s, l) = eval_rs(
                rs.data(),
                state,
                &self.valid_policy,
                &self.valid_ev,
                self.count,
                None,
            );
            (Some(s), l)
        } else {
            (None, None)
        }
    }
}

/// Computes score for a vecset.
pub fn eval_rs<
    M: Evaluator<Sparse, f32>,
    P: Policy,
    I: ParallelIterator<Item = DatasetItem<Sparse>>,
>(
    rs_vec: I,
    state: &M,
    policy: &P,
    ev: &DatasetEvaluator,
    seed: usize,
    output_file_opt: Option<String>,
) -> (f32, Option<ScoreLogger>) {
    // Compute the query level metrics and accrue their loggers
    let (metrics, _loggers): (Vec<Metrics>, Vec<ScoreLogger>) = rs_vec
        .map(|(idx, rs)| {
            // Grab all the indices as ordered by the policy
            let (indices, scores_opt) = policy.evaluate(state, rs.borrow(), seed as u32, idx);

            // Loop over all query evaluators and aggregate them into a hashmap
            let mut metrics = Metrics::new();
            let mut logger: ScoreLogger = ScoreLogger::new(None);
            for metadata_extractor in ev.metadata.iter() {
                let (name, value): (&str, MetaType) = metadata_extractor.extract(rs.borrow());
                metrics.add_metric(name.into(), MetricType::from(value));
            }
            for named_scorer in ev.query.iter() {
                let (name, score, n_logger): (&str, f32, Option<ScoreLogger>) =
                    named_scorer.score(rs.borrow(), &indices, &scores_opt);
                metrics.add_metric(name.into(), MetricType::Num(score));
                logger.update(n_logger, None);
            }
            if let Some(scores) = scores_opt {
                metrics.labels = Some(rs.y.clone());
                metrics.scores = Some(scores);
            }
            (metrics, logger)
        })
        .unzip();

    if let Some(output_file) = output_file_opt {
        println!("{}", format!("Writing to file: {}", output_file));
        let mut f = File::create(output_file).expect("Unable to create file");
        for metric in &metrics {
            write!(f, "{:?}\n", metric.mapping).expect("Failed writing metrics to file");
        }
    }

    // Run them through market level indicators
    let mut weighted_scores = 0.0;
    let mut weights = 0.0;
    let mut logger: ScoreLogger = ScoreLogger::new(None);
    for (weight, indicator, metrics_filter) in ev.indicator.iter() {
        let filtered_metrics = &metrics_filter.filter(&metrics);
        let score = indicator.evaluate(filtered_metrics);
        logger.insert(indicator.name().to_string(), score);
        weighted_scores += weight * score;
        weights += weight;
    }

    (weighted_scores / weights, Some(logger))
}

extern crate hashbrown;
use model::WeightUpdater;
use std::fmt::Debug;

use self::hashbrown::HashMap;

#[derive(Clone)]
/// Logger for outputting scores
pub struct ScoreLogger {
    /// Score name to score value
    counts: HashMap<String, f32>,
}

impl ScoreLogger {
    /// Returns a new ScoreLogger
    pub fn new(counts_opt: Option<HashMap<String, f32>>) -> ScoreLogger {
        match counts_opt {
            Some(counts) => ScoreLogger { counts: counts },
            None => ScoreLogger {
                counts: HashMap::new(),
            },
        }
    }

    #[inline]
    /// Adds a score
    pub fn insert(&mut self, key: String, value: f32) -> () {
        self.counts.insert(key, value);
    }

    #[inline]
    /// Gets a score
    pub fn get(&self, key: &str) -> Option<f32> {
        self.counts.get(key).map(|x| *x)
    }

    /// Adds all the scores from the source logger to this one
    pub fn update(
        &mut self,
        source_logger_opt: Option<ScoreLogger>,
        weight_opt: Option<f32>,
    ) -> () {
        if let Some(source_logger) = source_logger_opt {
            let weight = weight_opt.unwrap_or(1.0);
            for (k, v) in source_logger.counts.into_iter() {
                let e = self.counts.entry(k).or_insert(0.);
                *e += weight * v;
            }
        }
    }

    /// Iterates over the logged scores
    pub fn iter(&self) -> impl Iterator<Item = (&String, &f32)> {
        self.counts.iter()
    }

    /// Applies a method to all score values
    pub fn apply(&mut self, function: &dyn Fn(f32) -> f32) {
        for val in self.counts.values_mut() {
            *val = function(*val);
        }
    }
}

impl Debug for ScoreLogger {
    /// Prints the scores
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // sort counts
        let mut data: Vec<(&String, &f32)> = self.counts.iter().collect();
        data.sort_by_key(|(k, _v)| *k);
        write!(f, "{:?}", data)
    }
}

#[derive(Clone)]
/// Current state
pub struct State<S: Clone> {
    /// Underlying model used by the policy
    pub model: S,
    /// Latest fitness computed
    pub fitness: f32,
    /// The logged scores from the last run
    pub logger: Option<ScoreLogger>,
}

impl<S: Clone> State<S> {
    /// Returns a new state
    pub fn new(model: S, fitness: f32, logger: Option<ScoreLogger>) -> Self {
        State {
            model: model,
            fitness: fitness,
            logger: logger,
        }
    }
}

/// Fitness function which evaluates state
pub trait Environment<State>: Send + Sync {
    /// Updates the environment with the provided
    /// `true` indicates the environment is stochastic
    fn step(&mut self) -> bool;

    /// Evaluates a given state, returning the fitness.  
    /// Higher is better.
    fn eval(&self, s: &State) -> (f32, Option<ScoreLogger>);

    /// Evaluates a given state, potentially against
    /// Validation data.  If there is no validation,
    /// returns None
    fn validate(&self, _s: &State) -> (Option<f32>, Option<ScoreLogger>) {
        (None, None)
    }
}

/// Computes a new gradient given a state
pub trait GradientSampler<State: Clone + Send, E: Environment<State>>: Send + Sync {
    /// Gradient type used to compute children
    type Gradient: Clone + Send + Sync;

    /// Produces a new gradient type representing Zero.  This should hold that a model
    /// + zero gradient == the original model.
    fn zero_gradient(&self) -> Self::Gradient;

    /// Loads a new gradient, given a state
    fn generate(&mut self, env: &E, s: &State, g: &mut Self::Gradient, seed: usize) -> ();

    /// Applies a gradient to a state
    fn apply(&self, s: &State, g: &Self::Gradient, ns: &mut State) -> ();
}

/// Train to define optimizer methods
pub trait Optimizer {
    /// Runs the optimizer.  The particular details of how each optimizer runs is
    /// documented within the individual methods.  Largely, however, each optimizer
    /// expects the following: an initial model state, an environment that can evaluate
    /// states and emit a fitness score for that state, and a gradient sampler which
    /// can produce new gradients for evaluation within ES optimizers.
    fn run<S: Clone + Send + Sync, E, GS>(&self, init: S, env: &mut E, gs: &mut GS) -> State<S>
    where
        E: Environment<S>,
        GS: GradientSampler<S, E>,
        GS::Gradient: WeightUpdater + Send;
}

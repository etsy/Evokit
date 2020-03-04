extern crate rayon;

use std::clone::Clone;
use std::cmp::Ordering::Equal;
use std::f32;
use std::mem;
use std::time::SystemTime;

use self::rayon::prelude::*;

use model::WeightUpdater;
use optimizer::*;

// Precomputes the weights post fitness shaping
fn build_weights(size: usize) -> Vec<f32> {
    let fsize = size as f32;
    let noms: Vec<f32> = (1..(size + 1))
        .map(|i| (fsize + 0.5).ln() - (i as f32).ln())
        .collect();

    let denom: f32 = noms.iter().sum();
    noms.into_iter().map(|x| x / denom).collect()
}

/// Settings for simple ES
#[derive(Debug, Clone, PartialEq)]
pub struct Canonical {
    /// Number of top candidates to blend into the next generation
    pub parents: usize,

    /// Number of candidates to generate
    pub children: usize,

    /// Number of iterations to run the optimizer
    pub iterations: usize,

    /// How often to report optimization metrics
    pub report_iter: usize,

    /// When selected, will re-attempt a successful gradient.
    pub elitism: bool,

    /// Instead of only updating the model when the gradient step improves the fitness
    /// score, instead always updates the model.  This is useful when the fitness function
    /// is stochastic.
    pub always_update: bool,
}

impl Optimizer for Canonical {
    fn run<S: Clone + Send + Sync, E, GS>(&self, init: S, env: &mut E, gs: &mut GS) -> State<S>
    where
        E: Environment<S>,
        GS: GradientSampler<S, E>,
        GS::Gradient: WeightUpdater,
    {
        // States
        let mut state_grads = vec![(gs.zero_gradient(), 0f32); self.children];
        let weights = build_weights(self.parents);

        // Create Initial state
        let mut tmp_state = init.clone();
        let mut parent = State {
            model: init,
            fitness: f32::NEG_INFINITY,
            logger: Some(ScoreLogger::new(None)),
        };
        let mut best = parent.clone();
        let mut progress = false;

        let now = SystemTime::now();
        let mut last_valid = 0f32;

        for pass in 0..(self.iterations) {
            // Let the environment update itself, if needed
            if env.step() {
                let score_and_logger = env.eval(&parent.model);
                parent.fitness = score_and_logger.0;
                parent.logger = score_and_logger.1.clone();
            }

            if pass % self.report_iter == 0 {
                let (s, m) = now
                    .elapsed()
                    .map(|e| (e.as_secs(), e.subsec_millis()))
                    .unwrap_or((0, 0));
                // Get the fitness
                println!(
                    "Time:{}.{:03},\tIteration: {},\tFitness: {},\tValid: {},\t\
                         LastValid: {},\tStats: {:?}",
                    s, m, pass, parent.fitness, best.fitness, last_valid, parent.logger
                );
            }

            if !progress || (progress && !self.elitism) || self.always_update {
                // If we haven't made any progress, we need to sample more gradients
                // and recombine with weights

                // For each child, generate a new gradient
                for &mut (ref mut g, _score) in state_grads.iter_mut() {
                    // Sample new gradient
                    gs.generate(&env, &parent.model, g, pass);
                }

                state_grads
                    .par_iter_mut()
                    .for_each(|(ref g, ref mut score)| {
                        // Create the model + gradient and evaluate
                        let mut tmp_state = parent.model.clone();
                        gs.apply(&parent.model, g, &mut tmp_state);

                        *score = env.eval(&tmp_state).0;
                    });

                // Sort them
                state_grads.sort_by(|x, y| (y.1).partial_cmp(&x.1).unwrap_or(Equal));

                // Recombinate with the above weights
                for (i, &mut (ref mut g, _)) in
                    state_grads.iter_mut().enumerate().take(self.parents)
                {
                    // Scale gradient
                    g.scale_gradients(weights[i]);
                }
            }

            let mut new_parent = parent.model.clone();
            for &(ref g, _) in state_grads.iter().take(self.parents) {
                // Add to parent
                gs.apply(&new_parent, &g, &mut tmp_state);
                mem::swap(&mut new_parent, &mut tmp_state);
            }

            // Test if we've improved
            progress = match env.eval(&new_parent) {
                // We've improved our evaluation metric, let's check
                // to see if we improved validation
                (new_fitness, ref new_logger)
                    if new_fitness > parent.fitness || self.always_update =>
                {
                    // Update parent fitness
                    mem::swap(&mut parent.model, &mut new_parent);
                    parent.fitness = new_fitness;
                    parent.logger = new_logger.clone();

                    // Check if validation should improve
                    match env.validate(&parent.model) {
                        (Some(score), ref logger) if score > best.fitness => {
                            best.model = parent.model.clone();
                            best.fitness = score;
                            best.logger = logger.clone();
                            last_valid = score;
                        }
                        (Some(score), _) => {
                            last_valid = score;
                        }
                        // Otherwise, just update it
                        (None, _) => {
                            best.model = parent.model.clone();
                            best.fitness = parent.fitness;
                            best.logger = parent.logger.clone();
                        }
                    }
                    !self.always_update
                }
                _ => false,
            }
        }
        best
    }
}

//! Optimizer for Natural Evolutionary Strategies
//!
//!

extern crate float_ord;
extern crate rayon;

use std::clone::Clone;
use std::f32;
use std::mem;
use std::time::SystemTime;

use self::float_ord::FloatOrd;
use self::rayon::prelude::*;

use model::WeightUpdater;
use optimizer::*;

// Contains the momentum vector
struct Momentum<G: WeightUpdater> {
    gradient: G,
    mu: f32,
}

impl<G: WeightUpdater> Momentum<G> {
    fn update(&mut self, new_gradient: &mut G) -> () {
        self.gradient.scale_gradients(self.mu);
        self.gradient.add_gradients(&new_gradient);
        self.gradient.copy_gradients(new_gradient);
    }
}

/// Settings for Natural ES
#[derive(Debug, Clone, PartialEq)]
pub struct Natural {
    /// Number of search gradients to compute for blending
    pub children: usize,

    /// Number of steps to take before exiting the optimizer
    pub iterations: usize,

    /// Number of iterations between reporting metrics
    pub report_iter: usize,

    /// When provided, uses momentum to estimate the second order derivative
    pub momentum: Option<f32>,

    /// Learning rate.  Usually just be set to 1.
    pub alpha: f32,

    /// When set, uses fitness shaping instead of gradient blending.  This should
    /// usually be set to true as it helps to escape early local optima.
    pub shape: bool,
}

// Takes a list of scores and, in effect, performs z-whitening, scaling
// the results by the learning rate.  alpha corresponds to the learning rate
fn normalize_weights(scores: &mut [f32], alpha: f32) -> () {
    let n_scores = scores.len() as f32;
    assert!(n_scores > 1f32);

    let sum: f32 = scores.iter().sum();
    let mu = sum / n_scores;
    // Compute stdev
    let var: f32 = scores.iter().map(|v| (v - mu).powi(2)).sum::<f32>() / n_scores;
    let std = var.sqrt();

    // If everything is the same, everything weighs the same
    if std == 0f32 {
        for s in scores.iter_mut() {
            *s = 1f32;
        }
    } else {
        // We add an error epsilon (1e-6) to avoid the case where standard deviation
        // is close to zero, resulting in an overflow.
        let denom = std + 1e-6;
        // Whiten
        for s in scores.iter_mut() {
            *s = alpha * (*s - mu) / denom;
        }
    }
}

// Reweights the scores based on rank-based fitness shaping
fn fitness_shape(scores: &mut Vec<f32>) -> () {
    let mut i_scores: Vec<_> = scores.iter().map(|x| x.clone()).enumerate().collect();
    i_scores.sort_by_key(|(_i, x)| FloatOrd(-*x));

    let len = scores.len();
    let log_len = (len as f32 / 2. + 1.).ln();
    let mut sum = 0.;
    for (mut rank, (i, _s)) in i_scores.into_iter().enumerate() {
        rank += 1;
        let nom = (0f32).max(log_len - (rank as f32).ln());
        scores[i] = nom;
        sum += nom;
    }
    for s in scores.iter_mut() {
        *s = *s / sum - 1. / len as f32;
    }
}

impl Optimizer for Natural {
    fn run<S, E, GS>(&self, init: S, env: &mut E, gs: &mut GS) -> State<S>
    where
        S: Clone + Send + Sync,
        E: Environment<S> + Send,
        GS: GradientSampler<S, E> + Send,
        GS::Gradient: WeightUpdater + Send,
    {
        // Build pool of gradients
        let mut gradients = vec![gs.zero_gradient(); self.children * 2];
        let mut tmp_gradient = gs.zero_gradient();
        let mut scores = vec![0f32; self.children * 2];

        // Create Initial state
        let mut par_update = init.clone();

        // Track last valid
        let score = env.validate(&init).0;
        let mut last_valid = score.unwrap_or(f32::NEG_INFINITY);
        let mut parent = State {
            model: init.clone(),
            fitness: last_valid,
            logger: Some(ScoreLogger::new(None)),
        };
        let mut best = parent.clone();

        // Momentum!
        let mut mom = self.momentum.map(|mu| Momentum {
            gradient: gs.zero_gradient(),
            mu: mu,
        });

        let now = SystemTime::now();

        for pass in 0..(self.iterations) {
            if pass % self.report_iter == 0 {
                let (s, m) = now
                    .elapsed()
                    .map(|e| (e.as_secs(), e.subsec_millis()))
                    .unwrap_or((0, 0));

                let (fitness, logger) = env.eval(&parent.model);
                parent.fitness = fitness;
                parent.logger = logger;
                println!(
                    "Time: {}.{:03},\tIteration: {},\tFitness: {},\t\
                          Valid: {},\tLast Valid: {},\tStats: {:?}",
                    s, m, pass, parent.fitness, best.fitness, last_valid, parent.logger
                );
            }

            // Let the environment update itself, if needed
            env.step();

            // Create new search gradients for each child
            for i in 0..self.children {
                let mirror_idx = i * 2;

                {
                    let mut g = &mut gradients[mirror_idx];

                    // Sample new gradient
                    gs.generate(&env, &parent.model, &mut g, pass);

                    // Since we can't mutate a borrowed refernce to a vector,
                    // we copy it here and apply it in the antithetic gradient
                    g.copy_gradients(&mut tmp_gradient);
                }

                // For the antithetic sample, we simple inverse it.
                let mut mg = &mut gradients[mirror_idx + 1];
                mem::swap(&mut tmp_gradient, &mut mg);

                mg.scale_gradients(-1f32);
            }

            // Loop over each gradient in parallel, using thread_local variables for
            // evaluation
            scores.par_iter_mut().enumerate().for_each(|(i, s)| {
                let mut tmp_parent = parent.model.clone();

                // Apply the gradient to the parent
                gs.apply(&parent.model, &gradients[i], &mut tmp_parent);

                // Evaluate the function
                *s = env.eval(&tmp_parent).0;
            });

            if self.shape {
                fitness_shape(&mut scores);
            } else {
                // Rescale the weights
                normalize_weights(&mut scores, self.alpha);
            }

            // Combine the deltas to produce the actual gradient
            let mut new_gradient = gs.zero_gradient();
            for i in 0..gradients.len() {
                // In reverse since we negative scaled
                let g = &mut gradients[i];
                g.scale_gradients(scores[i]);
                new_gradient.add_gradients(g);
            }

            // If momentum, update
            if let Some(ref mut m) = mom {
                m.update(&mut new_gradient);
            }

            // Update the old parent
            gs.apply(&parent.model, &new_gradient, &mut par_update);
            mem::swap(&mut parent.model, &mut par_update);

            // Check validation
            let (score, logger) = env.validate(&parent.model);
            match score {
                Some(score) if score > best.fitness => {
                    best.model = parent.model.clone();
                    best.fitness = score;
                    best.logger = logger;
                    last_valid = score;
                }
                Some(score) => {
                    last_valid = score;
                }
                None => {
                    best = parent.clone();
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn assert_vectors(expected: &[f32], actual: &[f32]) -> () {
        assert_eq!(expected.len(), actual.len());
        for (e, a) in expected.iter().zip(actual) {
            assert!((e - a).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fitness_shape() {
        let mut scores = vec![0f32, 0.1, 0.05, 0.7, 0.2];
        fitness_shape(&mut scores);
        let expected = [-0.2, -0.12161282, -0.2, 0.43704253, 0.084570274];
        assert_vectors(&expected, &scores);
    }

    #[test]
    fn test_znorm() {
        let mut scores = vec![-1.0, 0.0, 0.5];
        let predicted = vec![-1.33630621, 0.26726124, 1.06904497];
        normalize_weights(&mut scores, 1.0);
        for i in 0..scores.len() {
            println!("scores: {}", (scores[i] - predicted[i]).abs());
            assert!((scores[i] - predicted[i]).abs() < 1e-5);
        }
    }
}

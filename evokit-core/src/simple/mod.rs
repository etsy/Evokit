//! Simple interfaces for building up a standard (1+1) Evolutionary Strategy

use std::clone::Clone;
use std::mem;
use std::time::SystemTime;

use model::WeightUpdater;
use optimizer::*;

/// Settings for simple ES
#[derive(Debug, Clone, PartialEq)]
pub struct Simple {
    /// Number of candidates to generate for test
    pub children: usize,

    /// Number of iterations before exit
    pub iterations: usize,

    /// Number of iterations between reporting metrics
    pub report_iter: usize,

    /// When selected, will re-attempt a successful gradient
    pub elitism: bool,
}

impl Optimizer for Simple {
    fn run<S: Clone + Send, E, GS>(&self, init: S, env: &mut E, gs: &mut GS) -> State<S>
    where
        E: Environment<S>,
        GS: GradientSampler<S, E>,
        GS::Gradient: WeightUpdater,
    {
        // Build pool of gradients
        let mut state_grads = vec![(gs.zero_gradient(), init.clone()); self.children];

        env.step();

        // Create Initial state
        let (fitness, logger) = env.eval(&init);
        let mut parent = State {
            model: init,
            fitness: fitness,
            logger: logger,
        };

        let (v_fitness, logger) = env.validate(&parent.model);
        let mut best = State {
            model: parent.model.clone(),
            fitness: v_fitness.unwrap_or(parent.fitness),
            logger: logger,
        };

        // Simple momentum
        let mut progress = false;

        // Momentum grad
        let mut mom_grad = gs.zero_gradient();

        let now = SystemTime::now();

        for pass in 0..(self.iterations) {
            if pass % self.report_iter == 0 {
                let (s, m) = now
                    .elapsed()
                    .map(|e| (e.as_secs(), e.subsec_millis()))
                    .unwrap_or((0, 0));
                println!(
                    "Time: {}.{:03},\tIteration: {},\tFitness: {},\tValid: {},\tStats: {:?}",
                    s, m, pass, parent.fitness, best.fitness, parent.logger
                );
            }

            // Let the environment update itself, if needed
            let (cur_best_fit, cur_best_logger) = if env.step() {
                env.eval(&parent.model)
            } else {
                (parent.fitness, parent.logger.clone())
            };

            // Current parent is the zero gradient
            let mut cur_best = State {
                model: None,
                fitness: cur_best_fit,
                logger: cur_best_logger,
            };

            // For each child
            for (c_idx, &mut (ref mut g, ref mut s)) in state_grads.iter_mut().enumerate() {
                if c_idx == 0 && progress && self.elitism {
                    // If we made progress last time, set the first child to the best
                    // gradient
                    mem::swap(&mut mom_grad, g);
                } else {
                    // Sample new gradient
                    gs.generate(&env, &parent.model, g, pass);
                }

                // Update temp state
                gs.apply(&parent.model, g, s);

                // Evaluate the function
                let (state_fitness, state_logger) = env.eval(s);

                // Store the best we've seen
                if state_fitness > cur_best.fitness {
                    cur_best.model = Some(c_idx);
                    cur_best.fitness = state_fitness;
                    cur_best.logger = state_logger;
                }
            }

            progress = match cur_best.model {
                // If we've made forward progress, swap best states
                Some(cidx) => {
                    let (ref mut grad, ref mut new_state) = state_grads[cidx];
                    // Swap best for temp
                    mem::swap(&mut parent.model, new_state);
                    mem::swap(&mut mom_grad, grad);
                    parent.fitness = cur_best.fitness;
                    parent.logger = cur_best.logger.clone();

                    // Check validation
                    let (score, logger) = env.validate(&parent.model);
                    match score {
                        Some(score) if score > best.fitness => {
                            best.model = parent.model.clone();
                            best.fitness = score;
                            best.logger = logger;
                        }
                        None => {
                            best = parent.clone();
                        }
                        _ => (),
                    }
                    true
                }

                // No progress made
                _ => false,
            };
        }
        best
    }
}

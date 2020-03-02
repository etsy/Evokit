//! Implements the Differential Evolution metaheuristic for optimization
extern crate float_ord;
extern crate rand;
extern crate rand_xorshift;

use std::time::SystemTime;

use self::float_ord::FloatOrd;
use self::rand::distributions::{Distribution, Normal, Uniform};
use self::rand::prelude::*;
use self::rand::SeedableRng;
use self::rand_xorshift::XorShiftRng;

use crate::optimizer::{Environment, State};

use super::{ContinuousGeneticOptimizer, Genes};

#[derive(Debug, Clone, Copy)]
/// Specifies how we blend different candidates
pub enum StochasticUpdate {
    /// Rather than re-evaluate a function, this adds a simple decay to the parent
    /// fitness to force exploration after a certain number of generations.
    Decay(f32),

    /// This uses the genetic algorithm variation of always increase fitness
    GAInherit(f32),

    /// Moving Exponential Averge variant of blending stochastic functions
    Inherit(f32),

    /// Assumes f is stationary for a given environment step and re-evaluates
    /// the function
    Evaluate,
}

/// How Genes are selected from parents for breeding new variants.
#[derive(Debug, Clone, Copy)]
pub enum CrossoverType {
    /// Selects genes from a uniform distribution if less than CR
    Uniform(f32),

    /// Selects a start and end gene, crossing over all in between.  This is known
    /// also as exponential.
    TwoPoint,
}

#[derive(Debug, Clone, Copy)]
pub enum DiscardStrategy {
    /// Never discards the children
    Never,

    /// Epoch(K, num_children) Discards num_children every K epochs
    Epoch { n_passes: usize, children: usize },
}

// Defines how we do crossover between two sets of genes.
enum Crossover<'a> {
    Uniform {
        dist: Uniform<f32>,
        cr: f32,
        cur_idx: usize,
        end_idx: usize,
        rand_idx: usize,
        rng: &'a mut XorShiftRng,
    },
    TwoPoint {
        start: usize,
        end: usize,
        num_genes: usize,
    },
}

impl<'a> Crossover<'a> {
    fn new(ct: CrossoverType, num_genes: usize, rng: &'a mut XorShiftRng) -> Self {
        match ct {
            CrossoverType::Uniform(cr) => {
                let index = Uniform::from(0..num_genes).sample(rng);
                Crossover::Uniform {
                    dist: Uniform::new_inclusive(0., 1.0),
                    cr: cr,
                    cur_idx: 0,
                    end_idx: num_genes,
                    rand_idx: index,
                    rng: rng,
                }
            }
            CrossoverType::TwoPoint => {
                let points = Uniform::from(0..num_genes);
                let mut start = points.sample(rng);
                let end = points.sample(rng);
                if start == end {
                    start = (start + 1) % num_genes;
                }

                Crossover::TwoPoint {
                    start: start,
                    end: end,
                    num_genes: num_genes,
                }
            }
        }
    }
}

impl<'a> Iterator for Crossover<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Crossover::Uniform {
                dist,
                cr,
                ref mut cur_idx,
                end_idx,
                rand_idx,
                ref mut rng,
            } => {
                if *cur_idx < *end_idx {
                    *cur_idx += 1;
                    if dist.sample(rng) < *cr || (*cur_idx - 1) == *rand_idx {
                        return Some(*cur_idx - 1);
                    }
                }
                None
            }

            Crossover::TwoPoint {
                ref mut start,
                ref end,
                ref num_genes,
            } => {
                if *start != *end {
                    let ret = Some(*start);
                    *start = (*start + 1) % num_genes;
                    ret
                } else {
                    None
                }
            }
        }
    }
}

/// Mutation types from literature
pub enum Mutation {
    /// x_r1 + F * (x_r2 - x_r3)
    Rand1,

    /// x_best + F * (x_r1 - x_r2)
    Best1,

    /// x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
    CTB1,

    /// x_best + F * (x_r1 - x_r2) + F * (x_r3 - x_r4)
    Best2,

    /// x_r1 + F(x_r2 - x_r3) + F(x_r4 - x_r5)
    Rand2,

    /// x_1 + Fr * (x_r2 - x_r3)
    Cur1,
}

pub struct DifferentialEvolution {
    /// Mutation strategy for combining different variatnts
    pub mut_type: Mutation,

    /// How we select which genes to combine
    pub crossover: CrossoverType,

    /// How to handle stochastic
    pub stoch_type: StochasticUpdate,

    /// How do we discard candidates
    pub discard_strat: DiscardStrategy,

    /// Total number of candidates to use
    pub candidates: usize,

    /// Mutation scale `F`
    pub f: f32,

    /// Number of iterations to run.
    pub iterations: usize,

    /// How often to report data
    pub report_iter: usize,

    /// Random seed for mutation
    pub seed: u64,
}

impl DifferentialEvolution {
    /// Merges three different genomes together
    fn merge_three<S: Genes<f32>, I: Iterator<Item = usize>>(
        &self,
        new_c: &mut S,
        r1: &S,
        r2: &S,
        r3: &S,
        f: f32,
        crossover: I,
    ) {
        for gene_idx in crossover {
            // Mutate
            let nv = r1.get_gene(gene_idx) + f * (r2.get_gene(gene_idx) - r3.get_gene(gene_idx));
            new_c.set_gene(gene_idx, nv);
        }
    }

    /// Merges five different genomes together
    fn merge_five<S: Genes<f32>, I: Iterator<Item = usize>>(
        &self,
        new_c: &mut S,
        r1: &S,
        r2: &S,
        r3: &S,
        r4: &S,
        r5: &S,
        f: f32,
        crossover: I,
    ) {
        for gene_idx in crossover {
            // Mutate
            let nv = r1.get_gene(gene_idx)
                + f * (r2.get_gene(gene_idx) - r3.get_gene(gene_idx))
                + f * (r4.get_gene(gene_idx) - r5.get_gene(gene_idx));
            new_c.set_gene(gene_idx, nv);
        }
    }

    /// Based on the mutation type, will combine 3 or more different candidates
    /// together to form a new candidate.
    fn generate_new_candidate<S: Clone + Genes<f32>>(
        &self,
        index: usize,
        indices: &[usize],
        best: &State<S>,
        cands: &[State<S>],
        mut rng: &mut XorShiftRng,
    ) -> S {
        let num_genes = best.model.num_genes();
        match self.mut_type {
            Mutation::Rand1 => {
                let choice: Vec<_> = cands.choose_multiple(&mut rng, 3).collect();
                let mut new_c = choice[0].model.clone();
                self.merge_three(
                    &mut new_c,
                    &choice[0].model,
                    &choice[1].model,
                    &choice[2].model,
                    self.f,
                    Crossover::new(self.crossover, num_genes, rng),
                );
                new_c
            }

            Mutation::Best1 => {
                let choice: Vec<_> = cands.choose_multiple(&mut rng, 2).collect();
                let mut new_c = best.model.clone();
                self.merge_three(
                    &mut new_c,
                    &best.model,
                    &choice[0].model,
                    &choice[1].model,
                    self.f,
                    Crossover::new(self.crossover, num_genes, rng),
                );
                new_c
            }

            Mutation::CTB1 => {
                let choice: Vec<_> = indices
                    .choose_multiple(&mut rng, 3)
                    .filter(|idx| **idx != index)
                    .take(2)
                    .collect();

                let mut new_c = cands[index].model.clone();
                self.merge_five(
                    &mut new_c,
                    &cands[index].model,
                    &best.model,
                    &cands[index].model,
                    &cands[*choice[0]].model,
                    &cands[*choice[1]].model,
                    self.f,
                    Crossover::new(self.crossover, num_genes, rng),
                );
                new_c
            }

            Mutation::Best2 => {
                let choice: Vec<_> = cands.choose_multiple(&mut rng, 4).collect();
                let mut new_c = best.model.clone();
                self.merge_five(
                    &mut new_c,
                    &best.model,
                    &choice[0].model,
                    &choice[1].model,
                    &choice[2].model,
                    &choice[3].model,
                    self.f,
                    Crossover::new(self.crossover, num_genes, rng),
                );
                new_c
            }

            Mutation::Rand2 => {
                let choice: Vec<_> = cands.choose_multiple(&mut rng, 5).collect();
                let mut new_c = choice[0].model.clone();
                self.merge_five(
                    &mut new_c,
                    &choice[0].model,
                    &choice[1].model,
                    &choice[2].model,
                    &choice[3].model,
                    &choice[4].model,
                    self.f,
                    Crossover::new(self.crossover, num_genes, rng),
                );
                new_c
            }

            Mutation::Cur1 => {
                // Sample F_r from the exp(X), X ~ N(1,0)
                let f_r: f32 = Normal::new(0., 1.).sample(&mut rng) as f32;
                let choice: Vec<_> = indices
                    .choose_multiple(&mut rng, 3)
                    .filter(|idx| **idx != index)
                    .take(2)
                    .collect();

                let mut new_c = cands[index].model.clone();
                self.merge_three(
                    &mut new_c,
                    &cands[index].model,
                    &cands[*choice[0]].model,
                    &cands[*choice[1]].model,
                    f_r.exp(),
                    Crossover::new(self.crossover, num_genes, rng),
                );
                new_c
            }
        }
    }
}

/// When the environment is stochastic, applies the stochastic update rule
/// to estimate the global best scores.
fn stochastic_update<S: Clone + Genes<f32>, Env: Environment<S>>(
    env: &mut Env,
    cand: S,
    prev_fitness: f32,
    su: StochasticUpdate,
) -> State<S> {
    let (fit, logger) = env.eval(&cand);
    match su {
        StochasticUpdate::Evaluate | StochasticUpdate::Decay(_) => State::new(cand, fit, logger),

        StochasticUpdate::GAInherit(gamma) => {
            let fitness = fit + prev_fitness * (1. - gamma);
            let logger = logger;
            State::new(cand, fitness, logger)
        }

        StochasticUpdate::Inherit(gamma) => {
            let fitness = gamma * fit + prev_fitness * (1. - gamma);
            let logger = logger;
            State::new(cand, fitness, logger)
        }
    }
}

// Updates parent candidates, which is a slightly differnt process
fn stochastic_parent_update<S: Clone + Genes<f32>, Env: Environment<S>>(
    env: &mut Env,
    mut cand: &mut State<S>,
    su: StochasticUpdate,
) {
    match su {
        // Only decay fitness, do _not_ re-evaluate.
        StochasticUpdate::Decay(gamma) => {
            cand.fitness *= gamma;
        }
        // No inheritance, just re-evaluate the model
        StochasticUpdate::Evaluate => {
            let (fit, logger) = env.eval(&cand.model);
            cand.fitness = fit;
            cand.logger = logger;
        }
        // Inherit, using an always increasing metric
        StochasticUpdate::GAInherit(gamma) => {
            let (fit, logger) = env.eval(&cand.model);
            cand.fitness = fit + cand.fitness * (1. - gamma);
            cand.logger = logger;
        }
        // INherit, blending previous fitness with the new fitness
        StochasticUpdate::Inherit(gamma) => {
            let (fit, logger) = env.eval(&cand.model);
            cand.fitness = gamma * fit + cand.fitness * (1. - gamma);
            cand.logger = logger;
        }
    }
}

// Gets the best candidate
fn get_best_candidate<'a, S: Clone>(cs: &'a [State<S>]) -> &State<S> {
    cs.iter()
        .max_by_key(|s| FloatOrd(s.fitness))
        .expect("Should have at least one candidate")
}

impl ContinuousGeneticOptimizer for DifferentialEvolution {
    fn run<S, E>(&self, init: S, env: &mut E) -> State<S>
    where
        E: Environment<S>,
        S: Clone + Genes<f32>,
    {
        // Check minimum number of candidates
        assert!(self.candidates > 5);

        let uniform = Uniform::new_inclusive(-1.0, 1.0);
        let mut rng = XorShiftRng::seed_from_u64(self.seed as u64);

        let create_candidate = |env: &E, random_init: bool, mut rng: &mut XorShiftRng| {
            let mut c = init.clone();
            if random_init {
                for i in 0..c.num_genes() {
                    c.set_gene(i, uniform.sample(&mut rng));
                }
            }

            // Evaluate fitness
            let (fitness, logger) = env.eval(&c);
            State::new(c, fitness, logger)
        };

        // Initialize candidates uniformly between [-1,1]
        env.step();
        let mut candidates: Vec<_> = (0..self.candidates)
            .map(|ci| create_candidate(&env, ci > 0, &mut rng))
            .collect();

        // Figure out best and compute validation
        let mut best_candidate = get_best_candidate(&candidates).clone();

        let (v_fit, v_logger) = env.validate(&best_candidate.model);

        let mut best_validation = State {
            model: best_candidate.model.clone(),
            fitness: v_fit.unwrap_or(best_candidate.fitness),
            logger: v_logger,
        };

        // Begin
        let mut indices: Vec<_> = (0..self.candidates).collect();
        let now = SystemTime::now();
        for pass in 0..self.iterations {
            if pass % self.report_iter == 0 {
                let (s, m) = now
                    .elapsed()
                    .map(|e| (e.as_secs(), e.subsec_millis()))
                    .unwrap_or((0, 0));

                let avg_fitness =
                    candidates.iter().map(|c| c.fitness).sum::<f32>() / candidates.len() as f32;

                // Get the fitness
                println!(
                    "Time:{}.{:03},\tIteration: {},\tAvg Fitness: {:.04},\t\
                Best Fitness: {:.04},\tBest Valid: {:.04},\tStats: {:?}",
                    s,
                    m,
                    pass,
                    avg_fitness,
                    best_candidate.fitness,
                    best_validation.fitness,
                    best_validation.logger
                );
            }

            let stochastic = env.step();
            // For each candidate, generate a new candidate based on the
            // sampling DE type.
            for c_idx in 0..self.candidates {
                let new_cand = self.generate_new_candidate(
                    c_idx,
                    &indices,
                    &best_candidate,
                    &candidates,
                    &mut rng,
                );

                let offspring = if stochastic {
                    stochastic_parent_update(env, &mut candidates[c_idx], self.stoch_type);
                    let p_fitness = candidates[c_idx].fitness;
                    stochastic_update(env, new_cand, p_fitness, self.stoch_type)
                } else {
                    // Evaluate new candidates fitness.
                    // fitness, replace it
                    let (fitness, logger) = env.eval(&new_cand);
                    State::new(new_cand, fitness, logger)
                };

                if offspring.fitness >= candidates[c_idx].fitness {
                    candidates[c_idx] = offspring;
                }
            }

            // Need to update our best_candidate when stochastic as well
            if stochastic {
                stochastic_parent_update(env, &mut best_candidate, self.stoch_type)
            }

            // Check to see if the fitness has improved
            let bc = get_best_candidate(&candidates);
            if bc.fitness > best_candidate.fitness {
                best_candidate = bc.clone();
                // Evaluate validation
                best_validation = match env.validate(&best_candidate.model) {
                    (Some(fit), ref logger) if fit > best_validation.fitness => {
                        State::new(best_candidate.model.clone(), fit, logger.clone())
                    }
                    (None, _) => best_candidate.clone(),
                    _ => best_validation,
                }
            }

            // Check to see if we need to recycle candidates
            if let DiscardStrategy::Epoch { n_passes, children } = self.discard_strat {
                if (pass + 1) % n_passes == 0 {
                    indices.sort_by_key(|x| FloatOrd(candidates[*x].fitness));
                    let replace_cands_idx = indices[..children].to_vec();
                    for c_idx in replace_cands_idx {
                        candidates[c_idx] = create_candidate(&env, true, &mut rng);
                    }
                }
            }
        }

        best_validation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::ScoreLogger;

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Pair(f32, f32);

    impl Genes<f32> for Pair {
        fn num_genes(&self) -> usize {
            2
        }

        fn get_gene(&self, idx: usize) -> f32 {
            if idx == 0 {
                self.0
            } else {
                self.1
            }
        }

        fn set_gene(&mut self, idx: usize, gene: f32) {
            if idx == 0 {
                self.0 = gene;
            } else {
                self.1 = gene;
            }
        }
    }

    struct MatyasEnv(f32, f32);

    impl Environment<Pair> for MatyasEnv {
        fn step(&mut self) -> bool {
            false
        }

        fn eval(&self, state: &Pair) -> (f32, Option<ScoreLogger>) {
            let Pair(mut x, mut y) = state;
            x += self.0;
            y += self.1;
            (-(0.26 * (x.powi(2) + y.powi(2)) - 0.48 * x * y), None)
        }
    }

    #[test]
    fn test_matyas() -> () {
        let de = DifferentialEvolution {
            mut_type: Mutation::CTB1,
            crossover: CrossoverType::Uniform(1.0),
            stoch_type: StochasticUpdate::Inherit(0.9),
            discard_strat: DiscardStrategy::Never,
            candidates: 30,
            f: 0.8,
            iterations: 200,
            report_iter: 10,
            seed: 1234,
        };

        let res = de.run(Pair(0., 0.), &mut MatyasEnv(-10., 10.));
        println!("fitness: {}, Model: {:?}", res.fitness, res.model);
        assert_eq!(res.model, Pair(10., -10.0));
    }

    #[test]
    fn test_crossover() {
        for i in 0..1000 {
            let mut rng = XorShiftRng::seed_from_u64(11 + i as u64);
            let crossover = Crossover::new(CrossoverType::TwoPoint, 4, &mut rng);
            // Max 4 genes
            for (i, gene_idx) in crossover.enumerate() {
                println!("Gene Index: {}", gene_idx);
                assert!(i < 4, "infinite loop!");
            }
        }
    }
}

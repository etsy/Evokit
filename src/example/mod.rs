//! Example Environment
extern crate es_core;
extern crate rand;
extern crate rand_xorshift;

use self::es_core::model::WeightUpdater;
use self::es_core::optimizer::{Environment, GradientSampler, ScoreLogger};
use std::f64;

use self::rand::distributions::{Distribution, Normal};
use self::rand::prelude::*;
use self::rand_xorshift::XorShiftRng;

#[derive(Clone, Debug, Copy)]
/// Model state
pub struct Pair(pub (f32, f32));

pub struct MatyasEnv;

impl Environment<Pair> for MatyasEnv {
    fn step(&mut self) -> bool {
        false
    }

    fn eval(&self, state: &Pair) -> (f32, Option<ScoreLogger>) {
        let (x, y) = state.0;
        (-(0.26 * (x.powi(2) + y.powi(2)) - 0.48 * x * y), None)
    }
}

pub struct AckleyEnv;

impl Environment<Pair> for AckleyEnv {
    fn step(&mut self) -> bool {
        false
    }

    fn eval(&self, state: &Pair) -> (f32, Option<ScoreLogger>) {
        let (x, y) = state.0;
        let x64 = x as f64;
        let y64 = y as f64;
        let e = 0.0;
        let ack = -20. * (-0.2 * (0.5 * (x64.powi(2) + y64.powi(2))).sqrt()).exp()
            - (0.5 * (2. * f64::consts::PI * x64).cos() + (2. * f64::consts::PI * y64).cos()).exp()
            + e
            + 20.;

        (-(ack as f32), None)
    }
}

/// For the example, this is a simple Gradient Sampler
pub struct SimpleGS {
    /// Normal distribution to sample from
    n: Normal,
    /// Ring to use when sampling
    rng: XorShiftRng,
}

impl SimpleGS {
    /// Create a new SimpleGS
    pub fn new() -> Self {
        SimpleGS {
            n: Normal::new(0., 1.),
            rng: XorShiftRng::seed_from_u64(645342312),
        }
    }
}

impl<E: Environment<Pair>> GradientSampler<Pair, E> for SimpleGS {
    type Gradient = Pair;

    fn zero_gradient(&self) -> Self::Gradient {
        Pair((0f32, 0f32))
    }

    fn generate(&mut self, _env: &E, _s: &Pair, g: &mut Self::Gradient, _seed: usize) -> () {
        (g.0).0 = self.n.sample(&mut self.rng) as f32;
        (g.0).1 = self.n.sample(&mut self.rng) as f32;
    }

    fn apply(&self, s: &Pair, g: &Self::Gradient, ns: &mut Pair) -> () {
        (ns.0).0 = (s.0).0 + (g.0).0;
        (ns.0).1 = (s.0).1 + (g.0).1;
    }
}

impl WeightUpdater for Pair {
    fn num_weights(&self) -> usize {
        2
    }

    fn update_gradients<F>(&mut self, f: &mut F) -> ()
    where
        F: FnMut() -> f32,
    {
        self.0 = (f(), f());
    }

    fn scale_gradients(&mut self, scale: f32) -> () {
        let (x, y) = self.0;
        self.0 = (x * scale, y * scale);
    }

    fn copy_gradients(&self, other: &mut Self) -> () {
        other.0 = self.0;
    }

    fn add_gradients(&mut self, other: &Self) -> () {
        (self.0).0 = (self.0).0 + (other.0).0;
        (self.0).1 = (self.0).1 + (other.0).1;
    }
}

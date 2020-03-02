extern crate rand;
extern crate rand_xorshift;

use self::rand::distributions::{Distribution, Normal, Uniform};
use self::rand::SeedableRng;
use self::rand_xorshift::XorShiftRng;

use model::{Evaluator, GradientFuser, Initializer, WeightUpdater};
use optimizer::{Environment, GradientSampler};

pub struct ModelGS<B>
where
    B: Initializer,
{
    p: f32,
    fixed_mask: bool,
    n: Normal,
    uniform: Uniform<f32>,
    rng: XorShiftRng,
    builder: B,
}

impl<B> ModelGS<B>
where
    B: Initializer,
{
    pub fn new(b: B, mask: f32, sd: f32, fixed_mask: bool, seed: u32) -> Self {
        ModelGS {
            p: mask,
            fixed_mask: fixed_mask,
            n: Normal::new(0.0, sd as f64),
            uniform: Uniform::new_inclusive(0.0, 1.0),
            rng: XorShiftRng::seed_from_u64(seed as u64),
            builder: b,
        }
    }
}

impl<B: Send + Sync, E> GradientSampler<B::Model, E> for ModelGS<B>
where
    B: Initializer,
    B::Model:
        Clone + WeightUpdater + Evaluator<Vec<f32>, f32> + GradientFuser<B::Model> + Send + Sync,
    E: Environment<B::Model>,
{
    type Gradient = B::Model;

    fn zero_gradient(&self) -> Self::Gradient {
        self.builder.zero()
    }

    fn generate(&mut self, _env: &E, _s: &B::Model, g: &mut Self::Gradient, seed: usize) -> () {
        if self.fixed_mask {
            let mut rng = XorShiftRng::seed_from_u64(seed as u64);
            g.update_gradients(&mut || {
                if self.p == 1.0 || self.uniform.sample(&mut rng) < self.p {
                    self.n.sample(&mut self.rng) as f32
                } else {
                    0.0
                }
            });
        } else {
            g.update_gradients(&mut || {
                if self.p == 1.0 || self.uniform.sample(&mut self.rng) < self.p {
                    self.n.sample(&mut self.rng) as f32
                } else {
                    0.0
                }
            });
        }
    }

    fn apply(&self, s: &B::Model, g: &Self::Gradient, mut ns: &mut B::Model) -> () {
        s.update(&g, &mut ns);
    }
}

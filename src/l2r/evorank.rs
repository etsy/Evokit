//! Defines interfaces for learning to rank functions
extern crate es_core;
extern crate es_models;

use std::fmt::Debug;
use std::marker::PhantomData;

use super::utils::eval_rs;

use self::es_models::dataset::Dataset;
use self::es_models::regularizer::WeightDecay;
use self::es_core::optimizer::{Environment,Logger};
use self::es_core::model::Evaluator;


#[derive(Debug)]
pub struct RankSet<A: Debug> {
    pub x: Vec<A>,
    pub y: Vec<f32>
}

impl <A: Debug> RankSet<A> {
    pub fn len(&self) -> usize {
        self.x.len()
    }
}

impl <A: Debug> RankSet<A> {
    pub fn new(x: Vec<A>, y: Vec<f32>) -> Self {
        let len = x.len();
        RankSet {x, y}
    }
}

pub struct LtrEnv<TS,VS,M,D> {
    pub train: Vec<(f32, TS)>, 
    pub valid: Option<VS>,
    pub wd: f32,
    pub k: Option<usize>,
    pub valid_k: Option<usize>,
    pub pd: PhantomData<M>,
    pub d: PhantomData<D>,
}

impl <TS,VS,M,D> LtrEnv<TS,VS,M,D> {
    pub fn new(
        train: Vec<(f32, TS)>,
        v: Option<VS>, 
        wd: f32,
        k: Option<usize>,
        valid_k: Option<usize>
    ) -> Self {
        LtrEnv { 
            train: train, 
            valid: v, 
            wd: wd,
            k: k, 
            valid_k: valid_k,
            pd: PhantomData,
            d: PhantomData
        }
    }
}

// Ranking Environment
impl <TS,VS,M,D> Environment<M> for LtrEnv<TS,VS,M,D> 
    where TS: Dataset<RankSet<D>>,
          VS: Dataset<RankSet<D>>,
          D: Debug + Sync + Send,
          M: Evaluator<D, f32> + WeightDecay {

    fn step(&mut self) -> bool { 
        let mut shuffle = false;
        for &mut (_, ref mut d) in self.train.iter_mut() {
            shuffle |= d.shuffle();
        }
        shuffle
    }

    fn eval(&self, state: &M) -> (f32, Option<Logger>) {
        // Compute weight decay
        let wd = if self.wd > 0.0 {
            self.wd * state.l2norm() 
        } else {
            0.0
        };

        let mut score = 0.0;
        let mut weight = 0.0;
        for (i, &(w, ref dataset)) in self.train.iter().enumerate() {
            let k = if i == 0 { self.k } else { None };
            score += w * eval_rs(dataset.data(), state, k);
            weight += w;
        }
        ((score / weight) - wd, None)
    }

    fn validate(&self, state: &M) -> (Option<f32>, Option<Logger>) {
        match &self.valid {
            &Some(ref rs) => (Some(eval_rs(rs.data(), state, self.valid_k)), None),
            &None => (None, None)
        }
    }
}



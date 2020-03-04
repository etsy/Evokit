use es_core;
use es_data;
use rand;
use rand_xorshift;
use serde_json;

use self::rand::distributions::{Distribution, Uniform};
use self::rand::SeedableRng;
use self::rand_xorshift::XorShiftRng;

use std::io::{Read, Write};

use self::es_core::model::{Evaluator, GradientFuser, Initializer, SerDe, WeightUpdater};

use self::es_data::datatypes::Sparse;
use self::es_data::intrinsics::{l2norm, scale, sum};

use super::base::BaseTree;
use super::histogram::Histogram;
use crate::regularizer::WeightDecay;
use crate::{add_vec, copy_vec, update_vec};

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
enum RandomTreeType {
    /// Allows updating on feature indices, splits, and values
    Full,

    /// Features are fixed, but splits and payloads are updatable
    SplitsValues,
}

impl RandomTreeType {
    #[inline]
    fn update_features(&self) -> bool {
        match self {
            RandomTreeType::Full => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
/// This variant of tree uses [0,1] to encode which feature is selected.  Similarly,
/// it uses splits in the form of [0,1] which then maps it onto the [min,max] for that
/// feature to be somewhat scale invariant
struct Tree {
    /// Number of features in the tree
    n_features: usize,

    /// Number of levels in the tree
    levels: u32,

    /// Dictates the updating strategy.
    rtt: RandomTreeType,

    /// Floating point feature indices
    features: Vec<f32>,

    /// Floating feature splits
    splits: Vec<f32>,

    /// Underlying speed tree with fully resolved indices, splits, and leaf values
    tree: BaseTree,
}

// Computes the sigmoid non-linearity, usefule in trees as a continuous feature
//selector
#[inline]
fn sigmoid(x: f32) -> f32 {
    1. / (1. + std::f32::consts::E.powf(-x))
}

// Inverse of sigmoid
#[inline]
fn logit(x: f32) -> f32 {
    (x / (1. - x)).ln()
}

impl Tree {
    /// Get the number of weights for the tree
    pub fn num_weights(&self) -> usize {
        self.features.len() + self.splits.len() + self.tree.values.len()
    }

    /// Initialize a tree of zeros
    pub fn zero(n_features: usize, levels: usize, rtt: RandomTreeType) -> Self {
        let base = BaseTree::zero(levels as u32);
        Tree {
            n_features: n_features,
            levels: levels as u32,
            features: vec![0.; base.features.len()],
            splits: vec![0.; base.features.len()],
            rtt: rtt,
            tree: base,
        }
    }

    #[inline]
    /// Given a dense vector, outputs the final prediction from this tree
    pub fn predict(&self, data: &[f32]) -> f32 {
        // Defers to the underlying tree
        self.tree.predict(&data)
    }

    /// This function is what updates the base trees splits.  This is a bit indirect
    /// and there's some sharing, but we need to translate the percentages
    /// into actual values to speed up prediction
    fn update_base_tree(&mut self, histogram: &Histogram) {
        for i in 0..self.features.len() {
            let f = self.features[i];
            let feature_index = ((self.n_features - 1) as f32 * sigmoid(f)).round() as usize;
            self.tree.features[i] = feature_index;
            self.tree.splits[i] = histogram.scale(feature_index, sigmoid(self.splits[i]));
        }
    }

    fn add_gradient_into(&self, grad: &Tree, into: &mut Self) {
        if self.rtt.update_features() {
            sum(&self.features, &grad.features, &mut into.features);
        }
        sum(&self.splits, &grad.splits, &mut into.splits);
        sum(&self.tree.values, &grad.tree.values, &mut into.tree.values);
    }

    fn update_gradients<F>(&mut self, f: &mut F) -> ()
    where
        F: FnMut() -> f32,
    {
        if self.rtt.update_features() {
            update_vec(&mut self.features, f);
        }
        update_vec(&mut self.splits, f);
        update_vec(&mut self.tree.values, f);
    }

    fn scale_gradients(&mut self, f: f32) -> () {
        if self.rtt.update_features() {
            scale(&mut self.features, f);
        }
        scale(&mut self.splits, f);
        scale(&mut self.tree.values, f);
    }

    fn copy_gradients(&self, other: &mut Self) {
        if self.rtt.update_features() {
            copy_vec(&self.features, &mut other.features);
        }
        copy_vec(&self.splits, &mut other.splits);
        copy_vec(&self.tree.values, &mut other.tree.values);
    }

    fn add_gradients(&mut self, other: &Self) {
        if self.rtt.update_features() {
            add_vec(&mut self.features, &other.features);
        }
        add_vec(&mut self.splits, &other.splits);
        add_vec(&mut self.tree.values, &other.tree.values);
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
/// Representation of a forest
pub struct Forest {
    /// Number of trees in the forest
    trees: Vec<Tree>,

    /// Feature histogram
    histogram: Histogram,
}

impl Forest {
    fn zero(
        n_trees: usize,
        n_features: usize,
        n_levels: usize,
        histogram: Histogram,
        rtt: RandomTreeType,
    ) -> Self {
        Forest {
            trees: (0..n_trees)
                .map(|_| Tree::zero(n_features, n_levels, rtt))
                .collect(),
            histogram: histogram,
        }
    }
}

// Forest Prediction
impl Evaluator<Vec<f32>, f32> for Forest {
    fn evaluate(&self, payload: &Vec<f32>) -> f32 {
        self.trees
            .iter()
            .map(|tree| tree.predict(payload.as_slice()))
            .sum::<f32>()
    }
}

// Forest Prediction
impl Evaluator<Sparse, f32> for Forest {
    fn evaluate(&self, payload: &Sparse) -> f32 {
        // if the length is short enough, we use a stack array;
        // History Lesson: We pick _400_ as a magic value as something that's
        // large enough to fit in most stack sizes but small enough that the risk
        // of overflowing the stack for a thread is small.  Why 400?  No idea,
        // it's been stolen over the years from other code that assumed the size.
        if payload.0 < 400 {
            let mut v = [0f32; 400];
            let idx: &[usize] = &payload.1;
            let vals: &[f32] = &payload.2;
            for i in 0..idx.len() {
                v[idx[i]] = vals[i];
            }
            self.trees
                .iter()
                .map(|tree| tree.predict(&v[..payload.0]))
                .sum::<f32>()
        } else {
            let payload = payload.to_dense().0;
            self.trees
                .iter()
                .map(|tree| tree.predict(payload.as_slice()))
                .sum::<f32>()
        }
    }
}

impl GradientFuser<Forest> for Forest {
    fn update(&self, grad: &Forest, into: &mut Self) -> () {
        for i in 0..self.trees.len() {
            self.trees[i].add_gradient_into(&grad.trees[i], &mut into.trees[i]);
            into.trees[i].update_base_tree(&into.histogram);
        }
    }
}

impl WeightUpdater for Forest {
    #[inline]
    fn num_weights(&self) -> usize {
        self.trees.len() * self.trees[0].num_weights()
    }

    #[inline]
    fn update_gradients<F>(&mut self, f: &mut F) -> ()
    where
        F: FnMut() -> f32,
    {
        for tree in self.trees.iter_mut() {
            tree.update_gradients(f);
            tree.update_base_tree(&self.histogram);
        }
    }

    fn scale_gradients(&mut self, f: f32) -> () {
        for tree in self.trees.iter_mut() {
            tree.scale_gradients(f);
            tree.update_base_tree(&self.histogram);
        }
    }

    fn copy_gradients(&self, other: &mut Self) {
        for i in 0..self.trees.len() {
            self.trees[i].copy_gradients(&mut other.trees[i]);
            other.trees[i].update_base_tree(&other.histogram);
        }
    }

    fn add_gradients(&mut self, other: &Self) {
        for (i, tree) in self.trees.iter_mut().enumerate() {
            tree.add_gradients(&other.trees[i]);
            tree.update_base_tree(&self.histogram);
        }
    }
}

impl WeightDecay for Forest {
    #[inline]
    fn l2norm(&self) -> f32 {
        // We only norm the values within the leaves
        self.trees
            .iter()
            .map(|tree| l2norm(&tree.tree.values))
            .sum()
    }
}

#[derive(Debug)]
/// Error conditions due to writing the tree
pub enum SerDeErr {
    /// Error when writing the json
    SerDeError(serde_json::Error),
}

impl SerDe for Forest {
    type Error = SerDeErr;

    fn save<A: Write>(&self, writer: &mut A) -> Result<(), Self::Error> {
        serde_json::to_writer(writer, &self).map_err(|e| SerDeErr::SerDeError(e))
    }

    fn load<A: Read>(reader: &mut A) -> Result<Self, Self::Error> {
        serde_json::from_reader(reader).map_err(|e| SerDeErr::SerDeError(e))
    }
}

/// Standard Forest builder: builds a forest which can learn on features, splits, and values
pub struct ForestBuilder {
    /// number of trees
    n_trees: usize,
    /// number of features
    n_features: usize,
    /// number of levels
    n_levels: usize,
    /// feature histogram
    histogram: Histogram,
}

impl ForestBuilder {
    /// Return a new ForestBuilder
    pub fn new<'a>(
        n_trees: usize,
        n_levels: usize,
        n_features: usize,
        data: impl Iterator<Item = (usize, f32)>,
    ) -> Self {
        ForestBuilder {
            n_trees: n_trees,
            n_features: n_features,
            n_levels: n_levels,
            histogram: Histogram::new(n_features, data),
        }
    }
}

impl Initializer for ForestBuilder {
    type Model = Forest;

    fn zero(&self) -> Self::Model {
        Forest::zero(
            self.n_trees,
            self.n_features,
            self.n_levels,
            self.histogram.clone(),
            RandomTreeType::Full,
        )
    }
}

/// Random forest feature builder;
pub struct RandomFeatureForestBuilder {
    /// number of trees
    n_trees: usize,
    /// number of features
    n_features: usize,
    /// number of levels
    n_levels: usize,
    /// The seed for randomization
    seed: u64,
    /// feature histogram
    histogram: Histogram,
}

impl RandomFeatureForestBuilder {
    /// Return a new RandomFeatureForestBuilder
    pub fn new(
        n_trees: usize,
        n_levels: usize,
        n_features: usize,
        seed: u64,
        data: impl Iterator<Item = (usize, f32)>,
    ) -> Self {
        RandomFeatureForestBuilder {
            n_trees: n_trees,
            n_features: n_features,
            n_levels: n_levels,
            seed: seed,
            histogram: Histogram::new(n_features, data),
        }
    }
}

impl Initializer for RandomFeatureForestBuilder {
    type Model = Forest;

    fn zero(&self) -> Self::Model {
        // Create an empty tree and set the features.
        let mut forest = Forest::zero(
            self.n_trees,
            self.n_features,
            self.n_levels,
            self.histogram.clone(),
            RandomTreeType::SplitsValues,
        );

        // Randomly set the features
        let uniform = Uniform::new_inclusive(0.0, 1.0);
        let mut prng = XorShiftRng::seed_from_u64(self.seed);
        for tree in forest.trees.iter_mut() {
            for feat in tree.features.iter_mut() {
                *feat = logit(uniform.sample(&mut prng));
            }

            tree.update_base_tree(&self.histogram);
        }

        forest
    }
}

#[cfg(test)]
mod tree_tests {
    use self::rand::distributions::{Distribution, Uniform};
    use self::rand::SeedableRng;
    use self::rand_xorshift::XorShiftRng;
    use super::Histogram;
    use super::*;
    use rand;
    use rand_xorshift;

    fn build_hist() -> Histogram {
        let data = vec![vec![0.3, -1.0], vec![7., 0.9], vec![10., 1.0]];
        let it = data
            .iter()
            .flat_map(|v| v.iter().enumerate().map(|(idx, f)| (idx, *f)));
        let histogram = Histogram::new(2, it);
        histogram
    }

    fn build_tree(hist: &Histogram) -> Tree {
        // Two levels
        let mut tree = Tree::zero(2, 2, RandomTreeType::Full);
        assert_eq!(tree.splits.len(), 3);
        assert_eq!(tree.tree.values.len(), 4);

        tree.features = vec![logit(0.4), logit(0.6), logit(0.7)];
        tree.splits = vec![logit(0.5), logit(0.7), logit(0.9)];
        tree.tree.values = vec![0.1, 0.2, 0.3, 0.4];
        tree.update_base_tree(&hist);

        tree
    }

    #[test]
    fn test_trees() {
        let histogram = build_hist();
        let tree = build_tree(&histogram);

        // Should go left then right
        let data = vec![0.4, 0.7];
        let result = tree.predict(&data);
        assert_eq!(0.2, result);

        // Should go right then left
        let data = vec![6., 0.3];
        let result = tree.predict(&data);
        assert_eq!(0.3, result);

        // Should go right then right
        let data = vec![5.5, 0.9];
        let result = tree.predict(&data);
        assert_eq!(0.4, result);

        // Should go left then left, testing lte
        let data = vec![histogram.scale(0, 0.5), histogram.scale(1, 0.7)];
        let result = tree.predict(&data);
        assert_eq!(0.1, result);
    }

    #[test]
    fn test_forest() {
        // Duplicate the trees
        let histogram = build_hist();
        let mut forest = Forest::zero(2, 2, 2, histogram.clone(), RandomTreeType::Full);
        forest.trees[0] = build_tree(&histogram);
        forest.trees[1] = build_tree(&histogram);

        // Should go left then right
        let data = vec![0.4, 0.7];
        let result = forest.evaluate(&data);
        assert_eq!(0.4, result);

        // Should go right then left
        let data = vec![6., 0.3];
        let result = forest.evaluate(&data);
        assert_eq!(0.6, result);

        // Should go right then right
        let data = vec![5.5, 0.9];
        let result = forest.evaluate(&data);
        assert_eq!(0.8, result);

        // Should go left then left, testing lte
        let data = vec![histogram.scale(0, 0.5), histogram.scale(1, 0.7)];
        let result = forest.evaluate(&data);
        assert_eq!(0.2, result);
    }

    #[test]
    fn test_rand_forest() {
        let uniform = Uniform::new_inclusive(0., 1.0);
        let mut prng = XorShiftRng::seed_from_u64(123123123);
        // Duplicate the trees
        let histogram = build_hist();
        for _ in 0..1000 {
            let mut forest = Forest::zero(2, 2, 4, histogram.clone(), RandomTreeType::Full);
            forest.update_gradients(&mut || uniform.sample(&mut prng));
            let d = vec![uniform.sample(&mut prng), uniform.sample(&mut prng)];
            forest.evaluate(&d);
        }
    }

    #[test]
    fn test_forest_updates() {
        let uniform = Uniform::new_inclusive(0., 1.0);
        let mut prng = XorShiftRng::seed_from_u64(123123123);

        // Duplicate the trees
        let histogram = build_hist();
        let mut forest = Forest::zero(2, 2, 4, histogram.clone(), RandomTreeType::Full);
        forest.update_gradients(&mut || uniform.sample(&mut prng));
        let mut f1 = forest.clone();

        forest.update_gradients(&mut || uniform.sample(&mut prng));
        let mut f2 = forest.clone();
        for i in 0..forest.trees.len() {
            let ltree = &f1.trees[i];
            let rtree = &f2.trees[i];
            assert!(ltree.splits != rtree.splits);
            assert!(ltree.tree.values != rtree.tree.values);
        }

        f1.copy_gradients(&mut f2);
        for i in 0..forest.trees.len() {
            let ltree = &f1.trees[i];
            let rtree = &f2.trees[i];
            assert_eq!(ltree.splits, rtree.splits);
            assert_eq!(ltree.tree.values, rtree.tree.values);
        }

        let mut old_f1 = f1.clone();
        f1.add_gradients(&f2);
        for (i, tree) in f1.trees.iter().enumerate() {
            let rtree = &old_f1.trees[i];
            for si in 0..tree.splits.len() {
                assert!(tree.splits[si] == 2. * rtree.splits[si]);
                assert!(tree.features[si] == 2. * rtree.features[si]);
            }
            for vi in 0..tree.tree.values.len() {
                assert!(tree.tree.values[vi] == 2. * rtree.tree.values[vi]);
            }
        }
        old_f1.scale_gradients(2.);
        for i in 0..forest.trees.len() {
            let ltree = &f1.trees[i];
            let rtree = &old_f1.trees[i];
            assert!(ltree.splits == rtree.splits);
            assert!(ltree.tree.values == rtree.tree.values);
        }
    }
}

extern crate es_core;
extern crate es_data;

use self::es_core::model::Evaluator;

use self::es_data::dataset::types::SparseData;
use self::es_data::dataset::*;
use self::es_data::datatypes::Sparse;

use crate::l2r::aggregator::utils::AggBuilder;
use crate::l2r::anchor::eval_rs;
use crate::l2r::load::AnchorSetup;

/// Evaluates a model + policy on a test set using the provided scoring config
pub fn evaluate_test_data<M: Evaluator<Sparse, f32>, B: AggBuilder>(
    model: &M,
    test_file_input: String,
    parser: SparseData,
    stochastic_model: bool,
    test_setup: AnchorSetup<B>,
    seed: u32,
    test_output_file_opt: Option<String>,
) -> () {
    println!("Testing against {}", test_file_input);
    let now = std::time::Instant::now();
    let test = LazyDataset::new(parser, &test_file_input);
    let iters = if stochastic_model { 5 } else { 1 };
    for it in 0..iters {
        let (score, logger_opt) = eval_rs(
            test.data(),
            model,
            &test_setup.policy,
            &test_setup.evaluator,
            (seed << 1 + it) as usize,
            test_output_file_opt.clone(),
        );

        println!("Test Fitness: {}", score);
        println!("Test Logger: {:?}", logger_opt);
    }
    println!("Test runtime (secs): {}", now.elapsed().as_secs());
}

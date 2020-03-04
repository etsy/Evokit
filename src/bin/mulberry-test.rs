#[macro_use]
extern crate clap;
extern crate es;
extern crate es_core;
extern crate es_data;
extern crate es_models;

use clap::{App, Arg, ArgMatches};
use std::fs::File;

use self::es_core::model::SerDe;
use es::l2r::aggregator::mean::MeanAggBuilder;
use es::l2r::load::read_config;
use es_models::nn::Network;

use es::bin_utils::tester::evaluate_test_data;
use es_data::dataset::types::SparseData;

fn run(
    test_file_input: String,
    dims: usize,
    model_path: String,
    seed: u32,
    scoring_config: String,
    stochastic_model: bool,
    test_output_file_opt: Option<String>,
) -> () {
    let dims = if stochastic_model { dims + 1 } else { dims };
    let parser = SparseData(dims);
    println!("Dims: {}", dims);

    println!("{}", format!("Loading scoring config: {}", scoring_config));
    // TODO: this aggregator should be part of the model config.
    let agg = MeanAggBuilder(dims);
    let (_train_setup, _valid_setup, test_setup) =
        read_config(agg, scoring_config, stochastic_model).expect("Error reading config");

    let mut f = File::open(&model_path).expect(&format!("file `{}` failed to open!", &model_path));
    // TODO: how do we support loading other types of models?
    let model = Network::load(&mut f).expect("Error loading model!");

    evaluate_test_data(
        &model,
        test_file_input,
        parser,
        stochastic_model,
        test_setup,
        seed,
        test_output_file_opt,
    );
}

fn parse<'a>() -> ArgMatches<'a> {
    let base = App::new("Mulberry Test")
        .version("0.0.1")
        .about("Multi-objective Blackbox Neuro-Evolution Test Framework")
        .arg(
            Arg::with_name("test")
                .required(true)
                .short("t")
                .long("test")
                .takes_value(true)
                .help("Tests against the provided dataset"),
        )
        .arg(
            Arg::with_name("scoring_config")
                .short("sc")
                .long("scoring-config")
                .required(true)
                .takes_value(true)
                .help("Path to scoring config json"),
        )
        .arg(
            Arg::with_name("model_path")
                .long("model-path")
                .takes_value(true)
                .required(true)
                .help("Load model from path"),
        )
        .arg(
            Arg::with_name("features")
                .long("features")
                .takes_value(true)
                .help("Number of features in the feature vector")
                .required(true),
        )
        .arg(
            Arg::with_name("test_output_file")
                .long("test_output_file")
                .takes_value(true)
                .help("Path to save query level scores."),
        )
        .arg(
            Arg::with_name("stochastic")
                .long("stochastic")
                .help("If provided, the model is a stochastic model."),
        )
        .arg(
            Arg::with_name("seed")
                .short("s")
                .long("seed")
                .takes_value(true)
                .help("Random seed for reproducability."),
        );

    base.get_matches()
}

fn main() {
    let args = parse();

    let seed = value_t!(args, "seed", u32).unwrap_or(2018);

    let model_path = args.value_of("model_path").unwrap().into();

    let scoring_config = value_t!(args, "scoring_config", String).expect("Need a scoring config!");
    let test_output = value_t!(args, "test_output_file", String).ok();
    let stochastic = args.is_present("stochastic");

    let tname = args.value_of("test").unwrap().into();
    let features =
        value_t!(args, "features", usize).expect("Number of features needed to load model");

    run(
        tname,
        features,
        model_path,
        seed,
        scoring_config,
        stochastic,
        test_output,
    );
}

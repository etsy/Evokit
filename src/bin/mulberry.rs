#[macro_use]
extern crate clap;
extern crate es;
extern crate es_core;
extern crate es_data;
extern crate es_models;

use clap::{App, Arg, ArgMatches};

use es_core::evolution::ContinuousGeneticOptimizer;
use es_core::optimizer::Optimizer;

use es_models::nn::{NNModel, NonLinearity};

use es::l2r::aggregator::mean::MeanAggBuilder;
use es::l2r::anchor::AnchorLtrEnv;
use es::l2r::load::read_config;

use es_data::dataset::types::SparseData;

use es::bin_utils::args::{ArgAugmenter, DatasetArgs, DatasetMeta, Opt, OptimizerArgs};
use es::bin_utils::loaders::{build_init, build_l2r_datasets, write_model};
use es::bin_utils::model_params::ModelParams;
use es::bin_utils::tester::evaluate_test_data;

fn run(
    dm: DatasetMeta,
    mp: ModelParams,
    opt: Opt,
    weight_decay: f32,
    prob: f32,
    fixed_mask: bool,
    seed: u32,
    random_start: bool,
    scoring_config: String,
    stochastic_model: bool,
    test_output_file_opt: Option<String>,
) -> () {
    let dims = if stochastic_model {
        dm.dims + 1
    } else {
        dm.dims
    };
    println!("Dims: {}", dims);

    let parser = SparseData(dims);
    // Load up main dataset
    let (_dims, train_ds, valid_ds) =
        build_l2r_datasets(&dm.fname, dm.vname, dm.mini_batch, dm.freeze, seed, &parser);

    let builder = if let Some(mut n) = mp.hidden_nodes {
        n.push(1);
        NNModel::new(dims, &n, mp.act)
    } else {
        NNModel::new(dims, &[1], mp.act)
    };

    println!("{}", format!("Loading scoring config: {}", scoring_config));
    // TODO: this aggregator should be part of the model config.
    let agg = MeanAggBuilder(dims);
    let (train_setup, valid_setup, test_setup) =
        read_config(agg, scoring_config, stochastic_model).expect("Error reading config");

    let train_ds = vec![(1f32, train_ds)];
    let mut env = AnchorLtrEnv::new(
        train_ds,
        valid_ds,
        weight_decay,
        train_setup.policy,
        valid_setup.policy,
        train_setup.evaluator,
        valid_setup.evaluator,
    );

    let (init_state, mut gs) = build_init(
        builder,
        random_start,
        seed,
        prob,
        fixed_mask,
        1f32,
        &mp.load_model_path,
    );

    let now = std::time::Instant::now();
    let best = match opt {
        Opt::Simple(o) => o.run(init_state, &mut env, &mut gs),
        Opt::Canonical(o) => o.run(init_state, &mut env, &mut gs),
        Opt::Natural(o) => o.run(init_state, &mut env, &mut gs),
        Opt::DiffEvo(o) => o.run(init_state, &mut env),
    };
    println!("Train runtime (secs): {}", now.elapsed().as_secs());

    println!("Valid fitness: {}", best.fitness);
    println!("Valid Logger: {:?}", best.logger);
    // Test?
    if let Some(name) = dm.tname {
        evaluate_test_data(
            &best.model,
            name,
            parser,
            stochastic_model,
            test_setup,
            seed,
            test_output_file_opt,
        );
    }

    // Write out if asked
    write_model(&best.model, mp.save_model_path);
}

fn parse<'a>() -> ArgMatches<'a> {
    let base = App::new("Mulberry")
        .version("0.0.1")
        .about("Multi-objective Blackbox Neuro-Evolution Framework");

    let base = OptimizerArgs.add_args(base);
    let base = DatasetArgs.add_args(base);
    base.arg(
        Arg::with_name("scoring_config")
            .short("sc")
            .long("scoring-config")
            .required(true)
            .takes_value(true)
            .help("Path to scoring config json"),
    )
    .arg(
        Arg::with_name("test_output_file")
            .long("test_output_file")
            .takes_value(true)
            .help("Path to scoring config json"),
    )
    .arg(
        Arg::with_name("stochastic")
            .long("stochastic")
            .help("If provided, trains a stochastic model"),
    )
    .arg(
        Arg::with_name("save_model")
            .long("save-model")
            .takes_value(true)
            .help("Saves final model to a path"),
    )
    .arg(
        Arg::with_name("load_model")
            .long("load-model")
            .takes_value(true)
            .help("Load model from path"),
    )
    .arg(
        Arg::with_name("seed")
            .short("s")
            .long("seed")
            .takes_value(true)
            .help("Random seed for reproducability.  Bound between"),
    )
    .arg(
        Arg::with_name("weight_decay")
            .short("w")
            .long("weight-decay")
            .takes_value(true)
            .help("Coefficient for weight decay"),
    )
    .arg(
        Arg::with_name("prob")
            .short("p")
            .long("prob")
            .takes_value(true)
            .help("Masking probability."),
    )
    .arg(
        Arg::with_name("fixed_mask")
            .long("fixed-mask")
            .requires("prob")
            .help("If provided, ensures that masking probabilities are the same across children"),
    )
    .arg(
        Arg::with_name("hidden_nodes")
            .long("hidden")
            .multiple(true)
            .takes_value(true)
            .help("If provided, optimizes a single hidden-layer neural net"),
    )
    .arg(
        Arg::with_name("activation")
            .long("activation")
            .takes_value(true)
            .requires("hidden_nodes")
            .possible_values(&["relu", "sigmoid", "tanh", "elu"])
            .help("Activation function to use for hidden nodes"),
    )
    .arg(
        Arg::with_name("random_start")
            .long("random-start")
            .help("If added, the initial vector is randomly initialized"),
    )
    .arg(
        Arg::with_name("features")
            .long("features")
            .takes_value(true)
            .help("Number of features in the feature vector")
            .required(true),
    )
    .get_matches()
}

fn main() {
    let args = parse();

    let opt = OptimizerArgs.load_from_args(&args);
    let dm = DatasetArgs.load_from_args(&args);

    let weight_decay = value_t!(args, "weight_decay", f32).unwrap_or(0f32);
    let seed = value_t!(args, "seed", u32).unwrap_or(2018);
    let prob = value_t!(args, "prob", f32).unwrap_or(1.0);
    let fixed_mask = args.is_present("fixed_mask");
    assert!(prob <= 1.0 && prob > 0.0);

    let hn = args
        .values_of("hidden_nodes")
        .and_then(|vals| vals.map(|v| v.parse().ok()).collect());

    let act = args
        .value_of("activation")
        .and_then(|val| match val {
            "relu" => Some(NonLinearity::ReLu),
            "tanh" => Some(NonLinearity::Tanh),
            "sigmoid" => Some(NonLinearity::Sigmoid),
            "elu" => Some(NonLinearity::ELU),
            _ => None,
        })
        .unwrap_or(NonLinearity::ReLu);

    let random_start = args.is_present("random_start");

    let save_model_path = args.value_of("save_model");
    let load_model_path = args.value_of("load_model");

    let model_params = ModelParams {
        hidden_nodes: hn,
        act,
        load_model_path,
        save_model_path,
    };

    let scoring_config = value_t!(args, "scoring_config", String).expect("Need a scoring config!");
    let test_output = value_t!(args, "test_output_file", String).ok();
    let stochastic = args.is_present("stochastic");

    // Get priors
    run(
        dm,
        model_params,
        opt,
        weight_decay,
        prob,
        fixed_mask,
        seed,
        random_start,
        scoring_config,
        stochastic,
        test_output,
    );
}

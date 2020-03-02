#[macro_use]
extern crate clap;
extern crate es;
extern crate es_core;
extern crate es_data;
extern crate es_models;

use clap::{App, Arg, ArgMatches};

use es_core::evolution::ContinuousGeneticOptimizer;
use es_core::optimizer::{Environment, Optimizer};

use es_models::nn::{NNModel, NonLinearity};

use es_data::dataset::types::SparseData;
use es_data::dataset::*;
use es_data::load::read_libsvm;

use es::l2r::LtrEnv;

use es::bin_utils::args::{ArgAugmenter, DatasetArgs, DatasetMeta, Opt, OptimizerArgs};
use es::bin_utils::loaders::{build_init, build_l2r_datasets, build_train_dataset, write_model};
use es::bin_utils::model_params::ModelParams;

fn test_ds<D: Dataset<es_data::datatypes::Sparse>>(d: D) -> D {
    d
}

fn l2r(
    dm: DatasetMeta,
    mp: ModelParams,
    opt: Opt,
    weight_decay: f32,
    k: Option<usize>,
    valid_k: Option<usize>,
    test_k: Option<usize>,
    prob: f32,
    fixed_mask: bool,
    seed: u32,
    random_start: bool,
    prior_ds: Vec<&str>,
    prior_w: f32,
) -> () {
    let parser = SparseData(dm.dims);
    // Load up main dataset
    let (dims, d, valid_set) =
        build_l2r_datasets(&dm.fname, dm.vname, dm.mini_batch, dm.freeze, seed, &parser);

    // Load up priors, if any
    let mut train_ds = vec![(1f32, d)];
    for path in prior_ds {
        let (_, pd) = build_train_dataset(path, dm.mini_batch, dm.freeze, seed, &parser);
        train_ds.push((prior_w, test_ds(pd)));
    }

    let builder = if let Some(mut n) = mp.hidden_nodes {
        n.push(1);
        NNModel::new(dims, &n, mp.act)
    } else {
        NNModel::new(dims, &[1], mp.act)
    };

    let mut env = LtrEnv::new(train_ds, valid_set, weight_decay, k, valid_k);
    let (init_state, mut gs) = build_init(
        builder,
        random_start,
        seed,
        prob,
        fixed_mask,
        1f32,
        &mp.load_model_path,
    );

    let best = match opt {
        Opt::Simple(o) => {
            println!("Using Simple optimizer");
            o.run(init_state, &mut env, &mut gs)
        }
        Opt::Canonical(o) => {
            println!("Using Canonical optimizer");
            o.run(init_state, &mut env, &mut gs)
        }
        Opt::Natural(o) => {
            println!("Using Natural optimizer");
            o.run(init_state, &mut env, &mut gs)
        }
        Opt::DiffEvo(o) => {
            println!("Using Differntial Evolution optimizer");
            o.run(init_state, &mut env)
        }
    };

    println!("Valid fitness: {}", best.fitness);
    // Test?
    if let Some(name) = dm.tname {
        println!("Testing against {}", name);
        let rs = read_libsvm(&parser, &name).expect("Error reading test file");

        let test = StaticDataset::new(rs);
        let train = vec![(1f32, EmptyDataset::new())];
        let env = LtrEnv::new(train, Some(test), 0.0, k, test_k);
        let score_and_logger = env.validate(&best.model);
        let score = score_and_logger.0.unwrap();
        println!("Test Fitness: {}", score);
    }

    // Write out if asked
    write_model(&best.model, mp.save_model_path);
}

fn parse<'a>() -> ArgMatches<'a> {
    let base = App::new("Evo-Rank")
        .version("0.0.1")
        .author("Andrew S. <refefer@gmail.com>")
        .about("Learning to Rank framework using a variety of optimizers");

    // add optimizer arguments
    let base = OptimizerArgs.add_args(base);
    let base = DatasetArgs.add_args(base);
    base.arg(
        Arg::with_name("rank_level")
            .short("k")
            .long("train-k")
            .takes_value(true)
            .help("Train NDCG@K.  Omission doesn't limit to rank"),
    )
    .arg(
        Arg::with_name("valid_k")
            .short("a")
            .long("valid-k")
            .takes_value(true)
            .help("Valid NDCG@K.  Sets the validation K"),
    )
    .arg(
        Arg::with_name("test_k")
            .long("test-k")
            .takes_value(true)
            .help("Test NDCG@K.  Sets the NDCG@K"),
    )
    .arg(
        Arg::with_name("prior-datasets")
            .long("prior-datasets")
            .multiple(true)
            .takes_value(true)
            .help("If provided, learns against the prior policy"),
    )
    .arg(
        Arg::with_name("prior-weight")
            .long("prior-weight")
            .takes_value(true)
            .requires("prior-datasets")
            .help("Sets the weight of the prior policy.  If omitted, uses 1.0"),
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
        Arg::with_name("weight_decay")
            .short("w")
            .long("weight-decay")
            .takes_value(true)
            .help("Coefficient for weight decay"),
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

    let k = value_t!(args, "rank_level", usize).ok();
    let valid_k = value_t!(args, "valid_k", usize).ok();
    let test_k = value_t!(args, "test_k", usize).ok();

    // Get priors
    let pd: Vec<&str> = args
        .values_of("prior-datasets")
        .map(|vals| vals.collect())
        .unwrap_or(vec![]);

    let pdw = value_t!(args, "prior-weight", f32).unwrap_or(1f32);

    l2r(
        dm,
        model_params,
        opt,
        weight_decay,
        k,
        valid_k,
        test_k,
        prob,
        fixed_mask,
        seed,
        random_start,
        pd,
        pdw,
    );
}

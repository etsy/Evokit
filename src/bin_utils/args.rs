extern crate es_core;

use clap::{App, Arg, ArgMatches, SubCommand};

use self::es_core::canonical::Canonical;
use self::es_core::evolution::*;
use self::es_core::nes::Natural;
use self::es_core::simple::Simple;

/// Trait to add new arguments to the current app
pub trait ArgAugmenter {
    /// Type of struct to output from this parser
    type Output;

    /// Specifies arguments to add
    fn add_args<'a, 'b>(&self, app: App<'a, 'b>) -> App<'a, 'b>;

    /// Parses the arguments
    fn load_from_args<'a>(&self, args: &ArgMatches<'a>) -> Self::Output;
}

/// Enum defining the types of optimizers
pub enum Opt {
    /// Simple optimizer
    Simple(Simple),
    /// canonical optimizer
    Canonical(Canonical),
    /// natural (NES) optimizer
    Natural(Natural),
    /// differential evolution optimizer
    DiffEvo(DifferentialEvolution),
}

/// Struct defining the optimizer arguments using ArgAugmenter
pub struct OptimizerArgs;

impl ArgAugmenter for OptimizerArgs {
    type Output = Opt;

    /// Specifies arguments to add for the optimizer
    fn add_args<'a, 'b>(&self, app: App<'a, 'b>) -> App<'a, 'b> {
        app
      .arg(Arg::with_name("lambda")
           .short("l")
           .long("lambda")
           .takes_value(true)
           .help("Number of children in the (P, λ)"))
      .arg(Arg::with_name("iters")
           .short("i")
           .long("iters")
           .takes_value(true)
           .help("Number of iterations to run before exiting"))
      .arg(Arg::with_name("report_iters")
           .short("r")
           .long("report")
           .takes_value(true)
           .help("How often to report progress."))
      .subcommand(SubCommand::with_name("simple")
          .arg(Arg::with_name("elitism")
               .short("e")
               .long("elitism")
               .help("Whether to enable elitism")))

      .subcommand(SubCommand::with_name("canonical")
          .arg(Arg::with_name("children")
              .long("children")
              .takes_value(true)
              .required(true)
              .help("Uses Canonical ES with the provided number of parents for recombination"))
          .arg(Arg::with_name("elitism")
               .short("e")
               .long("elitism")
               .help("Whether to enable elitism"))

          .arg(Arg::with_name("always-update")
              .long("always-update")
              .help("Approximates gradients rather than only updating on improvements")))

      .subcommand(SubCommand::with_name("natural")
          .arg(Arg::with_name("momentum")
               .long("momentum")
               .takes_value(true)
               .help("Gamma parameter for momementum"))
          .arg(Arg::with_name("fitness_shaping")
               .long("fitness-shaping")
               .help("If provided, optimizes via fitness shaping"))
          .arg(Arg::with_name("alpha")
               .long("alpha")
               .takes_value(true)
               .help("Learning rate.  Defaults to 1.0")))
      .subcommand(SubCommand::with_name("diff-evo")
          .arg(Arg::with_name("F")
               .long("F")
               .takes_value(true)
               .help("Mutation rate for difference vectors"))
          .arg(Arg::with_name("mutation")
               .long("mutation")
               .takes_value(true)
               .possible_values(&["Rand1", "Rand2", "Best1", "Best2", "CTB1", "Cur1"])
               .help("Mutation type for DE"))
          .arg(Arg::with_name("crossover")
               .long("crossover")
               .takes_value(true)
               .possible_values(&["binomial", "exponential"])
               .help("Type of crossover to use during mutations"))
          .arg(Arg::with_name("stoch_type")
               .long("su")
               .takes_value(true)
               .possible_values(&["decay", "ga_inherit", "inherit", "evaluate"])
               .help("How to handle stochastic environments."))
          .arg(Arg::with_name("stoch_param")
               .long("su-param")
               .takes_value(true)
               .requires("stoch_type")
               .help("Parameter for the stochastic update policy, if needed."))
          .arg(Arg::with_name("epoch-discard")
               .long("discard")
               .takes_value(true)
               .multiple(true)
               .min_values(2)
               .max_values(2)
               .help("--discard <epochs> <num_children>. If provided, discards 
                     num_children every <epochs>"))
          .arg(Arg::with_name("cr")
               .long("cr")
               .takes_value(true)
               .help("Crossover rate when crossover is binomial.  Default is 0.1"))
          .arg(Arg::with_name("mut-seed")
               .long("mutation-seed")
               .takes_value(true)
               .help("Optional seed for randomization within DE.")))
    }

    /// Parses the arguments for the optimizer
    fn load_from_args<'a>(&self, args: &ArgMatches<'a>) -> Self::Output {
        let lambda = value_t!(args, "lambda", usize).unwrap_or(1);
        let report_iters = value_t!(args, "report_iters", usize).unwrap_or(1);
        let iterations = value_t!(args, "iters", usize).unwrap_or(100);

        if let Some(subargs) = args.subcommand_matches("simple") {
            let elitism = subargs.is_present("elitism");
            Opt::Simple(Simple {
                children: lambda,
                iterations: iterations,
                report_iter: report_iters,
                elitism: elitism,
            })
        } else if let Some(subargs) = args.subcommand_matches("canonical") {
            let elitism = subargs.is_present("elitism");
            let all_update = subargs.is_present("always-update");
            let children =
                value_t!(subargs, "children", usize).expect("--children is required for canonical");
            // Yes, this is backward and shall be fixed
            Opt::Canonical(Canonical {
                parents: children,
                children: lambda,
                iterations: iterations,
                report_iter: report_iters,
                elitism: elitism,
                always_update: all_update,
            })
        } else if let Some(subargs) = args.subcommand_matches("natural") {
            let alpha = value_t!(subargs, "alpha", f32).unwrap_or(1.0);
            let momentum = value_t!(subargs, "momentum", f32).ok();
            let shape = subargs.is_present("fitness_shaping");
            Opt::Natural(Natural {
                children: lambda,
                iterations: iterations,
                report_iter: report_iters,
                momentum: momentum,
                alpha: alpha,
                shape: shape,
            })
        } else if let Some(subargs) = args.subcommand_matches("diff-evo") {
            let mut_rate = value_t!(subargs, "F", f32).unwrap_or(1.0);
            let mut_type = value_t!(subargs, "mutation", String).unwrap_or("Rand2".into());

            let mutation = match mut_type.as_ref() {
                "Rand1" => Mutation::Rand1,
                "Rand2" => Mutation::Rand2,
                "Best1" => Mutation::Best1,
                "Best2" => Mutation::Best2,
                "CTB1" => Mutation::CTB1,
                "Cur1" => Mutation::Cur1,
                _ => panic!(format!("Undefined mutation type: {}", mut_type)),
            };
            let cr = value_t!(subargs, "cr", f32).unwrap_or(0.1);
            let co_type = value_t!(subargs, "crossover", String).unwrap_or("exponential".into());

            let co = match co_type.as_ref() {
                "binomial" => CrossoverType::Uniform(cr),
                "exponential" => CrossoverType::TwoPoint,
                _ => panic!(format!("Undefined crossover type: {}", co_type)),
            };

            let su_type = value_t!(subargs, "stoch_type", String).unwrap_or("evaluate".into());
            let su_param = value_t!(subargs, "stoch_param", f32).unwrap_or(0.99);

            let su = match su_type.as_ref() {
                "decay" => StochasticUpdate::Decay(cr),
                "ga_inherit" => StochasticUpdate::GAInherit(su_param),
                "inherit" => StochasticUpdate::Inherit(su_param),
                "evaluate" => StochasticUpdate::Evaluate,
                _ => panic!(format!("Undefined stochastic update: {}", su_type)),
            };
            let ds = if let Some(discard) = subargs.values_of("epoch-discard") {
                let params: Vec<_> = discard.collect();
                let epoch: usize = params[0]
                    .parse()
                    .expect("epoch needs to be a positive integer");
                let num_children: usize = params[1]
                    .parse()
                    .expect("num_children needs to be a positive integer");
                DiscardStrategy::Epoch {
                    n_passes: epoch,
                    children: num_children,
                }
            } else {
                DiscardStrategy::Never
            };

            let seed = value_t!(subargs, "mut-seed", u64).unwrap_or(20192019);

            let de = DifferentialEvolution {
                mut_type: mutation,
                crossover: co,
                stoch_type: su,
                discard_strat: ds,
                candidates: lambda,
                f: mut_rate,
                iterations: iterations,
                report_iter: report_iters,
                seed: seed,
            };
            Opt::DiffEvo(de)
        } else {
            panic!("Unable to parse the args.  This should never be hit")
        }
    }
}

/// Specifies the metadata for the dataset
pub struct DatasetMeta {
    /// Name of training set file
    pub fname: String,
    /// Name of validation set file
    pub vname: Option<String>,
    /// Name of test set file
    pub tname: Option<String>,
    /// Number of dimensions of the sparse vector
    pub dims: usize,
    /// If provided, use mini-batches of the provided size
    pub mini_batch: Option<f32>,
    /// Whether to freeze the minibatch
    pub freeze: u32,
}

/// Struct defining the dataset arguments using ArgAugmenter
pub struct DatasetArgs;

impl ArgAugmenter for DatasetArgs {
    type Output = DatasetMeta;

    /// Specifies arguments to add for the dataset
    fn add_args<'a, 'b>(&self, app: App<'a, 'b>) -> App<'a, 'b> {
        app.arg(
            Arg::with_name("train")
                .index(1)
                .required(true)
                .help("Training file"),
        )
        .arg(
            Arg::with_name("valid")
                .short("v")
                .long("valid")
                .takes_value(true)
                .help("Use validation data"),
        )
        .arg(
            Arg::with_name("test")
                .short("t")
                .long("test")
                .takes_value(true)
                .help("After training, tests against the provided dataset"),
        )
        .arg(
            Arg::with_name("features")
                .long("features")
                .takes_value(true)
                .help("Number of features in the feature vector"),
        )
        .arg(
            Arg::with_name("minibatch")
                .short("m")
                .long("minibatch")
                .takes_value(true)
                .help("Percentage of ranked sets to use in each pass"),
        )
        .arg(
            Arg::with_name("freeze")
                .short("f")
                .long("freeze")
                .takes_value(true)
                .help("Number of iterations to freeze the batch"),
        )
    }

    /// Parses the arguments for the dataset
    fn load_from_args<'a>(&self, args: &ArgMatches<'a>) -> Self::Output {
        let fname = args.value_of("train").unwrap().into();
        let vname = value_t!(args, "valid", String).ok();
        let tname = value_t!(args, "test", String).ok();
        let minibatch = value_t!(args, "minibatch", f32).ok();
        let freeze = value_t!(args, "freeze", u32).unwrap_or(1);
        let features =
            value_t!(args, "features", usize).expect("Number of features needed to load model");

        DatasetMeta {
            fname: fname,
            vname: vname,
            tname: tname,
            mini_batch: minibatch,
            freeze: freeze,
            dims: features,
        }
    }
}

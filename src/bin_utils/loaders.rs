extern crate es_core;
extern crate es_data;
extern crate rand;
extern crate rand_xorshift;

use std::fmt::Debug;
use std::fs::File;

use self::rand::distributions::{Distribution, Uniform};
use self::rand::SeedableRng;
use self::rand_xorshift::XorShiftRng;

use self::es_core::model::sampler::ModelGS;
use self::es_core::model::{Initializer, SerDe, WeightUpdater};

use self::es_data::dataset::types::DataParse;
use self::es_data::dataset::{Dataset, EagerIterator, MinibatchDataset, StaticDataset};
use self::es_data::datatypes::Dimension;
use self::es_data::load::read_libsvm;

/// Reads a libsvm file and processes it to generate a dataset
pub fn build_train_dataset<
    Data: 'static + Debug + Dimension<Out = usize> + Send + Sync + Clone,
    D: DataParse<Out = Data>,
>(
    fname: &str,
    mini_batch: Option<f32>,
    freeze: u32,
    seed: u32,
    parser: &D,
) -> (
    usize,
    Box<dyn Dataset<Data, Iter = EagerIterator<Data>> + Send + Sync>,
) {
    println!("Loading dataset at {}", fname);
    let train_set =
        read_libsvm(parser, fname).expect(&format!("Error reading training file: {}", fname));

    println!("Query sets: {}", train_set.len());
    println!(
        "Examples: {}",
        train_set.iter().map(|rs| rs.x.len()).sum::<usize>()
    );

    let dims = train_set[0].x[0].dims();

    // Minibatch or full dataset?
    let train: Box<dyn Dataset<Data, Iter = EagerIterator<Data>> + Send + Sync> = match mini_batch {
        Some(p) => {
            assert!(p > 0.0 && p < 1.0);
            assert!(freeze > 0);
            Box::new(MinibatchDataset::new(train_set, p, freeze, seed + 1))
        }
        None => Box::new(StaticDataset::new(train_set)),
    };
    (dims, train)
}

/// Loads the training dataset. If a validation dataset was provided, loads this as well.
///
/// This also outputs the dimensions. This is pulled from the first record in the training set. As
/// libsvm is a sparse format though, this may not be the true dimension value. This is only used
/// for ES-Rank, not Mulberry.
pub fn build_l2r_datasets<
    Data: 'static + Debug + Dimension<Out = usize> + Send + Sync + Clone,
    D: DataParse<Out = Data>,
>(
    fname: &str,
    vname: Option<String>,
    mini_batch: Option<f32>,
    freeze: u32,
    seed: u32,
    parser: &D,
) -> (
    usize,
    Box<dyn Dataset<Data, Iter = EagerIterator<Data>> + Send + Sync>,
    Option<StaticDataset<Data>>,
) {
    let now = std::time::Instant::now();
    let (dims, train) = build_train_dataset(fname, mini_batch, freeze, seed, parser);

    // If we have a validation dataset, use it
    let valid_set = vname.map(|name| {
        println!("Reading validation dataset {}...", name);
        let ds =
            read_libsvm(parser, &name).expect(&format!("Error reading validation file: {}", name));

        println!("Query sets: {}", ds.len());
        println!(
            "Examples: {}",
            ds.iter().map(|rs| rs.x.len()).sum::<usize>()
        );
        StaticDataset::new(ds)
    });
    println!("Loading runtime (secs): {}", now.elapsed().as_secs());

    (dims, train, valid_set)
}

/// Initializes the underlying model. Option for a random start.
fn initialize<M: WeightUpdater>(m: &mut M, random_start: bool, seed: u32) {
    if random_start {
        let uniform = Uniform::new_inclusive(-1.0, 1.0);
        let mut prng = XorShiftRng::seed_from_u64((seed as u64) << 3 + 3);
        m.update_gradients(&mut || uniform.sample(&mut prng));
    }
}

/// Builds the initial state. If a previous model was provided, we will load that for incremental training
pub fn build_init<
    E: Debug,
    M: WeightUpdater + SerDe<Error = E> + Clone + Send + Sync,
    B: Initializer<Model = M>,
>(
    builder: B,
    random_start: bool,
    seed: u32,
    prob: f32,
    fixed_mask: bool,
    sd: f32,
    load_model_path: &Option<&str>,
) -> (M, ModelGS<B>) {
    // Load the model, if provided, otherwise initialize a new one
    let init_state = if let Some(path) = load_model_path {
        let mut f = File::open(path).expect(&format!("file `{}` failed to open!", path));
        M::load(&mut f).expect("Error loading model!")
    } else {
        let mut i_state = builder.zero();
        initialize(&mut i_state, random_start, seed);
        i_state
    };
    let gs = ModelGS::new(builder, prob, sd, fixed_mask, seed);
    (init_state, gs)
}

/// Write out the model to disk
pub fn write_model<E: Debug, M: SerDe<Error = E>>(model: &M, smp: Option<&str>) -> () {
    match smp {
        Some(fname) => {
            println!("Writing model to {}", fname);
            let mut f = File::create(fname).expect(&format!("file `{}` failed to open!", fname));
            model.save(&mut f).expect("Error writing model!")
        }
        _ => (),
    }
}

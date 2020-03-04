extern crate es;
extern crate es_core;

use self::es::example::{MatyasEnv, Pair, SimpleGS};
use self::es_core::nes::Natural;
use self::es_core::optimizer::Optimizer;

fn main() -> () {
    // Use a Natural Evolutionary Strategies with a slight momentum
    let optimizer = Natural {
        children: 100,
        iterations: 100,
        report_iter: 10,
        momentum: Some(0.1),
        alpha: 1.,
        shape: true,
    };

    let init = Pair((10f32, 10f32));

    let results = optimizer.run(init, &mut MatyasEnv, &mut SimpleGS::new());
    let model = results.model;
    println!(
        "Best Score: {}, Best Model: {},{}",
        results.fitness,
        (model.0).0,
        (model.0).1
    );
}

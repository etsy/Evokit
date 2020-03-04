extern crate es_data;

use self::es_data::datatypes::Sparse;

/// Copy the data from one sparse vector to another
pub fn copy_anchor(from: &Sparse, to: &mut Sparse) -> () {
    to.0 = from.0;
    to.1.truncate(0);
    to.2.truncate(0);
    to.1.extend_from_slice(&from.1);
    to.2.extend_from_slice(&from.2);
}

/// Computes the set difference of two sparse vectors
pub fn set_difference(a: &Sparse, b: &Sparse, result: &mut Sparse) -> () {
    // Clear the result
    result.0 = a.0;
    result.1.truncate(0);
    result.2.truncate(0);

    let mut i = 0;
    let mut j = 0;

    while i < a.1.len() && j < b.1.len() {
        if a.1[i] < b.1[j] {
            result.1.push(a.1[i]);
            result.2.push(a.2[i]);
            i += 1;
        } else if a.1[i] > b.1[j] {
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }

    for idx in i..(a.1.len()) {
        result.1.push(a.1[idx]);
        result.2.push(a.2[idx]);
    }
}

/// Combines two vectors using the provided function
/// combiner_fn takes an index in base, base's value, and addend's value. This allows us to access
/// other SparseVectors in the aggregator
pub fn combine_vectors_with_init_and_combiner(
    base: &Sparse,
    addend: &Sparse,
    init_fn: &dyn Fn(f32) -> Option<f32>,
    combiner_fn: &dyn Fn(usize, f32, f32) -> f32,
    result: &mut Sparse,
) -> () {
    // Clear the result
    result.0 = base.0;
    result.1.truncate(0);
    result.2.truncate(0);

    let mut i = 0;
    let mut j = 0;

    while i < base.1.len() && j < addend.1.len() {
        // Feature isn't in addend
        if base.1[i] < addend.1[j] {
            result.1.push(base.1[i]);
            result.2.push(base.2[i]);
            i += 1;

        // Feature isn't in base
        } else if base.1[i] > addend.1[j] {
            let val = init_fn(addend.2[j]);
            if val.is_some() {
                result.1.push(addend.1[j]);
                result.2.push(val.expect("Expected an initial value"));
            }
            j += 1;
        } else {
            result.1.push(base.1[i]);
            result.2.push(combiner_fn(i, base.2[i], addend.2[j]));
            i += 1;
            j += 1;
        }
    }

    for idx in i..(base.1.len()) {
        result.1.push(base.1[idx]);
        result.2.push(base.2[idx]);
    }

    for idx in j..(addend.1.len()) {
        let val = init_fn(addend.2[idx]);
        if val.is_some() {
            result.1.push(addend.1[idx]);
            result.2.push(val.expect("Expected an initial value"));
        }
    }
}

/// Given a method for combining vectors, apply it to the two and store in result
/// combiner_fn always adds after modifying the addend
pub fn combine_vectors_with_func(
    base: &Sparse,
    addend: &Sparse,
    modify_fn: &dyn Fn(f32) -> f32,
    result: &mut Sparse,
) -> () {
    // idx is unused
    let combiner_fn = |_idx: usize, b: f32, a: f32| b + modify_fn(a);
    let init_fn = |x: f32| Some(modify_fn(x));
    combine_vectors_with_init_and_combiner(base, addend, &init_fn, &combiner_fn, result);
}

/// Initial function. Nothing is applied
pub fn init_none_fn(_x: f32) -> Option<f32> {
    None
}

/// Incremental mean computation
pub fn mean_incrementer_fn(idx: usize, b: f32, a: f32, num_items: &Vec<f32>) -> f32 {
    // num_items should not include the new item yet, i.e. self.num_items = N - 1
    // idx should be the same for base and num_items since we haven't updated either yet.
    b + (a - b) / (num_items[idx] + 1.)
}

/// Compute xk = xi + xj
pub fn add_vector(xi: &Sparse, xj: &Sparse, xk: &mut Sparse) -> () {
    // Need an identity function that doesn't return an Option
    fn identity_fn(x: f32) -> f32 {
        x
    };
    combine_vectors_with_func(xi, xj, &identity_fn, xk);
}

/// Trait for defining aggregator methods
pub trait Aggregator {
    /// Given a sparse vector, add it to the aggregator
    fn update(&mut self, item: &Sparse) -> ();

    /// Get the current aggregator vector
    fn read(&self) -> &Sparse;
}

/// Trait to define how to initialize an aggregator as they need to be initialized each call to a Beam Policy
pub trait AggBuilder: Send + Sync {
    /// Specifies the type of aggregator
    type Agg: Aggregator + Clone;

    /// Gets the starting point for the aggregator
    fn start(&self) -> Self::Agg;
}

/// Validates the aggregator is as expected (for tests)
/// expected_values: values after aggregating each vector in data one at a time.
pub fn validate_agg(
    aggregator: &mut dyn Aggregator,
    data: &Vec<Sparse>,
    expected_values: &Vec<Sparse>,
) -> () {
    for (example, expected) in data.iter().zip(expected_values.iter()) {
        aggregator.update(example);
        let agg_vec = aggregator.read();

        // assert aggregator and expected are the same
        assert_eq!(agg_vec.0, expected.0);
        assert_eq!(agg_vec.1.len(), expected.1.len());
        assert_eq!(agg_vec.2.len(), expected.2.len());
        assert_eq!(agg_vec.1, expected.1);
        for (agg_val, exp_val) in agg_vec.2.iter().zip(expected.2.iter()) {
            assert!((agg_val - exp_val).abs() < 1e-6);
        }
    }
}

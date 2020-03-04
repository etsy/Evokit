//! Metrics
//! ---
//! Contains a set of optimization metrics
//!
//! These are useful for different scorers
extern crate es_data;
extern crate float_ord;
extern crate hashbrown;

use self::es_data::dataset::types::{MetaType, Metadata};
use self::hashbrown::HashMap;

use self::float_ord::FloatOrd;

/// Computes DCG@K for a given relevance set
fn dcg(scores: &[f32], k: usize) -> f64 {
    let mut rdcg = 0f64;
    for i in 0..k {
        let s = scores[i];
        rdcg += ((2f64).powi(s as i32) - 1.) / (2. + i as f64).log2()
    }
    rdcg
}

/// Computes NDCG@K for a given relevance set
pub fn ndcg(scores: &mut [f32], k: Option<usize>) -> f64 {
    let size = k.unwrap_or(scores.len()).min(scores.len());
    let r_dcg = dcg(scores, size);

    // Sort them in ascending order
    scores.sort_by_key(|v| FloatOrd(-*v));
    let idcg = dcg(scores, size);
    if idcg > 0.0 {
        r_dcg / idcg
    } else {
        0.0
    }
}

#[inline]
/// Gets relevance for ERR
fn get_relevance(score: f32, score_max: f32) -> f32 {
    (2f32.powf(score) - 1.) / 2f32.powf(score_max)
}

/// Computes ERR. Assumes scores are sorted
pub fn get_err(scores: &[f32], k_opt: Option<usize>) -> f32 {
    let k = k_opt.unwrap_or(scores.len()).min(scores.len());
    let score_max = scores
        .iter()
        .max_by_key(|x| FloatOrd(**x))
        .expect("Must have a maximum score");

    let mut err = 0.0;
    let mut p = 1.0;
    for rank in 1..=k {
        let relevance = get_relevance(scores[rank - 1], *score_max);
        err += p * relevance / (rank as f32);
        p *= 1. - relevance;
    }

    err
}

/// Gets the weights for sub-topics for Discrete-ERRIA. Computes p(t | q)
pub fn get_subtopic_weights(subtopics: &[u32]) -> HashMap<u32, f32> {
    let mut weights = HashMap::new();
    let num_examples = subtopics.len();
    if num_examples == 0 {
        return weights;
    }
    for topic in subtopics.iter() {
        let counter = weights.entry(*topic).or_insert(0.);
        *counter += 1.;
    }

    for (_, val) in weights.iter_mut() {
        *val /= num_examples as f32;
    }

    weights
}

/// Gets the subtopics. Run this once
/// # Arguments
///
/// * data: Data to get subtopics from
/// * field_name: field containing the topic
/// * discretize_fn specifies the name of the bucket and how to handle missing data.
pub fn get_subtopics<F>(data: &[&Metadata], field_name: &String, discretize_fn: F) -> Vec<u32>
where
    F: Fn(Option<&MetaType>) -> u32,
{
    let mut topics = Vec::new();
    for metadata in data.iter() {
        let value = metadata.get(field_name);
        topics.push(discretize_fn(value));
    }
    topics
}

/// Computes Discrete-ERRIA. Assumes the scores are sorted.
/// # Arguments
///
/// * scores: labels
/// * subtopics: subtopic for each doc
/// * subtopic_weights: weight for each topic
/// * k_opt: top-K docs to compute this over
pub fn get_err_ia(
    scores: &[f32],
    subtopics: &[u32],
    subtopic_weights: &HashMap<u32, f32>,
    k_opt: Option<usize>,
) -> f32 {
    let mut err_ia: f32 = 0.0;
    for (topic, prob_topic_given_query) in subtopic_weights.iter() {
        // Set the score for any doc without this topic to 0.
        // Can't just filter as we need the index
        let topic_scores: Vec<f32> = scores
            .iter()
            .enumerate()
            .map(|(i, &x)| if subtopics[i] == *topic { x } else { 0f32 })
            .collect();
        let err_at_k_for_topic = get_err(&topic_scores, k_opt);
        err_ia += prob_topic_given_query * err_at_k_for_topic;
    }

    err_ia
}

/// Computes cumulative values for gini coefficient
pub fn compute_cumulative_values(data: &[f32]) -> Vec<f32> {
    let mut cumulative = Vec::with_capacity(data.len() + 1);
    let mut total = 0.;
    for val in data {
        cumulative.push(total);
        total += val;
    }
    cumulative.push(total);

    if total == 0. {
        return cumulative;
    }

    for val in cumulative.iter_mut() {
        *val /= total;
    }

    cumulative
}

/// Compute the gini coefficient for the provided income & population
pub fn get_gini_coefficient(income_and_population: &mut [(f32, f32)]) -> f32 {
    // No inequality if there are no examples.
    if income_and_population.is_empty() {
        return 0.;
    }
    // Sort the incomes and population so the cumulative wealth is below the optimal line
    income_and_population.sort_by(|a, b| {
        let a_ratio = a.0 / a.1;
        let b_ratio = b.0 / b.1;
        a_ratio.partial_cmp(&b_ratio).expect("should unwrap float")
    });
    let income = income_and_population
        .iter()
        .map(|x| x.0)
        .collect::<Vec<f32>>();
    let population = income_and_population
        .iter()
        .map(|x| x.1)
        .collect::<Vec<f32>>();

    // Compute cumulative populations and wealth
    let wealth_cumulative = compute_cumulative_values(&income);
    let population_cumulative = compute_cumulative_values(&population);
    let income_total = wealth_cumulative.last().expect("Must have an income value");
    let population_total = population_cumulative
        .last()
        .expect("Must have a population value");

    // If no income to spread or no population, there is no inequality
    if income_total.abs() <= 1e-6 || population_total.abs() <= 1e-6 {
        return 0.;
    }

    let mut gini = 0.;
    for i in 1..wealth_cumulative.len() {
        gini += (population_cumulative[i] - population_cumulative[i - 1])
            * (wealth_cumulative[i] + wealth_cumulative[i - 1]);
    }
    gini
}

/// Find the percentile given a set of values. This requires some interpolation
fn interpolate(vals: &[f32], percentile: usize, interpolate_arg_opt: Option<f32>) -> f32 {
    let interpolate_arg = interpolate_arg_opt.unwrap_or(0.5);
    let v_len = vals.len() as f32;
    let pos =
        (v_len + 1. - 2. * interpolate_arg) * (percentile as f32) / 100. + interpolate_arg - 1.;
    if (pos.ceil() as usize) == 0 {
        vals[0]
    } else if (pos.floor() as usize) == (vals.len() - 1) {
        vals[vals.len() - 1]
    } else {
        let left = vals[pos.floor() as usize];
        let right = vals[pos.ceil() as usize];
        let delta = pos.fract();
        left * (1. - delta) + right * delta
    }
}

/// Compute a set of percentiles and average them
pub fn get_percentiles(
    vals: &mut [f32],
    percentiles: &[usize],
    interpolate_arg_opt: Option<f32>,
) -> f32 {
    // Can happen at test time
    if vals.is_empty() {
        std::f32::NAN
    } else {
        vals.sort_by_key(|x| FloatOrd(*x));
        let s: f32 = percentiles
            .iter()
            .map(|p| interpolate(&vals, *p, interpolate_arg_opt))
            .sum();
        s / percentiles.len() as f32
    }
}

/// Computes the mean
/// # Arguments
///
/// * `scores` list of numbers to average
/// * `k_opt` number of top docs to include. If none is provided, uses all docs
pub fn get_mean(data: &[f32], k_opt: Option<usize>) -> f32 {
    let k = k_opt.unwrap_or(data.len()).min(data.len());
    let total = &data[..k].iter().sum::<f32>();
    total / (k as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let data = [1., 2., 6.];
        assert_eq!(get_mean(&data, None), 3.);
        assert_eq!(get_mean(&data, Some(2)), 1.5);
        assert_eq!(get_mean(&data, Some(10)), 3.);
    }

    #[test]
    fn test_ndcg() {
        let mut t1 = vec![4., 0., 2., 1., 2.];
        assert!((ndcg(&mut t1.clone(), None) - 0.96110010).abs() < 1e-6);
        assert!((ndcg(&mut t1, Some(2)) - 0.8879528).abs() < 1e-6);
        assert_eq!(ndcg(&mut t1, Some(0)), 0f64);
    }

    #[test]
    fn test_err() {
        let scores = vec![4., 0., 2., 1., 2.];
        assert_eq!(get_err(&scores, Some(0)), 0f32);
        assert!((get_err(&scores, Some(1)) - 0.9375).abs() < 1e-6);
        assert!((get_err(&scores, Some(2)) - 0.9375).abs() < 1e-6);
        assert!((get_err(&scores, Some(3)) - 0.94140625).abs() < 1e-6);
        assert!((get_err(&scores, Some(4)) - 0.9421997).abs() < 1e-6);
        assert!((get_err(&scores, Some(5)) - 0.94398493).abs() < 1e-6);
        assert_eq!(get_err(&scores, None), get_err(&scores, Some(scores.len())));
        assert_eq!(
            get_err(&scores, Some(10)),
            get_err(&scores, Some(scores.len()))
        );
    }

    #[test]
    fn test_gini() {
        {
            let mut data = vec![(0.4, 0.05), (0.6, 0.95)];
            assert!((get_gini_coefficient(&mut data) - 0.65).abs() < 1e-6);
        }
        {
            let mut data = vec![(0.2, 0.1), (0.8, 0.9)];
            assert!((get_gini_coefficient(&mut data) - 0.9).abs() < 1e-6);
        }
    }

    #[test]
    fn test_get_subtopic_weights() {
        let mut str_data = Vec::new();
        let mut expected = HashMap::new();
        for i in 0..10 {
            {
                let mut metadata = Metadata::new();
                metadata.insert("taxonomy".to_string(), MetaType::Str(format!("{:?}", i)));
                str_data.push(metadata);
                expected.insert(i, 1. / 30.);
            }
            {
                let mut metadata = Metadata::new();
                metadata.insert(
                    "taxonomy".to_string(),
                    MetaType::Str(format!("2{:?}", i / 10)),
                );
                str_data.push(metadata);
                expected.insert(20 + i / 10, 1. / 3.);
            }
            {
                let metadata = Metadata::new();
                str_data.push(metadata);
                expected.insert(std::u32::MAX, 1. / 3.);
            }
        }
        let discretize_fn = |x: Option<&MetaType>| match x {
            Some(MetaType::Str(val)) => val.parse::<u32>().expect("should be a number"),
            None => std::u32::MAX,
            _ => panic!("Should have some string data"),
        };
        let sub: Vec<_> = str_data.iter().collect();
        let subtopics = get_subtopics(&sub, &"taxonomy".to_string(), &discretize_fn);
        let weights = get_subtopic_weights(&subtopics);

        assert_eq!(subtopics.len(), sub.len());
        println!("Weights: {:?}", weights);
        println!("expected: {:?}", expected);
        assert_eq!(weights.len(), expected.len());
        for (key, val) in expected.iter() {
            assert!(weights.contains_key(key));
            let actual_val = weights.get(key).expect("key should be in weights");
            assert!((val - actual_val).abs() < 1e-6);
        }
    }

    #[test]
    fn test_err_ia() {
        let mut cat1_metadata = Metadata::new();
        cat1_metadata.insert("taxonomy".to_string(), MetaType::Str("1".to_string()));
        let mut cat2_metadata = Metadata::new();
        cat2_metadata.insert("taxonomy".to_string(), MetaType::Str("2".to_string()));
        let scores = vec![
            (4., &cat1_metadata),
            (0., &cat2_metadata),
            (2., &cat1_metadata),
            (1., &cat2_metadata),
            (2., &cat2_metadata),
        ];

        let discretize_fn = |x: Option<&MetaType>| match x {
            Some(MetaType::Str(val)) => val.parse::<u32>().expect("should be a number"),
            None => std::u32::MAX,
            _ => panic!("Should have some string data"),
        };
        let metadata: Vec<_> = scores.iter().map(|x| x.1).collect();
        let just_scores: Vec<_> = scores.iter().map(|x| x.0).collect();
        let subtopics = get_subtopics(&metadata, &"taxonomy".to_string(), &discretize_fn);

        let weights = get_subtopic_weights(&subtopics);
        assert_eq!(
            get_err_ia(&just_scores, &subtopics, &weights, Some(0)),
            0f32
        );
        assert!((get_err_ia(&just_scores, &subtopics, &weights, Some(1)) - 0.375).abs() < 1e-6);
        assert!((get_err_ia(&just_scores, &subtopics, &weights, Some(2)) - 0.375).abs() < 1e-6);
        assert!((get_err_ia(&just_scores, &subtopics, &weights, Some(3)) - 0.3765625).abs() < 1e-6);
        assert!((get_err_ia(&just_scores, &subtopics, &weights, Some(4)) - 0.4140625).abs() < 1e-6);
        assert!((get_err_ia(&just_scores, &subtopics, &weights, Some(5)) - 0.4815625).abs() < 1e-6);
        assert_eq!(
            get_err_ia(&just_scores, &subtopics, &weights, None),
            get_err_ia(&just_scores, &subtopics, &weights, Some(5))
        );
        assert_eq!(
            get_err_ia(&just_scores, &subtopics, &weights, Some(10)),
            get_err_ia(&just_scores, &subtopics, &weights, Some(5))
        );
    }

    #[test]
    fn test_interpolate() {
        {
            let values = vec![2.0, 4.0];
            assert_eq!(interpolate(&values, 0, None), 2.0);
            assert_eq!(interpolate(&values, 25, None), 2.0);
            assert_eq!(interpolate(&values, 50, None), 3.0);
            assert_eq!(interpolate(&values, 100, None), 4.0);
        }
        {
            let values = vec![2.0, 4.0, 100.0];
            assert_eq!(interpolate(&values, 50, None), 4.0);
        }
        {
            // Example from wikipedia
            let values = vec![15.0, 20.0, 35.0, 40.0, 50.0];
            assert_eq!(interpolate(&values, 5, None), 15.0);
            assert_eq!(interpolate(&values, 30, None), 20.0);
            assert_eq!(interpolate(&values, 40, None), 27.5);
            assert_eq!(interpolate(&values, 95, None), 50.0);
        }
        {
            let values = vec![2.0, 4.0];
            assert_eq!(interpolate(&values, 0, Some(1.0)), 2.0);
            assert_eq!(interpolate(&values, 10, Some(1.0)), 2.2);
            assert_eq!(interpolate(&values, 25, Some(1.0)), 2.5);
            assert_eq!(interpolate(&values, 75, Some(1.0)), 3.5);
            assert_eq!(interpolate(&values, 100, Some(1.0)), 4.0);
        }
    }

    #[test]
    fn test_get_percentiles() {
        let mut values = vec![1000.0, 20.0, 100.0];
        let quantiles = vec![50];
        assert_eq!(get_percentiles(&mut values, &quantiles, None), 100.0);
    }
}

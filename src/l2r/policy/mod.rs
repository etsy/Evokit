//! Policy
//! ---
//!
//! This module contains the policy for ranking a list of documents
//!
//! Types of Policies:
//! - Random (for baselines)
//! - Pointwise
//! - BeamSearch
//!
//! You'll add them to the scoring config.
//!
//! ```json
//!    { "policy": {
//!        "BeamSearch": {
//!          "num_candidates": 3
//!        }
//!      }
//!    },
//! ```
extern crate es_core;
extern crate es_data;

use self::es_data::dataset::Grouping;
use self::es_data::datatypes::Sparse;
use l2r::aggregator::utils::AggBuilder;
use l2r::load::LastPassConfig;

use self::es_core::model::Evaluator;
use crate::l2r::utils::get_float_field;

/// Policy trait
pub mod utils;

/// Random policy
pub mod random;

/// Pointwise policy
pub mod pointwise;

/// Beam policy
pub mod beam;

use crate::l2r::policy::beam::BeamSearchPolicy;
use crate::l2r::policy::pointwise::PointwisePolicy;
use crate::l2r::policy::random::RandomPolicy;
use crate::l2r::policy::utils::*;

/// Defines a wrapper for the policy
pub enum PolicyWrapper<B> {
    /// Random
    Random(RandomPolicy),
    /// Pointwise
    Pointwise(PointwisePolicy),
    /// Beam
    BeamSearch(BeamSearchPolicy<B>),
}

/// Defines a policy and any last pass interventions
pub struct PolicySet<B> {
    /// Wrapped policy
    pub policy_wrapper: PolicyWrapper<B>,
    /// Last pass config
    pub last_pass: Option<LastPassConfig>,
}

impl<B: AggBuilder> Policy for PolicySet<B> {
    /// Lets the upstream environment know that the evaluation is stochastic
    fn is_stochastic(&self) -> bool {
        use self::PolicyWrapper::*;
        match &self.policy_wrapper {
            Random(ref p) => p.is_stochastic(),
            Pointwise(ref p) => p.is_stochastic(),
            BeamSearch(ref p) => p.is_stochastic(),
        }
    }

    /// Runs the wrapped policy to sort the documents then runs last pass
    fn evaluate<M: Evaluator<Sparse, f32>>(
        &self,
        state: &M,
        rs: &Grouping<Sparse>,
        seed: u32,
        idx: usize,
    ) -> (Vec<usize>, Option<Vec<f32>>) {
        use self::PolicyWrapper::*;
        // Let the underlying policy run
        // TODO: in order to run fitness shaping, we will need to return the id & the score (what score should we use for greedy?)
        let (sorted_ids, scores) = match &self.policy_wrapper {
            Random(ref p) => p.evaluate(state, rs, seed, idx),
            Pointwise(ref p) => p.evaluate(state, rs, seed, idx),
            BeamSearch(ref p) => p.evaluate(state, rs, seed, idx),
        };

        // Run any interventions
        match self.last_pass {
            // Partition vector based on metadata
            Some(LastPassConfig::Boost(ref boost_config)) => {
                let ids_and_last_pass: Vec<_> = sorted_ids
                    .into_iter()
                    .map(|i| {
                        (
                            i,
                            get_float_field(&rs.metadata[i], &boost_config.field_name),
                        )
                    })
                    .collect();
                // Partition the documents based on the last pass field
                let (boost_candidates, _others): (Vec<_>, Vec<_>) = ids_and_last_pass
                    .iter()
                    .partition(|(_i, b)| b > &boost_config.greater_than_threshold);
                // Only grab up to the maximum number needed to boost
                let number_to_boost =
                    std::cmp::min(boost_config.max_number_to_boost, boost_candidates.len());
                let mut boosted_ids: Vec<_> = boost_candidates[0..number_to_boost]
                    .iter()
                    .map(|(i, _b)| *i)
                    .collect();
                // Keep the remaining documents in the original sorted order
                let mut remaining_ids: Vec<_> = ids_and_last_pass
                    .iter()
                    .filter(|(i, _b)| !boosted_ids.contains(i))
                    .map(|(i, _b)| *i)
                    .collect();

                // Add the remaining ids after the boosted ones
                boosted_ids.append(&mut remaining_ids);
                (boosted_ids, None)
            }
            _ => (sorted_ids, scores),
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate es_models;
    use self::es_data::datatypes::Dense;
    use self::es_models::linear::DenseWrapper;
    use super::*;

    use self::es_data::dataset::types::{MetaType, Metadata};
    use l2r::aggregator::count::CountAggBuilder;
    use l2r::load::BoostConfig;

    #[test]
    fn test_pointwise_with_last_pass() {
        // verify we sort in descending order
        let last_pass = Some(LastPassConfig::Boost(BoostConfig {
            field_name: "is_inc".to_string(),
            greater_than_threshold: 0.0,
            max_number_to_boost: 2,
        }));
        let base_policy = PointwisePolicy::new(false);
        let policy_wrapper: PolicyWrapper<CountAggBuilder> = PolicyWrapper::Pointwise(base_policy);
        let policy = PolicySet {
            policy_wrapper,
            last_pass,
        };

        // Identity on a single value
        let state = DenseWrapper { w: Dense(vec![1.]) };
        {
            let x = vec![
                Sparse(1, vec![0], vec![-1.]),
                Sparse(1, vec![0], vec![0.]),
                Sparse(1, vec![0], vec![2.]),
            ];

            let y = vec![1.0; 3];

            let metadata1: Metadata = [("is_inc".to_string(), MetaType::Num(1.0))]
                .iter()
                .cloned()
                .collect();
            let metadata2: Metadata = [("is_inc".to_string(), MetaType::Num(0.0))]
                .iter()
                .cloned()
                .collect();
            let metadata3: Metadata = [("is_inc".to_string(), MetaType::Num(1.0))]
                .iter()
                .cloned()
                .collect();
            let md = vec![metadata1, metadata2, metadata3];
            let rs = Grouping::new_with_md(x, y, md);

            let (sorted_ids, _scores) = policy.evaluate(&state, &rs, 1234, 0);
            assert_eq!(sorted_ids, vec![2, 0, 1]);
        }
        {
            let x = vec![
                Sparse(1, vec![0], vec![-1.]),
                Sparse(1, vec![0], vec![0.]),
                Sparse(1, vec![0], vec![2.]),
                Sparse(1, vec![0], vec![-3.]),
            ];

            let y = vec![1.0; 4];

            let metadata1: Metadata = [("is_inc".to_string(), MetaType::Num(1.0))]
                .iter()
                .cloned()
                .collect();
            let metadata2: Metadata = [("is_inc".to_string(), MetaType::Num(0.0))]
                .iter()
                .cloned()
                .collect();
            let metadata3: Metadata = [("is_inc".to_string(), MetaType::Num(1.0))]
                .iter()
                .cloned()
                .collect();
            let metadata4: Metadata = [("is_inc".to_string(), MetaType::Num(1.0))]
                .iter()
                .cloned()
                .collect();
            let md = vec![metadata1, metadata2, metadata3, metadata4];
            let rs = Grouping::new_with_md(x, y, md);

            let (sorted_ids, _scores) = policy.evaluate(&state, &rs, 1234, 0);
            assert_eq!(sorted_ids, vec![2, 0, 1, 3]);
        }
    }
}

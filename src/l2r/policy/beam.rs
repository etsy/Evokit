extern crate es_core;
extern crate es_data;
extern crate rand;
extern crate rand_xorshift;

use std::mem;

use self::rand::distributions::{Distribution, Uniform};
use self::rand::seq::SliceRandom;
use self::rand::SeedableRng;
use self::rand_xorshift::XorShiftRng;

use std::cmp::Ordering;
use std::collections::binary_heap::{BinaryHeap, PeekMut};

use self::es_core::model::Evaluator;
use self::es_data::dataset::Grouping;
use self::es_data::datatypes::Sparse;
use l2r::aggregator::utils::{AggBuilder, Aggregator};

use crate::l2r::policy::utils::*;

#[derive(Clone)]
/// Defines an ordering of documents that we could output
struct CandidateOrdering<A> {
    /// Score from the underlying model
    score: f32,
    /// Aggregator for the docs in the current path
    aggregator: A,
    /// Docs selected for the current path
    current_path: Vec<usize>,
    /// Docs that still need to be selected
    remaining: Vec<usize>,
}

impl<A: Aggregator> PartialEq for CandidateOrdering<A> {
    /// Defines if two CandidateOrderings are equal
    fn eq(&self, other: &Self) -> bool {
        self.current_path == other.current_path
    }
}

impl<A: Aggregator> Eq for CandidateOrdering<A> {}

impl<A: Aggregator> PartialOrd for CandidateOrdering<A> {
    /// Partially compares two orderings
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.score.partial_cmp(&self.score)
    }
}

impl<A: Aggregator> Ord for CandidateOrdering<A> {
    /// Compares two orderings
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// Policy for running beam search
pub struct BeamSearchPolicy<B> {
    /// Indicates how to combine information across docs
    agg_builder: B,
    /// Whether this is a stochastic policy
    stochastic: bool,
    /// Whether to subset the docs before running the search
    subset: Option<usize>,
    /// Seed
    seed: Option<u32>,
    /// Number of candidates to explore
    num_candidates: usize,
}

impl<B: AggBuilder> BeamSearchPolicy<B> {
    /// Creates a new BeamSearchPolicy
    pub fn new(
        builder: B,
        subset: Option<usize>,
        seed: Option<u32>,
        stochastic: bool,
        num_candidates: Option<usize>,
    ) -> Self {
        BeamSearchPolicy {
            agg_builder: builder,
            stochastic: stochastic,
            subset: subset,
            seed: seed,
            num_candidates: num_candidates.unwrap_or(1),
        }
    }

    /// Updates the scores and paths for an Ordering
    /// This is used to avoid having to allocate more memory for new orderings. Instead we use old orderings
    fn update_candidate(
        &self,
        candidate: &mut PeekMut<CandidateOrdering<B::Agg>>,
        remaining_in: &Vec<usize>,
        current_path_in: &Vec<usize>,
        score_in: f32,
        added_index: &usize,
        aggregator_in: B::Agg,
    ) -> () {
        let score_out = &mut candidate.score;
        *score_out = score_in;

        candidate.aggregator = aggregator_in;

        let current_path_out = &mut candidate.current_path;
        current_path_out.clear();
        for idx in current_path_in {
            current_path_out.push(*idx);
        }
        current_path_out.push(*added_index);

        let remaining_out = &mut candidate.remaining;
        remaining_out.clear();
        for idx in remaining_in {
            if idx != added_index {
                remaining_out.push(*idx);
            }
        }
    }

    /// Updates the starting candidates
    /// This is used to avoid having to allocate more memory for new orderings. Instead we use old orderings
    fn update_starting_candidates(
        &self,
        starting_candidates: &mut Vec<CandidateOrdering<B::Agg>>,
        updated_candidates: &mut Vec<CandidateOrdering<B::Agg>>,
    ) -> () {
        // number of starting candidates will always be less than the updated candidates
        for (i, base_candidate) in starting_candidates.iter_mut().enumerate() {
            mem::swap(
                &mut base_candidate.aggregator,
                &mut updated_candidates[i].aggregator,
            );
            mem::swap(&mut base_candidate.score, &mut updated_candidates[i].score);
            mem::swap(
                &mut base_candidate.current_path,
                &mut updated_candidates[i].current_path,
            );
            mem::swap(
                &mut base_candidate.remaining,
                &mut updated_candidates[i].remaining,
            );
        }
        // This clone will only happen on the first time
        if self.num_candidates > starting_candidates.len() {
            let start_index = starting_candidates.len();
            for i in start_index..updated_candidates.len() {
                starting_candidates.push(updated_candidates[i].clone());
            }
        }
    }

    /// Expands the starting candidates to find new candidate orderings
    fn explore_candidates<M: Evaluator<Sparse, f32>>(
        &self,
        state: &M,
        current_candidates: &Vec<CandidateOrdering<B::Agg>>,
        updated_candidates: &mut BinaryHeap<CandidateOrdering<B::Agg>>,
        mut delta_vec: &mut Sparse,
        rs: &Grouping<Sparse>,
        bit: f32,
    ) -> () {
        for current_ordering in current_candidates.iter() {
            // Loop over the remaining indices, evaluate with the given anchor
            for idx in current_ordering.remaining.iter() {
                let row = &rs.x[*idx];
                row.combine_into(
                    &current_ordering.aggregator.read(),
                    #[inline]
                    |l, r| l.unwrap_or(0.) - r.unwrap_or(0.),
                    &mut delta_vec,
                );
                if self.stochastic {
                    let vec_idxs = &mut delta_vec.1;
                    vec_idxs.push(delta_vec.0 - 1);
                    let vec_values = &mut delta_vec.2;
                    vec_values.push(bit);
                    let vec_size = &mut delta_vec.0;
                    *vec_size += 1;
                }
                let score = state.evaluate(&delta_vec);

                // updated_candidates is already initialized to the size we need.
                if let Some(mut min_candidate) = updated_candidates.peek_mut() {
                    if min_candidate.score <= score {
                        // TODO: can we avoid cloning the aggregator too?
                        let mut aggregator = current_ordering.aggregator.clone();
                        aggregator.update(row);

                        self.update_candidate(
                            &mut min_candidate,
                            &current_ordering.remaining,
                            &current_ordering.current_path,
                            score,
                            idx,
                            aggregator,
                        );
                    }
                }
            }
        }
    }
}

impl<B: AggBuilder> Policy for BeamSearchPolicy<B> {
    /// Lets the upstream environment know that the evaluation is stochastic
    fn is_stochastic(&self) -> bool {
        self.seed.is_none() || self.stochastic
    }

    /// Runs beam search to get an ordered list of documents
    fn evaluate<M: Evaluator<Sparse, f32>>(
        &self,
        state: &M,
        rs: &Grouping<Sparse>,
        seed: u32,
        idx: usize,
    ) -> (Vec<usize>, Option<Vec<f32>>) {
        let mut remaining: Vec<usize> = (0..rs.len()).collect();

        // To speed up computation, we can choose a subset of results to optimize for
        if let Some(sub_k) = self.subset {
            let seed = self.seed.unwrap_or(seed << 2 + 1) as u64;
            let mut rng = XorShiftRng::seed_from_u64(seed);
            remaining.as_mut_slice().shuffle(&mut rng);
            // TODO: will remaining ever be longer than subset if we truncate???
            remaining.truncate(sub_k);
        }

        // For stochastic policy
        let mut rng = XorShiftRng::seed_from_u64(seed as u64 + idx as u64);
        let dist = Uniform::new_inclusive(-1f32, 1.);
        let bit = dist.sample(&mut rng) as f32;

        let total_iterations = self.subset.unwrap_or(remaining.len()).min(remaining.len());
        // Initialize the starting candidates with only one candidate but with the capacity for more
        let mut starting_candidates = Vec::with_capacity(self.num_candidates);
        starting_candidates.push(CandidateOrdering {
            score: std::f32::NEG_INFINITY,
            aggregator: self.agg_builder.start(),
            current_path: Vec::with_capacity(remaining.len()),
            remaining: remaining.clone(),
        });
        // Initialize the candidates that we will modify will -inf score so they will be replaced.
        // current_path & remaining don't matter as we will overwrite them
        let mut updated_candidates: BinaryHeap<_> = vec![
            CandidateOrdering {
                score: std::f32::NEG_INFINITY,
                aggregator: self.agg_builder.start(),
                current_path: Vec::with_capacity(remaining.len()),
                remaining: Vec::with_capacity(remaining.len()),
            };
            self.num_candidates
        ]
        .into();

        let mut delta_vec = Sparse(rs.x[0].0, vec![], vec![]);
        for _i in 0..total_iterations {
            self.explore_candidates(
                state,
                &starting_candidates,
                &mut updated_candidates,
                &mut delta_vec,
                rs,
                bit,
            );
            // TODO: does converting to a vector allocate more memory or just change the interface?
            let mut updated_candidate_vec = updated_candidates.into_vec();
            self.update_starting_candidates(&mut starting_candidates, &mut updated_candidate_vec);

            // Set the updated candidate scores to -Inf
            for mut candidate in updated_candidate_vec.iter_mut() {
                candidate.score = std::f32::NEG_INFINITY;
            }
            updated_candidates = updated_candidate_vec.into();
        }

        // it's ranked in ascending order
        let winning_candidate = starting_candidates.pop();
        let (mut final_ordering, remaining_after_search) = winning_candidate
            .map(|x| (x.current_path, x.remaining))
            .unwrap_or((vec![], vec![]));

        // TODO: do we need to add these ids?
        // Add the remaining idxs, if needed
        for idx in remaining_after_search.into_iter() {
            final_ordering.push(idx);
        }
        (final_ordering, None)
    }
}

#[cfg(test)]
mod tests {
    extern crate es_models;

    use self::es_data::datatypes::Dense;
    use self::es_models::linear::DenseWrapper;
    use super::*;
    use l2r::aggregator::total::TotalAggBuilder;

    fn assert_candidate_ordering_is_valid<A: Aggregator>(
        candidate: &CandidateOrdering<A>,
        current_path: Vec<usize>,
        remaining: Vec<usize>,
        agg_size: usize,
        agg_dims: Vec<usize>,
        agg_values: Vec<f32>,
    ) {
        assert_eq!(candidate.current_path, current_path);
        assert_eq!(candidate.remaining, remaining);

        let aggregator = candidate.aggregator.read();
        assert_eq!(aggregator.0, agg_size);
        assert_eq!(aggregator.1, agg_dims);
        assert_eq!(aggregator.2, agg_values);
    }

    #[test]
    fn test_beam_explore_candidates() {
        // Identity on a single value
        let x = vec![
            Sparse(1, vec![0], vec![-1.]),
            Sparse(1, vec![0], vec![0.]),
            Sparse(1, vec![0], vec![5.]),
            Sparse(1, vec![0], vec![7.]),
        ];
        let y = vec![1.0; 4];
        let rs = Grouping::new(x, y);

        {
            let state = DenseWrapper { w: Dense(vec![1.]) };
            let builder = TotalAggBuilder(1);
            let policy = BeamSearchPolicy::new(builder, None, None, false, Some(2));
            let mut aggregator = policy.agg_builder.start();
            let previous_winner = &rs.x[3];
            aggregator.update(previous_winner);
            let starting_candidates = vec![CandidateOrdering {
                score: 0.4,
                aggregator: aggregator,
                current_path: vec![3],
                remaining: vec![0, 1, 2],
            }];
            let mut updated_candidates: BinaryHeap<_> = vec![
                CandidateOrdering {
                    score: std::f32::NEG_INFINITY,
                    aggregator: policy.agg_builder.start(),
                    current_path: vec![3],
                    remaining: vec![0, 1, 2]
                };
                2
            ]
            .into();
            let mut delta_vec = Sparse(rs.x[0].0, vec![], vec![]);
            policy.explore_candidates(
                &state,
                &starting_candidates,
                &mut updated_candidates,
                &mut delta_vec,
                &rs,
                0.,
            );
            let candidate_list = updated_candidates.into_sorted_vec();
            assert_eq!(candidate_list.len(), 2);
            // score = 5 - 7 = -2
            assert_candidate_ordering_is_valid(
                &candidate_list[0],
                vec![3, 2],
                vec![0, 1],
                1,
                vec![0],
                vec![12.],
            );

            // score 0 - 7 = -7
            assert_candidate_ordering_is_valid(
                &candidate_list[1],
                vec![3, 1],
                vec![0, 2],
                1,
                vec![0],
                vec![7.],
            );
        }

        {
            // extra bit for the identify function because this is stochastic
            let state = DenseWrapper {
                w: Dense(vec![1., 1.]),
            };
            let builder = TotalAggBuilder(1);
            let policy = BeamSearchPolicy::new(builder, None, None, true, Some(2));
            let mut aggregator = policy.agg_builder.start();
            let previous_winner = &rs.x[3];
            aggregator.update(previous_winner);
            let starting_candidates = vec![CandidateOrdering {
                score: 0.4,
                aggregator: aggregator,
                current_path: vec![3],
                remaining: vec![0, 1, 2],
            }];
            let mut updated_candidates: BinaryHeap<_> = vec![
                CandidateOrdering {
                    score: std::f32::NEG_INFINITY,
                    aggregator: policy.agg_builder.start(),
                    current_path: vec![3],
                    remaining: vec![0, 1, 2]
                };
                2
            ]
            .into();
            let mut delta_vec = Sparse(rs.x[0].0, vec![], vec![]);
            policy.explore_candidates(
                &state,
                &starting_candidates,
                &mut updated_candidates,
                &mut delta_vec,
                &rs,
                0.,
            );
            let candidate_list = updated_candidates.into_sorted_vec();
            assert_eq!(candidate_list.len(), 2);
            assert_candidate_ordering_is_valid(
                &candidate_list[0],
                vec![3, 2],
                vec![0, 1],
                1,
                vec![0],
                vec![12.],
            );

            assert_candidate_ordering_is_valid(
                &candidate_list[1],
                vec![3, 1],
                vec![0, 2],
                1,
                vec![0],
                vec![7.],
            );
        }
    }

    #[test]
    fn test_beam_selects_best_candidates() {
        // Identity on a single value
        let x = vec![
            Sparse(1, vec![0], vec![-1.]),
            Sparse(1, vec![0], vec![0.]),
            Sparse(1, vec![0], vec![5.]),
            Sparse(1, vec![0], vec![7.]),
        ];
        let y = vec![1.0; 4];
        let rs = Grouping::new(x, y);

        let state = DenseWrapper { w: Dense(vec![1.]) };
        let builder = TotalAggBuilder(1);
        let policy = BeamSearchPolicy::new(builder, None, None, false, Some(2));

        // given multiple options, only keep the best 2
        let mut aggregator1 = policy.agg_builder.start();
        let previous_winner1 = &rs.x[3];
        aggregator1.update(previous_winner1);

        let mut aggregator2 = policy.agg_builder.start();
        let previous_winner2 = &rs.x[2];
        aggregator2.update(previous_winner2);
        let starting_candidates = vec![
            CandidateOrdering {
                score: 0.4,
                aggregator: aggregator1,
                current_path: vec![3],
                remaining: vec![0, 1, 2],
            },
            CandidateOrdering {
                score: 0.3,
                aggregator: aggregator2,
                current_path: vec![2],
                remaining: vec![0, 1, 3],
            },
        ];
        let mut updated_candidates: BinaryHeap<_> = vec![
            CandidateOrdering {
                score: std::f32::NEG_INFINITY,
                aggregator: policy.agg_builder.start(),
                current_path: vec![],
                remaining: vec![0, 1, 2]
            };
            2
        ]
        .into();
        let mut delta_vec = Sparse(rs.x[0].0, vec![], vec![]);
        policy.explore_candidates(
            &state,
            &starting_candidates,
            &mut updated_candidates,
            &mut delta_vec,
            &rs,
            0.,
        );
        let candidate_list = updated_candidates.into_sorted_vec();
        assert_eq!(candidate_list.len(), 2);

        // score = 7 - 5 = 2
        assert_candidate_ordering_is_valid(
            &candidate_list[0],
            vec![2, 3],
            vec![0, 1],
            1,
            vec![0],
            vec![12.],
        );

        // score = 5 - 7 = -2
        assert_candidate_ordering_is_valid(
            &candidate_list[1],
            vec![3, 2],
            vec![0, 1],
            1,
            vec![0],
            vec![12.],
        );
    }

    #[test]
    fn test_beam_evaluate() {
        let x = vec![
            Sparse(1, vec![0], vec![0.]),
            Sparse(1, vec![0], vec![1.]),
            Sparse(1, vec![0], vec![2.]),
            Sparse(1, vec![0], vec![3.]),
            Sparse(1, vec![0], vec![4.]),
            Sparse(1, vec![0], vec![5.]),
            Sparse(1, vec![0], vec![6.]),
            Sparse(1, vec![0], vec![7.]),
        ];
        let y = vec![1.0; 3];
        let rs = Grouping::new(x, y);

        let state = DenseWrapper { w: Dense(vec![1.]) };
        {
            let builder = TotalAggBuilder(1);
            let policy = BeamSearchPolicy::new(builder, None, None, false, None);
            let (sorted_ids, _scores) = policy.evaluate(&state, &rs, 1234, 0);
            assert_eq!(sorted_ids, vec![7, 6, 5, 4, 3, 2, 1, 0]);
        }
        {
            let builder = TotalAggBuilder(1);
            let policy = BeamSearchPolicy::new(builder, None, None, false, Some(1));
            let (sorted_ids, _scores) = policy.evaluate(&state, &rs, 1234, 0);
            assert_eq!(sorted_ids, vec![7, 6, 5, 4, 3, 2, 1, 0]);
        }
        {
            let builder = TotalAggBuilder(1);
            let policy = BeamSearchPolicy::new(builder, None, None, false, Some(2));
            let (sorted_ids, _scores) = policy.evaluate(&state, &rs, 1234, 0);
            assert_eq!(sorted_ids, vec![6, 7, 5, 4, 3, 2, 1, 0]);
        }
        {
            let builder = TotalAggBuilder(1);
            let policy = BeamSearchPolicy::new(builder, None, None, false, Some(3));
            let (sorted_ids, _scores) = policy.evaluate(&state, &rs, 1234, 0);
            // higher is better
            // current_path: [5, 6, 4, 2, 3, 0, 1], idx: 7, score: -14.0, min score: -inf
            // current_path: [5, 6, 4, 2, 3, 0, 7], idx: 1, score: -26.0, min score: -inf
            // current_path: [5, 6, 4, 2, 3, 1, 7], idx: 0, score: -28.0, min score: -inf
            assert_eq!(sorted_ids, vec![5, 6, 4, 2, 3, 0, 1, 7]);
        }
    }

    #[test]
    fn test_beam_evaluate_subset() {
        let x = vec![
            // Only 0, 1, 2 will be used
            Sparse(1, vec![0], vec![0.]),
            Sparse(1, vec![0], vec![1.]),
            Sparse(1, vec![0], vec![2.]),
            Sparse(1, vec![0], vec![3.]),
            Sparse(1, vec![0], vec![4.]),
            Sparse(1, vec![0], vec![5.]),
            Sparse(1, vec![0], vec![6.]),
            Sparse(1, vec![0], vec![7.]),
        ];
        let y = vec![1.0; 3];
        let rs = Grouping::new(x, y);

        let state = DenseWrapper { w: Dense(vec![1.]) };
        {
            let builder = TotalAggBuilder(1);
            let policy = BeamSearchPolicy::new(builder, Some(3), None, false, Some(2));
            let (sorted_ids, _scores) = policy.evaluate(&state, &rs, 1234, 0);
            assert_eq!(sorted_ids, vec![2, 1, 0]);
        }
    }
}

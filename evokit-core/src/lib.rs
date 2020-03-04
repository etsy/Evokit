//! Evokit-Core
//! ===
//!
//! This library contains all the necessary components for optimizing models using
//! black-box optimization methods.  Out of the box, it contains two main optimizers:
//! Evolutionary Strategies (ES) and Differential Evolution (DE).
//!
//! Evolutionary Strategies
//! ---
//!
//! This provides three flavors of ES for optimization, with differnt tradeoffs in terms
//! of speed and accuracy.
//!
//! Simple
//! ---
//! This implements a simple (1+1) Evolutionary Strategies.  This in effect
//! can be thought of as stochastic hill climbing: if a gradient is found which is an
//! improved solution, it use that solution.  Otherwise it will keep the existing
//! solution.  This is useful for simple models when speed is paramount.
//!
//! Canonical Evolutionary Strategies
//! ---
//! This implements a (1+λ) Evolutionary Strategies, where the λ children are generated
//! for each parent and the top K are blended to form a new gradient.  It offers both
//! elitism as well as optional always update.
//!
//! Natural Evolutionary Strategies
//! ---
//! This implements a (1+λ) Evolutionary Strategies where the gradient computatation is
//! determined by a weighted combination of all candidate gradients.  This particular
//! implementation uses a fixed α for learning rate ala Salimans et al.  It offers
//! optional Fitness Shaping (Wiestra et al.) which can be useful for escaping early
//! minima.
//!
//!
//! Differential Evolution
//! ---
//! This implements a number of blending strategies for optimizing solutions.
//!

#![warn(missing_docs, unused)]

/// Defines the interfaces for Model types for use in neuro-evolution optimizers.
pub mod model;

/// Defines interfaces for Environments and States
pub mod optimizer;

/// Defines the Canonical ES optimizer
pub mod canonical;

/// Defines a simple (1+1)-ES optimizer
pub mod simple;

/// Defines an optimizer rbased on a variation on Natural Evolutionary Strategies
pub mod nes;

/// Defines the Differential Evolution optimizer.
pub mod evolution;

use std::ops::*;

use crate::intrinsics::scale;

#[cfg(test)]
/// Method to compare two f32 values. This is for testing
pub fn cmp_f32_vec(l: &[f32], r: &[f32], eps: f32) -> bool {
    assert_eq!(l.len(), r.len());
    for i in 0..l.len() {
        if (l[i] - r[i]).abs() > eps {
            return false;
        }
    }
    true
}

/// Defines dense vectors
#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct Dense(pub Vec<f32>);

/// Sparse datatype
#[derive(Debug, Clone)]
pub struct Sparse(pub usize, pub Vec<usize>, pub Vec<f32>);

// Shouldn't use this in practice as it's expensive, but useful for testing
#[cfg(test)]
impl PartialEq for Sparse {
    /// Compare two sparse vectors
    fn eq(&self, other: &Sparse) -> bool {
        if self.0 == other.0 && self.1 == other.1 {
            cmp_f32_vec(&self.2, &other.2, 1e-5)
        } else {
            false
        }
    }
}

impl Sparse {
    /// This combines two sparse vectors with a joining function by saving into `out`
    pub fn combine_into<F>(&self, right: &Sparse, f: F, out: &mut Sparse)
    where
        F: Fn(Option<f32>, Option<f32>) -> f32,
    {
        out.0 = self.0.max(right.0);
        let idxs = &mut out.1;
        idxs.clear();
        let vals = &mut out.2;
        vals.clear();
        let mut i = 0;
        let mut j = 0;

        {
            let add_non_empty = &mut |idx: usize, v: f32| {
                if v != 0.0 {
                    idxs.push(idx);
                    vals.push(v);
                }
            };

            while i < self.1.len() && j < right.1.len() {
                // Feature isn't in right
                if self.1[i] < right.1[j] {
                    let v = f(Some(self.2[i]), None);
                    add_non_empty(self.1[i], v);
                    i += 1;

                // Feature isn't in self
                } else if self.1[i] > right.1[j] {
                    let v = f(None, Some(right.2[j]));
                    add_non_empty(right.1[j], v);
                    j += 1;
                } else {
                    let v = f(Some(self.2[i]), Some(right.2[j]));
                    add_non_empty(self.1[i], v);
                    i += 1;
                    j += 1;
                }
            }

            for idx in i..(self.1.len()) {
                let v = f(Some(self.2[idx]), None);
                add_non_empty(self.1[idx], v);
            }

            for idx in j..(right.1.len()) {
                let v = f(None, Some(right.2[idx]));
                add_non_empty(right.1[idx], v);
            }
        }
    }

    /// combines the current vector with `right` by applying F and outputs that vector
    pub fn combine<F>(&self, right: &Sparse, f: F) -> Sparse
    where
        F: Fn(Option<f32>, Option<f32>) -> f32,
    {
        let mut out = Sparse(0, vec![], vec![]);
        self.combine_into(right, f, &mut out);
        out
    }

    /// Converts a sparse vector to a dense vector
    pub fn to_dense(&self) -> Dense {
        let mut d = vec![0.; self.0];
        for i in 0..self.1.len() {
            d[self.1[i]] = self.2[i];
        }
        Dense(d)
    }
}

impl<'a, 'b> Add<&'a Sparse> for &'b Sparse {
    type Output = Sparse;

    /// Add two sparse vectors
    fn add(self, other: &'a Sparse) -> Self::Output {
        self.combine(
            other,
            #[inline]
            |l, r| l.unwrap_or(0.) + r.unwrap_or(0.),
        )
    }
}

impl<'a> AddAssign<&'a Sparse> for Sparse {
    fn add_assign(&mut self, other: &'a Sparse) {
        *self = (self as &Sparse) + other;
    }
}

impl<'a, 'b> Mul<&'a Sparse> for &'b Sparse {
    type Output = Sparse;

    fn mul(self, other: &'a Sparse) -> Self::Output {
        self.combine(
            other,
            #[inline]
            |l, r| l.unwrap_or(0.) * r.unwrap_or(0.),
        )
    }
}

impl<'a> MulAssign<&'a Sparse> for Sparse {
    fn mul_assign(&mut self, other: &'a Sparse) {
        *self = (self as &Sparse) * other;
    }
}

impl Mul<f32> for &Sparse {
    type Output = Sparse;

    fn mul(self, other: f32) -> Self::Output {
        let mut out = self.clone();
        scale(&mut out.2, other);
        out
    }
}

impl MulAssign<f32> for Sparse {
    fn mul_assign(&mut self, other: f32) {
        scale(&mut self.2, other);
    }
}

impl Div<f32> for &Sparse {
    type Output = Sparse;

    fn div(self, other: f32) -> Self::Output {
        let mut out = self.clone();
        scale(&mut out.2, 1. / other);
        out
    }
}

impl DivAssign<f32> for Sparse {
    fn div_assign(&mut self, other: f32) {
        scale(&mut self.2, 1. / other);
    }
}

impl<'a, 'b> Div<&'a Sparse> for &'b Sparse {
    type Output = Sparse;

    fn div(self, other: &'a Sparse) -> Self::Output {
        // Divide by zero in floating point land is actually ok, although the error
        // is rarely useful.  Still, to conform to the Div spec, we maintain the
        // standard brokeness
        self.combine(
            other,
            #[inline]
            |l, r| l.unwrap_or(0.) / r.unwrap_or(0.),
        )
    }
}

impl<'a> DivAssign<&'a Sparse> for Sparse {
    fn div_assign(&mut self, other: &'a Sparse) {
        *self = (self as &Sparse) / other;
    }
}

impl<'a, 'b> Sub<&'a Sparse> for &'b Sparse {
    type Output = Sparse;

    fn sub(self, other: &'a Sparse) -> Self::Output {
        self.combine(
            other,
            #[inline]
            |l, r| l.unwrap_or(0.) - r.unwrap_or(0.),
        )
    }
}

impl<'a> SubAssign<&'a Sparse> for Sparse {
    fn sub_assign(&mut self, other: &'a Sparse) {
        *self = (self as &Sparse) - other;
    }
}

/// Train to define the size of a vector
pub trait Dimension {
    /// Dimension type. usually usize
    type Out;
    /// Get the dimension
    fn dims(&self) -> Self::Out;
}

impl Dimension for Vec<f32> {
    type Out = usize;
    fn dims(&self) -> Self::Out {
        self.len()
    }
}

impl Dimension for Sparse {
    type Out = usize;
    fn dims(&self) -> Self::Out {
        self.0
    }
}

#[cfg(test)]
mod test_datatypes {
    use super::*;

    #[test]
    fn test_sparse_ops() {
        let s1 = Sparse(10, vec![1, 4, 7, 8], vec![1., 4., 7., 2.]);
        let s2 = Sparse(10, vec![1, 3, 7, 8], vec![-2., 3., 1., 2.]);

        let res = &s1 + &s2;
        assert_eq!(
            res,
            Sparse(10, vec![1, 3, 4, 7, 8], vec![-1., 3., 4., 8., 4.])
        );

        let res = &s1 - &s2;
        assert_eq!(res, Sparse(10, vec![1, 3, 4, 7], vec![3., -3., 4., 6.]));

        let res = &s1 / 10.;
        assert_eq!(res, Sparse(10, vec![1, 4, 7, 8], vec![0.1, 0.4, 0.7, 0.2]));

        let res = &s2 * 10.;
        assert_eq!(res, Sparse(10, vec![1, 3, 7, 8], vec![-20., 30., 10., 20.]));
    }
}

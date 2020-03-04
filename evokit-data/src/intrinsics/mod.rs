//! Helper functions for intrinsics
use std::arch::x86_64::*;
use std::mem;

/// AVX based summation
pub fn sum(v1: &[f32], v2: &[f32], v3: &mut [f32]) -> () {
    assert_eq!(v1.len(), v2.len());
    assert_eq!(v1.len(), v3.len());
    let mut i = 0;
    unsafe {
        let mut extract: [f32; 8] = mem::MaybeUninit::uninit().assume_init();
        if v1.len() >= 8 {
            while i < (v1.len() - 8) {
                let l = _mm256_set_ps(
                    v1[i],
                    v1[i + 1],
                    v1[i + 2],
                    v1[i + 3],
                    v1[i + 4],
                    v1[i + 5],
                    v1[i + 6],
                    v1[i + 7],
                );
                let r = _mm256_set_ps(
                    v2[i],
                    v2[i + 1],
                    v2[i + 2],
                    v2[i + 3],
                    v2[i + 4],
                    v2[i + 5],
                    v2[i + 6],
                    v2[i + 7],
                );
                let res = _mm256_add_ps(l, r);

                // Extract
                _mm256_store_ps(&mut extract[0] as *mut f32, res);
                for j in 0..8 {
                    v3[i + j] = extract[7 - j];
                }
                i += 8;
            }
        }
    }

    // Remainder
    for i in i..v1.len() {
        v3[i] = v1[i] + v2[i]
    }
}

/// AVX based dot product
pub fn l2norm(v1: &[f32]) -> f32 {
    let res = dot(v1, v1);
    res.sqrt()
}

/// AVX based inplace summation
pub fn inplace_sum(v1: &mut [f32], v2: &[f32]) -> () {
    assert_eq!(v1.len(), v2.len());
    let mut i = 0;
    unsafe {
        if v1.len() >= 8 {
            let mut extract: [f32; 8] = mem::MaybeUninit::uninit().assume_init();
            while i < (v1.len() - 8) {
                let l = _mm256_set_ps(
                    v1[i],
                    v1[i + 1],
                    v1[i + 2],
                    v1[i + 3],
                    v1[i + 4],
                    v1[i + 5],
                    v1[i + 6],
                    v1[i + 7],
                );
                let r = _mm256_set_ps(
                    v2[i],
                    v2[i + 1],
                    v2[i + 2],
                    v2[i + 3],
                    v2[i + 4],
                    v2[i + 5],
                    v2[i + 6],
                    v2[i + 7],
                );
                let res = _mm256_add_ps(l, r);
                _mm256_store_ps(&mut extract[0] as *mut f32, res);
                for j in 0..8 {
                    v1[i + j] = extract[7 - j];
                }
                i += 8;
            }
        }
    }

    // Remainder
    for i in i..v1.len() {
        v1[i] += v2[i]
    }
}

/// AVX based dot product
pub fn dot(v1: &[f32], v2: &[f32]) -> f32 {
    assert_eq!(v1.len(), v2.len());
    let mut sum = 0.0;
    let mut i = 0;
    unsafe {
        if v1.len() >= 8 {
            let mut sv = _mm256_setzero_ps();
            let mut extract: [f32; 8] = mem::MaybeUninit::uninit().assume_init();
            while i < (v1.len() - 8) {
                let l = _mm256_set_ps(
                    v1[i],
                    v1[i + 1],
                    v1[i + 2],
                    v1[i + 3],
                    v1[i + 4],
                    v1[i + 5],
                    v1[i + 6],
                    v1[i + 7],
                );
                let r = _mm256_set_ps(
                    v2[i],
                    v2[i + 1],
                    v2[i + 2],
                    v2[i + 3],
                    v2[i + 4],
                    v2[i + 5],
                    v2[i + 6],
                    v2[i + 7],
                );
                sv = _mm256_fmadd_ps(l, r, sv);
                i += 8;
            }
            _mm256_store_ps(&mut extract[0] as *mut f32, sv);
            sum = extract[0]
                + extract[1]
                + extract[2]
                + extract[3]
                + extract[4]
                + extract[5]
                + extract[6]
                + extract[7];
        }
    }

    // Remainder
    for i in i..v1.len() {
        sum += v1[i] * v2[i];
    }
    sum
}

/// sse based dot product
pub fn dot_sse(v1: &[f32], v2: &[f32]) -> f32 {
    assert_eq!(v1.len(), v2.len());
    let mut sum: f32 = 0.0;
    let mut i = 0;
    unsafe {
        if v1.len() >= 4 {
            let mut zeros = _mm_setzero_ps();
            while i < (v1.len() - 4) {
                let l = _mm_set_ps(v1[i], v1[i + 1], v1[i + 2], v1[i + 3]);
                let r = _mm_set_ps(v2[i], v2[i + 1], v2[i + 2], v2[i + 3]);
                let res = _mm_dp_ps(l, r, 241);
                zeros = _mm_add_ps(res, zeros);
                i += 4;
            }
            let num = _mm_extract_ps(zeros, 0);
            sum += mem::transmute::<i32, f32>(num);
        }
    }

    // Remainder
    for i in i..v1.len() {
        sum += v1[i] * v2[i];
    }
    sum
}

/// AVX based dot product
pub fn sparse_dot(idx: &[usize], v1: &[f32], v2: &[f32]) -> f32 {
    assert_eq!(idx.len(), v1.len());
    assert!(idx.len() <= v2.len());
    let mut sum = 0.0;
    let mut i = 0;
    unsafe {
        if idx.len() >= 8 {
            let mut sv = _mm256_setzero_ps();
            let mut extract: [f32; 8] = mem::MaybeUninit::uninit().assume_init();
            while i < (idx.len() - 8) {
                let l = _mm256_set_ps(
                    v1[i],
                    v1[i + 1],
                    v1[i + 2],
                    v1[i + 3],
                    v1[i + 4],
                    v1[i + 5],
                    v1[i + 6],
                    v1[i + 7],
                );
                let r = _mm256_set_ps(
                    v2[idx[i]],
                    v2[idx[i + 1]],
                    v2[idx[i + 2]],
                    v2[idx[i + 3]],
                    v2[idx[i + 4]],
                    v2[idx[i + 5]],
                    v2[idx[i + 6]],
                    v2[idx[i + 7]],
                );
                sv = _mm256_fmadd_ps(l, r, sv);
                i += 8;
            }
            _mm256_store_ps(&mut extract[0] as *mut f32, sv);
            sum = extract[0]
                + extract[1]
                + extract[2]
                + extract[3]
                + extract[4]
                + extract[5]
                + extract[6]
                + extract[7];
        }
    }

    // Remainder
    for i in i..v1.len() {
        sum += v1[i] * v2[idx[i]];
    }
    sum
}

/// AVX based scale
pub fn scale(v1: &mut [f32], s: f32) -> () {
    let mut i = 0;
    unsafe {
        if v1.len() >= 8 {
            let sv = _mm256_set1_ps(s);
            let zero = _mm256_setzero_ps();
            let mut extract: [f32; 8] = mem::MaybeUninit::uninit().assume_init();
            while i < (v1.len() - 8) {
                let l = _mm256_set_ps(
                    v1[i],
                    v1[i + 1],
                    v1[i + 2],
                    v1[i + 3],
                    v1[i + 4],
                    v1[i + 5],
                    v1[i + 6],
                    v1[i + 7],
                );
                let res = _mm256_fmadd_ps(l, sv, zero);
                _mm256_store_ps(&mut extract[0] as *mut f32, res);
                for j in 0..8 {
                    v1[i + j] = extract[7 - j];
                }
                i += 8;
            }
        }
    }

    // Remainder
    for i in i..v1.len() {
        v1[i] *= s;
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    use self::test::Bencher;
    use super::*;

    #[test]
    fn test_sum() {
        let v1 = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        let mut res = [0.0; 10];
        sum(&v1, &v1, &mut res);
        assert_eq!(res, [2., 4., 6., 8., 10., 12., 14., 16., 18., 20.]);

        let v1 = [1., 2., 3., 4., 5.];
        let mut res = [0.0; 5];
        sum(&v1, &v1, &mut res);
        assert_eq!(res, [2., 4., 6., 8., 10.]);
    }

    #[test]
    fn test_dot() {
        let v1 = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        assert_eq!(dot(&v1, &v1), 385.);

        let v1 = [1., 2., 3., 4., 5.];
        let v2 = [0.0; 5];
        assert_eq!(dot(&v1, &v2), 0.);
    }

    #[test]
    fn test_l2norm() {
        let v1 = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        assert!(l2norm(&v1) - 19.6214168 < 1e-6);

        let v1 = [1.];
        assert_eq!(l2norm(&v1), 1.);
    }

    #[test]
    fn test_scale() {
        let mut v1 = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        scale(&mut v1, 2.);
        assert_eq!(v1, [2., 4., 6., 8., 10., 12., 14., 16., 18., 20.]);
    }

    #[bench]
    fn bench_sum_avx(b: &mut Bencher) {
        let v1: Vec<f32> = (0..200).map(|x| x as f32).collect();
        let mut v2 = v1.clone();
        b.iter(|| sum(&v1, &v1, &mut v2));
    }

    #[bench]
    fn bench_sum_standard(b: &mut Bencher) {
        let v1: Vec<f32> = (0..200).map(|x| x as f32).collect();
        let mut v2 = v1.clone();
        b.iter(|| {
            for i in 0..v1.len() {
                v2[i] = v1[i] + v1[i];
            }
        });
    }

    #[bench]
    fn bench_inp_sum_avx(b: &mut Bencher) {
        let mut v1: Vec<f32> = (0..200).map(|x| x as f32).collect();
        let v2 = v1.clone();
        b.iter(|| inplace_sum(&mut v1, &v2));
    }

    #[bench]
    fn bench_inp_sum_standard(b: &mut Bencher) {
        let mut v1: Vec<f32> = (0..200).map(|x| x as f32).collect();
        let v2 = v1.clone();
        b.iter(|| {
            for i in 0..v1.len() {
                v1[i] += v2[i];
            }
        });
    }

    #[bench]
    fn bench_dot_avx(b: &mut Bencher) {
        let v1: Vec<f32> = (0..200).map(|x| x as f32).collect();
        b.iter(|| dot(&v1, &v1));
    }

    #[bench]
    fn bench_dot_sse(b: &mut Bencher) {
        let v1: Vec<f32> = (0..200).map(|x| x as f32).collect();
        b.iter(|| dot_sse(&v1, &v1));
    }

    #[bench]
    fn bench_dot_standard(b: &mut Bencher) {
        let v1: Vec<f32> = (0..200).map(|x| x as f32).collect();
        b.iter(|| {
            let mut s = 0f32;
            for i in 0..v1.len() {
                s += v1[i] * v1[i];
            }
            s
        });
    }

    #[bench]
    fn bench_scale_avx(b: &mut Bencher) {
        let mut v1: Vec<f32> = (0..200).map(|x| x as f32).collect();
        b.iter(|| scale(&mut v1, 2.0));
    }

    #[bench]
    fn bench_scale_standard(b: &mut Bencher) {
        let mut v1: Vec<f32> = (0..200).map(|x| x as f32).collect();
        b.iter(|| {
            for i in 0..v1.len() {
                v1[i] *= 2.0;
            }
        });
    }

    #[bench]
    fn bench_sparse_dot(b: &mut Bencher) {
        let idx = [1usize, 117, 141, 132, 12, 14, 187, 31];
        let v1 = [1.23, 571., -2.2, 7.1, 1.0, 0.01, 13., 12.];
        let v2: Vec<f32> = (0..200).map(|x| x as f32).collect();
        b.iter(|| {
            sparse_dot(&idx, &v1, &v2);
        });
    }

    #[bench]
    fn bench_sparse_dot_standard(b: &mut Bencher) {
        let idx: [usize; 8] = [1usize, 117, 141, 132, 12, 14, 187, 31];
        let v1 = [1.23, 571., -2.2, 7.1, 1.0, 0.01, 13., 12.];
        let v2: Vec<f32> = (0..200).map(|x| x as f32).collect();
        b.iter(|| {
            let mut s = 0.;
            for i in 0..idx.len() {
                s += v1[i] * v2[idx[i]];
            }
            s
        });
    }
}

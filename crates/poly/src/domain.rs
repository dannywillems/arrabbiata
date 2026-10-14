//! Evaluation domains: multiplicative subgroups of order `2^k`, and the
//! number-theoretic transform (NTT) between coefficients and evaluations.
//!
//! The domain `H = {1, w, ..., w^(n-1)}` of a primitive `n`-th root of unity
//! `w` is what PlonK-style protocols interpolate traces on. Evaluation on `H`
//! is the discrete Fourier transform over `F`, computed here by the
//! iterative radix-2 Cooley-Tukey algorithm in `O(n log n)`.
//!
//! ```
//! use arrabbiata_field::{Field, Goldilocks as F};
//! use arrabbiata_poly::{Domain, UniPoly};
//!
//! let domain = Domain::<F>::new(2).unwrap(); // n = 4
//! let p = UniPoly::new(vec![F::new(1), F::new(2), F::new(3)]);
//! let evaluations = domain.evaluate(&p).unwrap();
//! assert_eq!(evaluations[1], p.evaluate(domain.element(1)));
//! assert_eq!(domain.interpolate(&evaluations).unwrap().coefficient(2), F::new(3));
//! ```

use alloc::{vec, vec::Vec};

use arrabbiata_field::TwoAdicField;

use crate::ring::UniPoly;

/// The subgroup of order `2^log_size` of the multiplicative group of `F`.
///
/// ```
/// use arrabbiata_field::{Field, Goldilocks as F};
/// use arrabbiata_poly::Domain;
/// let h = Domain::<F>::new(3).unwrap();
/// assert_eq!(h.size(), 8);
/// assert_eq!(h.element(8), F::ONE);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Domain<F> {
    log_size: u32,
    size: usize,
    generator: F,
    generator_inverse: F,
    size_inverse: F,
}

impl<F> Domain<F>
where
    F: TwoAdicField,
{
    /// The domain of size `2^log_size`, or `None` if the field has no such
    /// subgroup.
    ///
    /// ```
    /// use arrabbiata_field::Goldilocks as F;
    /// use arrabbiata_poly::Domain;
    /// assert!(Domain::<F>::new(10).is_some());
    /// assert!(Domain::<F>::new(40).is_none());
    /// ```
    #[must_use]
    pub fn new(log_size: u32) -> Option<Self> {
        // usize is at least 32 bits on every target we support, and the
        // two-adicity of the fields we use is at most 32.
        if log_size >= usize::BITS {
            return None;
        }
        let generator = F::two_adic_generator(log_size)?;
        let size = 1usize << log_size;
        Some(Self {
            log_size,
            size,
            generator,
            generator_inverse: generator.inverse()?,
            size_inverse: F::from_u64(size as u64).inverse()?,
        })
    }

    /// The smallest domain with at least `n` elements.
    ///
    /// ```
    /// use arrabbiata_field::Goldilocks as F;
    /// use arrabbiata_poly::Domain;
    /// assert_eq!(Domain::<F>::containing(5).unwrap().size(), 8);
    /// assert_eq!(Domain::<F>::containing(1).unwrap().size(), 1);
    /// ```
    #[must_use]
    pub fn containing(n: usize) -> Option<Self> {
        let mut log_size = 0;
        while (1usize << log_size) < n {
            log_size += 1;
        }
        Self::new(log_size)
    }

    /// The number of elements `n`.
    ///
    /// ```
    /// use arrabbiata_field::Goldilocks as F;
    /// use arrabbiata_poly::Domain;
    /// assert_eq!(Domain::<F>::new(4).unwrap().size(), 16);
    /// ```
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// `log2(n)`.
    ///
    /// ```
    /// use arrabbiata_field::Goldilocks as F;
    /// use arrabbiata_poly::Domain;
    /// assert_eq!(Domain::<F>::new(4).unwrap().log_size(), 4);
    /// ```
    #[must_use]
    pub const fn log_size(&self) -> u32 {
        self.log_size
    }

    /// The generator `w`, a primitive `n`-th root of unity.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_poly::Domain;
    /// let h = Domain::<F>::new(2).unwrap();
    /// assert_eq!(h.generator().pow(2), -F::ONE);
    /// ```
    #[must_use]
    pub const fn generator(&self) -> F {
        self.generator
    }

    /// `w^i`.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_poly::Domain;
    /// let h = Domain::<F>::new(2).unwrap();
    /// assert_eq!(h.element(0), F::ONE);
    /// assert_eq!(h.element(5), h.element(1));
    /// ```
    #[must_use]
    pub fn element(&self, i: usize) -> F {
        self.generator.pow((i % self.size) as u64)
    }

    /// The vanishing polynomial of the domain at `x`: `x^n - 1`.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_poly::Domain;
    /// let h = Domain::<F>::new(3).unwrap();
    /// assert_eq!(h.vanishing_at(h.element(3)), F::ZERO);
    /// assert_ne!(h.vanishing_at(F::new(2)), F::ZERO);
    /// ```
    #[must_use]
    pub fn vanishing_at(&self, x: F) -> F {
        x.pow(self.size as u64) - F::ONE
    }

    /// The first Lagrange polynomial `L_1` of the domain at `x`: the
    /// polynomial of degree `n - 1` equal to 1 at `1` and to 0 on the rest of
    /// the domain, `L_1(x) = (x^n - 1) / (n (x - 1))`.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_poly::Domain;
    /// let h = Domain::<F>::new(2).unwrap();
    /// assert_eq!(h.first_lagrange_at(F::ONE), F::ONE);
    /// assert_eq!(h.first_lagrange_at(h.element(1)), F::ZERO);
    /// ```
    #[must_use]
    pub fn first_lagrange_at(&self, x: F) -> F {
        match (x - F::ONE).inverse() {
            None => F::ONE,
            Some(inverse) => self.vanishing_at(x) * self.size_inverse * inverse,
        }
    }

    /// The `j`-th Lagrange polynomial of the domain at `x`: the polynomial
    /// of degree `n - 1` equal to 1 at `w^j` and to 0 on the rest of the
    /// domain, `L_j(x) = w^j (x^n - 1) / (n (x - w^j))`.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_poly::Domain;
    /// let h = Domain::<F>::new(2).unwrap();
    /// assert_eq!(h.lagrange_at(2, h.element(2)), F::ONE);
    /// assert_eq!(h.lagrange_at(2, h.element(3)), F::ZERO);
    /// // The Lagrange polynomials sum to one everywhere.
    /// let x = F::new(12345);
    /// let sum = (0..4).fold(F::ZERO, |acc, j| acc + h.lagrange_at(j, x));
    /// assert_eq!(sum, F::ONE);
    /// ```
    #[must_use]
    pub fn lagrange_at(&self, j: usize, x: F) -> F {
        let wj = self.element(j);
        match (x - wj).inverse() {
            None => F::ONE,
            Some(inverse) => {
                if self.vanishing_at(x).is_zero() {
                    F::ZERO
                } else {
                    wj * self.vanishing_at(x) * self.size_inverse * inverse
                }
            }
        }
    }

    /// The evaluations `[p(1), p(w), ..., p(w^(n-1))]`, or `None` if `p` has
    /// more than `n` coefficients.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_poly::{Domain, UniPoly};
    /// let h = Domain::<F>::new(1).unwrap();
    /// // p = 3 + 2x on {1, -1}.
    /// let e = h.evaluate(&UniPoly::new(vec![F::new(3), F::new(2)])).unwrap();
    /// assert_eq!(e, vec![F::new(5), F::new(1)]);
    /// ```
    #[must_use]
    pub fn evaluate(&self, p: &UniPoly<F>) -> Option<Vec<F>> {
        if p.len() > self.size {
            return None;
        }
        let mut values = vec![F::ZERO; self.size];
        let mut i = 0;
        while i < p.len() {
            values[i] = p.coefficient(i);
            i += 1;
        }
        self.transform(&mut values, self.generator);
        Some(values)
    }

    /// The unique polynomial of degree below `n` with the given evaluations
    /// on the domain, or `None` if there are not exactly `n` of them.
    ///
    /// ```
    /// use arrabbiata_field::{Field, Goldilocks as F};
    /// use arrabbiata_poly::Domain;
    /// let h = Domain::<F>::new(1).unwrap();
    /// let p = h.interpolate(&[F::new(5), F::new(1)]).unwrap();
    /// assert_eq!((p.coefficient(0), p.coefficient(1)), (F::new(3), F::new(2)));
    /// ```
    #[must_use]
    pub fn interpolate(&self, evaluations: &[F]) -> Option<UniPoly<F>> {
        if evaluations.len() != self.size {
            return None;
        }
        let mut values = evaluations.to_vec();
        self.transform(&mut values, self.generator_inverse);
        let mut i = 0;
        while i < values.len() {
            values[i] = values[i] * self.size_inverse;
            i += 1;
        }
        Some(UniPoly::new(values))
    }

    /// In-place radix-2 decimation-in-time NTT with root `root`, of order
    /// `n`: `values[i] <- sum_j values[j] root^(ij)`.
    fn transform(&self, values: &mut [F], root: F) {
        let n = self.size;
        // Bit-reversal permutation.
        let mut i = 0;
        while i < n {
            let j = reverse_bits(i, self.log_size);
            if i < j {
                values.swap(i, j);
            }
            i += 1;
        }
        // Butterflies: at each stage, blocks of length `len` combine two
        // halves with the twiddles of a primitive len-th root of unity.
        let mut len = 2;
        while len <= n {
            let step = root.pow((n / len) as u64);
            let mut start = 0;
            while start < n {
                let mut twiddle = F::ONE;
                let mut k = 0;
                while k < len / 2 {
                    let even = values[start + k];
                    let odd = values[start + k + len / 2] * twiddle;
                    values[start + k] = even + odd;
                    values[start + k + len / 2] = even - odd;
                    twiddle = twiddle * step;
                    k += 1;
                }
                start += len;
            }
            len *= 2;
        }
    }
}

const fn reverse_bits(mut x: usize, bits: u32) -> usize {
    let mut out = 0;
    let mut b = 0;
    while b < bits {
        out = (out << 1) | (x & 1);
        x >>= 1;
        b += 1;
    }
    out
}

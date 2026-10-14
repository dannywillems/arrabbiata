# elliptic curves

Groups of points of elliptic curves. `curve` holds the abstractions (the
`Curve` and `Pairing` traits, complete addition formulas, multi-scalar
multiplication) and the group laws every curve is property-tested against;
`pasta` and `bn254` are instances.

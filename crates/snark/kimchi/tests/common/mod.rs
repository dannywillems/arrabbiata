//! The four circuits of the Kimchi figures tools (`constraint_system`,
//! `verifier_index`), built as the tools build them.

use arrabbiata_field::Field;
use arrabbiata_kimchi::constraints::{
    CircuitGate, Connect, ConstraintSystem, GateType, LookupTable, RuntimeTableCfg, Wire,
};
use arrabbiata_pasta::Fp;

fn f(k: u64) -> Fp {
    Fp::from_u64(k)
}

fn gate(typ: GateType, row: usize, coeffs: usize, seed: u64) -> CircuitGate<Fp> {
    let values = (0..coeffs).map(|i| f(seed + i as u64)).collect();
    CircuitGate::new(typ, Wire::for_row(row), values)
}

/// The tool's circuits, by name.
pub fn circuits() -> Vec<(&'static str, ConstraintSystem<Fp>)> {
    let mut generic: Vec<CircuitGate<Fp>> = (0..10)
        .map(|row| gate(GateType::Generic, row, 10, 100 * row as u64))
        .collect();
    generic.connect_cell_pair((0, 0), (3, 1));
    generic.connect_cell_pair((3, 1), (7, 2));
    generic.connect_cell_pair((1, 0), (9, 6));
    let generic = ConstraintSystem::create(generic).public(2).build().unwrap();

    let kinds = [
        (GateType::Generic, 10),
        (GateType::Poseidon, 15),
        (GateType::CompleteAdd, 0),
        (GateType::VarBaseMul, 0),
        (GateType::Zero, 0),
        (GateType::EndoMul, 0),
        (GateType::EndoMulScalar, 0),
        (GateType::RangeCheck0, 1),
        (GateType::RangeCheck1, 0),
        (GateType::Zero, 0),
        (GateType::ForeignFieldAdd, 4),
        (GateType::ForeignFieldMul, 4),
        (GateType::Zero, 0),
        (GateType::Xor16, 0),
        (GateType::Rot64, 1),
        (GateType::Zero, 0),
    ];
    let mut every: Vec<CircuitGate<Fp>> = kinds
        .iter()
        .enumerate()
        .map(|(row, (typ, n))| gate(*typ, row, *n, 7 + 31 * row as u64))
        .collect();
    every.connect_cell_pair((0, 2), (14, 5));
    every.connect_cell_pair((8, 0), (11, 3));
    let every = ConstraintSystem::create(every).build().unwrap();

    let chunked: Vec<CircuitGate<Fp>> = (0..100)
        .map(|row| gate(GateType::Generic, row, 10, row as u64))
        .collect();
    let chunked = ConstraintSystem::create(chunked)
        .public(1)
        .max_poly_size(Some(32))
        .build()
        .unwrap();

    let mut lookup: Vec<CircuitGate<Fp>> = (0..4)
        .map(|row| gate(GateType::Generic, row, 10, 3 * row as u64))
        .collect();
    lookup.extend((4..8).map(|row| gate(GateType::Lookup, row, 0, 0)));
    let table = LookupTable {
        id: 1,
        data: (0..3)
            .map(|c| (0..5).map(|r| f(10 * c + r)).collect())
            .collect(),
    };
    let runtime = RuntimeTableCfg {
        id: 2,
        first_column: vec![f(1), f(2), f(3)],
    };
    let lookup = ConstraintSystem::create(lookup)
        .lookup(vec![table])
        .runtime(Some(vec![runtime]))
        .build()
        .unwrap();

    vec![
        ("generic", generic),
        ("every", every),
        ("chunked", chunked),
        ("lookup", lookup),
    ]
}

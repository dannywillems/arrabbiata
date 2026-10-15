//! The behaviour of each public item, one test per item (these were
//! the rustdoc examples).

#[test]
fn crate_overview() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;

    let (circuit, witness, ()) = build::<F, _, _>(|b| {
        let x = b.witness(F::new(3));
        let y = b.mul(&x, &x);
        let out = b.public_input(F::new(9));
        b.assert_equal(&y, &out);
    });
    assert!(circuit.is_satisfied(&witness));
}

#[test]
fn vanilla_overview() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;

    // Prove knowledge of x such that x^3 + x + 5 = 35.
    let (circuit, witness, ()) = build::<F, _, _>(|b| {
        let x = b.witness(F::new(3));
        let x2 = b.mul(&x, &x);
        let x3 = b.mul(&x2, &x);
        let lhs = b.add(&x3, &x);
        let lhs = b.add_constant(&lhs, F::new(5));
        let rhs = b.public_input(F::new(35));
        b.assert_equal(&lhs, &rhs);
    });
    assert!(circuit.is_satisfied(&witness));
    assert_eq!(witness.public_inputs(), &[F::new(35)]);
}

#[test]
fn vanilla_column() {
    use arrabbiata_frontend::vanilla::Column;
    assert_eq!(Column::B.index(), 1);
}

#[test]
fn vanilla_column_all() {
    use arrabbiata_frontend::vanilla::Column;
    assert_eq!(Column::ALL.len(), 3);
}

#[test]
fn vanilla_column_index() {
    use arrabbiata_frontend::vanilla::Column;
    assert_eq!(Column::C.index(), 2);
}

#[test]
fn vanilla_cell() {
    use arrabbiata_frontend::vanilla::{Cell, Column};
    let cell = Cell {
        column: Column::A,
        row: 3,
    };
    assert_eq!(cell.index(8), 3);
}

#[test]
fn vanilla_cell_index() {
    use arrabbiata_frontend::vanilla::{Cell, Column};
    assert_eq!(
        Cell {
            column: Column::C,
            row: 1
        }
        .index(4),
        9
    );
}

#[test]
fn vanilla_wire() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;
    build::<F, _, _>(|b| {
        let x = b.witness(F::new(4));
        assert_eq!(x.value(), F::new(4));
    });
}

#[test]
fn vanilla_wire_value() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;
    build::<F, _, _>(|b| {
        let x = b.witness(F::new(2));
        assert_eq!(b.mul(&x, &x).value(), F::new(4));
    });
}

#[test]
fn vanilla_wire_cell() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::{Column, build};
    build::<F, _, _>(|b| {
        let x = b.witness(F::new(2));
        assert_eq!(x.cell().column, Column::A);
    });
}

#[test]
fn vanilla_gate() {
    use arrabbiata_field::{Field, Goldilocks as F};
    use arrabbiata_frontend::vanilla::Gate;
    // c = a + b.
    let add = Gate {
        q_l: F::ONE,
        q_r: F::ONE,
        q_o: -F::ONE,
        ..Gate::default()
    };
    assert_eq!(add.evaluate(F::new(2), F::new(3), F::new(5)), F::ZERO);
}

#[test]
fn vanilla_gate_empty() {
    use arrabbiata_field::{Field, Goldilocks as F};
    use arrabbiata_frontend::vanilla::Gate;
    assert_eq!(
        Gate::<F>::empty().evaluate(F::new(1), F::new(2), F::new(3)),
        F::ZERO
    );
}

#[test]
fn vanilla_gate_evaluate() {
    use arrabbiata_field::{Field, Goldilocks as F};
    use arrabbiata_frontend::vanilla::Gate;
    let mul = Gate {
        q_m: F::ONE,
        q_o: -F::ONE,
        ..Gate::empty()
    };
    assert_eq!(mul.evaluate(F::new(3), F::new(4), F::new(12)), F::ZERO);
}

#[test]
fn vanilla_builder() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;
    let (circuit, witness, ()) = build::<F, _, _>(|b| {
        let x = b.witness(F::new(2));
        let _ = b.add(&x, &x);
    });
    assert!(circuit.is_satisfied(&witness));
}

#[test]
fn vanilla_build() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;
    let (circuit, _, square) = build::<F, _, _>(|b| {
        let x = b.witness(F::new(9));
        b.mul(&x, &x).value()
    });
    assert_eq!(square, F::new(81));
    assert!(circuit.num_rows().is_power_of_two());
}

#[test]
fn vanilla_builder_witness() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;
    build::<F, _, _>(|b| assert_eq!(b.witness(F::new(1)).value(), F::new(1)));
}

#[test]
fn vanilla_builder_public_input() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;
    let (_, w, ()) = build::<F, _, _>(|b| {
        b.public_input(F::new(7));
    });
    assert_eq!(w.public_inputs(), &[F::new(7)]);
}

#[test]
fn vanilla_builder_constant() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;
    build::<F, _, _>(|b| assert_eq!(b.constant(F::new(5)).value(), F::new(5)));
}

#[test]
fn vanilla_builder_add() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;
    build::<F, _, _>(|b| {
        let (x, y) = (b.witness(F::new(2)), b.witness(F::new(3)));
        assert_eq!(b.add(&x, &y).value(), F::new(5));
    });
}

#[test]
fn vanilla_builder_sub() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;
    build::<F, _, _>(|b| {
        let (x, y) = (b.witness(F::new(5)), b.witness(F::new(3)));
        assert_eq!(b.sub(&x, &y).value(), F::new(2));
    });
}

#[test]
fn vanilla_builder_mul() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;
    build::<F, _, _>(|b| {
        let (x, y) = (b.witness(F::new(5)), b.witness(F::new(3)));
        assert_eq!(b.mul(&x, &y).value(), F::new(15));
    });
}

#[test]
fn vanilla_builder_add_constant() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;
    build::<F, _, _>(|b| {
        let x = b.witness(F::new(5));
        assert_eq!(b.add_constant(&x, F::new(1)).value(), F::new(6));
    });
}

#[test]
fn vanilla_builder_scale() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;
    build::<F, _, _>(|b| {
        let x = b.witness(F::new(5));
        assert_eq!(b.scale(&x, F::new(3)).value(), F::new(15));
    });
}

#[test]
fn vanilla_builder_assert_equal() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;
    let (circuit, witness, ()) = build::<F, _, _>(|b| {
        let (x, y) = (b.witness(F::new(1)), b.witness(F::new(2)));
        b.assert_equal(&x, &y);
    });
    assert!(!circuit.is_satisfied(&witness));
}

#[test]
fn vanilla_builder_assert_zero() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;
    let (circuit, witness, ()) = build::<F, _, _>(|b| {
        let x = b.witness(F::new(3));
        let y = b.witness(F::new(3));
        let d = b.sub(&x, &y);
        b.assert_zero(&d);
    });
    assert!(circuit.is_satisfied(&witness));
}

#[test]
fn vanilla_circuit() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;
    let (circuit, _, ()) = build::<F, _, _>(|b| {
        b.public_input(F::new(1));
    });
    assert_eq!(circuit.public_rows(), &[0]);
}

#[test]
fn vanilla_witness() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;
    let (_, witness, ()) = build::<F, _, _>(|b| {
        b.witness(F::new(8));
    });
    assert_eq!(witness.wires()[0][0], F::new(8));
}

#[test]
fn vanilla_witness_new() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::{Witness, build};
    let (circuit, honest, ()) = build::<F, _, _>(|b| {
        let x = b.witness(F::new(2));
        b.mul(&x, &x);
    });
    let mut wires = honest.wires().clone();
    wires[2][1] = F::new(5); // claim 2 * 2 = 5
    assert!(!circuit.is_satisfied(&Witness::new(wires, vec![])));
}

#[test]
fn vanilla_witness_wires() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;
    let (c, w, ()) = build::<F, _, _>(|_| ());
    assert_eq!(w.wires()[2].len(), c.num_rows());
}

#[test]
fn vanilla_witness_public_inputs() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;
    let (_, w, ()) = build::<F, _, _>(|_| ());
    assert!(w.public_inputs().is_empty());
}

#[test]
fn vanilla_circuit_num_rows() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;
    let (c, _, ()) = build::<F, _, _>(|b| {
        for i in 0..5 {
            b.witness(F::new(i));
        }
    });
    assert_eq!(c.num_rows(), 8);
}

#[test]
fn vanilla_circuit_gates() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;
    let (c, _, ()) = build::<F, _, _>(|b| {
        b.constant(F::new(3));
    });
    assert_eq!(c.gates()[0].q_c, -F::new(3));
}

#[test]
fn vanilla_circuit_sigma() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::{Cell, Column, build};
    let (c, _, ()) = build::<F, _, _>(|b| {
        let x = b.witness(F::new(1));
        let y = b.witness(F::new(1));
        b.assert_equal(&x, &y);
    });
    let n = c.num_rows();
    let (x, y) = (
        Cell {
            column: Column::A,
            row: 0,
        },
        Cell {
            column: Column::A,
            row: 1,
        },
    );
    assert_eq!(c.sigma()[x.index(n)], y.index(n));
    assert_eq!(c.sigma()[y.index(n)], x.index(n));
}

#[test]
fn vanilla_circuit_public_rows() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;
    let (c, _, ()) = build::<F, _, _>(|b| {
        b.witness(F::new(1));
        b.public_input(F::new(2));
    });
    assert_eq!(c.public_rows(), &[1]);
}

#[test]
fn vanilla_circuit_public_input_column() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;
    let (c, w, ()) = build::<F, _, _>(|b| {
        b.public_input(F::new(2));
    });
    assert_eq!(c.public_input_column(w.public_inputs())[0], -F::new(2));
}

#[test]
fn vanilla_circuit_is_satisfied() {
    use arrabbiata_field::Goldilocks as F;
    use arrabbiata_frontend::vanilla::build;
    let (c, w, ()) = build::<F, _, _>(|b| {
        let x = b.witness(F::new(6));
        let y = b.mul(&x, &x);
        let z = b.public_input(F::new(36));
        b.assert_equal(&y, &z);
    });
    assert!(c.is_satisfied(&w));
}

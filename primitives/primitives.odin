package primitives

import "core:testing"

/*
	GenerationalIndex is a small type containing an index and a generation.
	Used to insert elements that are referenceable, but enable a check to ensure one does
	not wrongfully target something too new.
*/
GenerationalIndex :: struct {
	index:      u64,
	generation: u64,
}

GenerationalIndexEq :: #force_inline proc(lhs: GenerationalIndex, rhs: GenerationalIndex) -> bool {
	return (lhs.index == rhs.index) && (lhs.generation == rhs.generation)
}

@(test)
test_generational_index_eq :: proc(t: ^testing.T) {
	a := GenerationalIndex {
		index      = 0,
		generation = 0,
	}
	b := GenerationalIndex {
		index      = 0,
		generation = 0,
	}

	eq := GenerationalIndexEq(a, b)

	if !testing.expect_value(t, eq, true) {return}
}

@(test)
test_generational_index_differing_index_neq :: proc(t: ^testing.T) {
	a := GenerationalIndex {
		index      = 0,
		generation = 0,
	}
	b := GenerationalIndex {
		index      = 1,
		generation = 0,
	}

	eq := GenerationalIndexEq(a, b)

	if !testing.expect_value(t, eq, false) {return}
}

@(test)
test_generational_index_differing_generation_neq :: proc(t: ^testing.T) {
	a := GenerationalIndex {
		index      = 0,
		generation = 0,
	}
	b := GenerationalIndex {
		index      = 0,
		generation = 1,
	}

	eq := GenerationalIndexEq(a, b)

	if !testing.expect_value(t, eq, false) {return}
}

@(test)
test_generational_index_differing_both_neq :: proc(t: ^testing.T) {
	a := GenerationalIndex {
		index      = 0,
		generation = 0,
	}
	b := GenerationalIndex {
		index      = 1,
		generation = 1,
	}

	eq := GenerationalIndexEq(a, b)

	if !testing.expect_value(t, eq, false) {return}
}

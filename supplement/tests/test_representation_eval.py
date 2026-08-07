from tools.representation_eval import constant_path_counts, encode_sketch


def test_constant_path_sketch_exposes_coercion_constructor():
    abbreviated = "(@Eq.{1} (WithTop.{0} NNReal) (ENNReal.ofNNReal x) y)"
    unfolded = "(@Eq.{1} (WithTop.{0} NNReal) (@WithTop.some.{0} NNReal x) y)"

    sketch_a = constant_path_counts(abbreviated)
    sketch_b = constant_path_counts(unfolded)

    assert sketch_a["Eq"] == 1
    assert sketch_a["ENNReal.ofNNReal"] == 1
    assert sketch_b["WithTop.some"] == 1
    assert sketch_a != sketch_b


def test_constant_path_sketch_counts_and_sorts():
    pp = "(@Foo.bar.{0} x) (@Foo.bar.{1} y) (Baz.qux z)"
    counts = constant_path_counts(pp)
    assert counts == {"Baz.qux": 1, "Foo.bar": 2}
    assert encode_sketch(counts) == "Baz.qux=1 Foo.bar=2"

import torch

from model_core.vm import StackVM
from model_core.vocab import FORMULA_VOCAB


def test_output_normalization_is_prefix_invariant() -> None:
    """Appending future observations must not alter any historical output."""
    prefix = torch.ones((1, 5), dtype=torch.float32)
    full = torch.cat((prefix, torch.tensor([[2.0, 3.0, 5.0]])), dim=1)

    prefix_result = StackVM._normalize_output(prefix)
    full_result = StackVM._normalize_output(full)[:, : prefix.shape[1]]

    torch.testing.assert_close(full_result, prefix_result, rtol=0.0, atol=0.0)


def test_execute_is_prefix_invariant_for_raw_feature_formula() -> None:
    vm = StackVM()
    prefix = torch.ones((1, 1, 5), dtype=torch.float32)
    full = torch.cat(
        (prefix, torch.tensor([[[2.0, 3.0, 5.0]]], dtype=torch.float32)), dim=2
    )

    prefix_result = vm.execute([0], prefix)
    full_result = vm.execute([0], full)

    assert prefix_result is not None and full_result is not None
    torch.testing.assert_close(
        full_result[:, : prefix.shape[2]], prefix_result, rtol=0.0, atol=0.0
    )


def test_execute_prefix_invariance_across_operator_formulas() -> None:
    names = FORMULA_VOCAB.token_names
    formulas = [
        [names.index("RET"), names.index("TS_MEAN_5")],
        [names.index("RET"), names.index("RET5"), names.index("ADD")],
        [names.index("RET"), names.index("TS_ZSCORE_20")],
    ]
    generator = torch.Generator().manual_seed(42)
    prefix = torch.randn((3, FORMULA_VOCAB.operator_offset, 40), generator=generator)
    future = torch.randn((3, FORMULA_VOCAB.operator_offset, 20), generator=generator)
    full = torch.cat((prefix, future), dim=2)
    vm = StackVM()

    for formula in formulas:
        prefix_result = vm.execute(formula, prefix)
        full_result = vm.execute(formula, full)
        assert prefix_result is not None and full_result is not None
        torch.testing.assert_close(
            full_result[:, : prefix.shape[2]], prefix_result, rtol=0.0, atol=0.0
        )


def test_invalid_mask_survives_numeric_cleaning() -> None:
    features = torch.ones((1, FORMULA_VOCAB.operator_offset, 20))
    features[:, 0, :5] = float("nan")
    result = StackVM().execute_with_diagnostics([0], features)

    assert result is not None
    assert torch.isfinite(result.values).all()
    assert result.invalid_mask[:, :5].all()
    assert not result.invalid_mask[:, 5:].any()

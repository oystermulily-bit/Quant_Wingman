import pytest

from model_core.engine import _build_walk_forward_folds


@pytest.mark.parametrize("total", [500, 7601])
def test_requested_purge_gap_is_never_silently_reduced(total: int) -> None:
    folds = _build_walk_forward_folds(
        T=total, n_folds=5, gap=20, label_horizon=2
    )
    assert len(folds) == 4
    for fold in folds:
        assert fold["gap"] == 20
        assert fold["val_start"] - fold["train_end"] >= 20
        assert fold["val_end"] <= total - 2


def test_walk_forward_is_expanding_and_ordered() -> None:
    folds = _build_walk_forward_folds(T=500, n_folds=5, gap=20, label_horizon=2)
    previous_train_end = -1
    previous_val_end = -1
    for fold in folds:
        assert fold["train_start"] == 0
        assert fold["train_end"] > previous_train_end
        assert fold["val_start"] >= previous_val_end
        previous_train_end = fold["train_end"]
        previous_val_end = fold["val_end"]


def test_gap_zero_remains_supported() -> None:
    folds = _build_walk_forward_folds(T=500, n_folds=5, gap=0, label_horizon=2)
    assert len(folds) == 4
    assert all(fold["val_start"] == fold["train_end"] for fold in folds)


def test_insufficient_data_fails_closed_instead_of_leaking() -> None:
    with pytest.raises(ValueError, match="insufficient data"):
        _build_walk_forward_folds(T=50, n_folds=5, gap=20, label_horizon=2)


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(ValueError):
        _build_walk_forward_folds(T=500, n_folds=1, gap=20)
    with pytest.raises(ValueError):
        _build_walk_forward_folds(T=500, n_folds=5, gap=-1)

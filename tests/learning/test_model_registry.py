"""Exercise semantic registry lookup and model construction boundaries.

Responsibilities:
  - Check semantic lookup, constructors, axis/depth limits, and cross-resolution admission

Design principles:
  - Supplied devices and explicit architecture capabilities govern construction

This module does NOT:
  - Train models or select scientific architecture defaults
"""

import sys
import types
from collections.abc import Callable
from typing import Any

import pytest
import torch
from support import configs

from src import domain, experiments, learning

# Imported before any test fakes neuralop; build_rno imports the cell module lazily.
from src.learning.models import learning_models_rno_cell as rno_cell
from src.learning.transient import learning_transient_rollout as rollout


def test_representative_semantic_identifiers_resolve() -> None:
    """Resolve one supported identifier through each public semantic registry."""
    assert learning.models.factory.resolve_model_kind("fno").kind == "fno"
    assert learning.losses.factory.resolve_data_loss_kind("relative_h1").kind == "relative_h1"
    assert learning.metrics.metrics.resolve_metric_kind("rmse").direction == "minimize"
    assert domain.tasks.registry.resolve_physics("steady_2d_brinkman").kind == "steady_2d_brinkman"


@pytest.mark.parametrize(
    ("resolver", "message"),
    [
        (learning.models.factory.resolve_model_kind, "Unknown model identifier"),
        (learning.losses.factory.resolve_data_loss_kind, "Unknown loss identifier"),
        (learning.metrics.metrics.resolve_metric_kind, "Unknown metric identifier"),
        (domain.tasks.registry.resolve_physics, "Unknown physics identifier"),
    ],
)
def test_unknown_semantic_identifier_fails_at_registry_boundary(
    resolver: Callable[[str], object],
    message: str,
) -> None:
    """Reject a neutral unknown value without depending on registry inventory."""
    with pytest.raises(ValueError, match=message):
        resolver("artificial-unsupported-id")


def test_model_validation_rejects_unsupported_axes_and_uno_depth() -> None:
    """Keep structural limits consistent between validation and construction."""
    fno_params = {
        "in_channels": 7,
        "out_channels": 3,
        "n_modes": [8, 8, 8],
        "hidden_channels": 4,
        "n_layers": 2,
    }
    with pytest.raises(ValueError, match="exactly two operator axes"):
        learning.models.factory.validate_model_params(
            "fno",
            fno_params,
            require_channels=True,
            operator_dimensionality=3,
        )

    uno_params = {
        "hidden_channels": 4,
        "modes_x": 8,
        "modes_y": 8,
        "n_layers": 3,
        "positional_embedding": None,
        "domain_padding": 0.1,
    }
    with pytest.raises(ValueError, match="supports exactly 5 or 7 layers"):
        learning.models.factory.validate_model_params(
            "uno",
            uno_params,
            require_channels=False,
            operator_dimensionality=2,
        )
    with pytest.raises(ValueError, match="supports exactly 5 or 7 layers"):
        learning.models.factory.build_uno(
            in_channels=7,
            out_channels=3,
            n_layers=3,
            hidden_channels=4,
            modes_x=8,
            modes_y=8,
            positional_embedding=None,
            domain_padding=0.1,
            uno_scalings=[[1.0, 1.0]] * 3,
        )


def test_model_factory_uses_only_supplied_concrete_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build on explicit CPU without querying CUDA or resolving a policy token."""
    config = configs.resolve(configs.direct_config())
    config["model"]["params"].update(
        {"n_modes": [2, 2], "hidden_channels": 2, "n_layers": 1},
    )

    def unexpected_cuda_query(*_args: Any, **_kwargs: Any) -> Any:
        message = "model factory queried CUDA availability"
        raise AssertionError(message)

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_cuda_query)
    model = learning.models.factory.build_model(config, device=torch.device("cpu"))
    assert {parameter.device for parameter in model.parameters()} == {torch.device("cpu")}
    with pytest.raises(TypeError, match=r"concrete CPU or CUDA torch\.device"):
        learning.models.factory.build_model(config, device="cpu")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="indexed CUDA device"):
        learning.models.factory.build_model(config, device=torch.device("cuda"))


def test_rno_requires_official_constructor_and_single_step_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass explicit one-step RNO semantics to the official constructor boundary."""
    captured: dict[str, Any] = {}
    converted: dict[str, Any] = {}
    monkeypatch.setattr(rno_cell, "use_tanh_candidate_cells", lambda model, *, n_layers: converted.update(model=model, n_layers=n_layers))

    class FakeRNO(torch.nn.Module):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()
            captured.update(kwargs)

    models = types.ModuleType("neuralop.models")
    models.__dict__["RNO"] = FakeRNO
    package = types.ModuleType("neuralop")
    package.__dict__["models"] = models
    monkeypatch.setitem(sys.modules, "neuralop", package)
    monkeypatch.setitem(sys.modules, "neuralop.models", models)
    model = learning.models.factory.build_rno(
        in_channels=3,
        out_channels=4,
        n_modes=[2, 2],
        hidden_channels=5,
        candidate_activation="tanh",
        positional_embedding=None,
        domain_padding=0.1,
        return_sequences=False,
    )
    assert isinstance(model, FakeRNO)
    assert converted == {"model": model, "n_layers": 4}
    assert captured["return_sequences"] is False
    assert captured["rno_skip"] is False
    assert captured["positional_embedding"] is None
    assert captured["domain_padding"] == 0.1
    common = {"in_channels": 3, "out_channels": 4, "n_modes": [2, 2], "hidden_channels": 5, "candidate_activation": "tanh"}
    common |= {"positional_embedding": None, "domain_padding": 0.1}
    with pytest.raises(ValueError, match="return_sequences=False"):
        learning.models.factory.build_rno(**common, return_sequences=True)
    with pytest.raises(ValueError, match="rno_skip=False"):
        learning.models.factory.build_rno(**common, rno_skip=True)
    with pytest.raises(ValueError, match="rno_skip must be false"):
        learning.models.factory.validate_model_params(
            "rno",
            {"n_modes": [2, 2], "hidden_channels": 5, "candidate_activation": "tanh", "rno_skip": True}
            | {"positional_embedding": None, "domain_padding": 0.1},
            require_channels=False,
            operator_dimensionality=2,
        )
    with pytest.raises(ValueError, match="Unknown model identifier"):
        learning.models.factory.resolve_model_kind("uno-rno")


_KIND_PARAMS: dict[str, dict[str, Any]] = {
    "fno": {"n_modes": [2, 2], "hidden_channels": 4, "n_layers": 2},
    "uno": {"modes_x": 8, "modes_y": 8, "hidden_channels": 4, "n_layers": 5},
    "rno": {"n_modes": [2, 2], "hidden_channels": 4, "candidate_activation": "tanh"},
}


@pytest.mark.parametrize("kind", ["fno", "uno", "rno"])
@pytest.mark.parametrize("key", ["positional_embedding", "domain_padding"])
def test_model_params_require_explicit_embedding_and_padding(kind: str, key: str) -> None:
    """Refuse a model that leaves the positional embedding or the domain padding to a library default."""
    params = {**_KIND_PARAMS[kind], "positional_embedding": None, "domain_padding": 0.1}
    learning.models.factory.validate_model_params(kind, params, require_channels=False, operator_dimensionality=2)
    del params[key]
    with pytest.raises(ValueError, match=rf"missing required key.*{key}"):
        learning.models.factory.validate_model_params(kind, params, require_channels=False, operator_dimensionality=2)


def test_builders_pass_domain_padding_and_uno_sizes_its_channel_mlps_from_the_hidden_width() -> None:
    """FNO, U-NO and RNO receive the padding; U-NO lifting and projection widths follow their ratios like the FNO's."""
    common = {"in_channels": 3, "out_channels": 2, "hidden_channels": 4, "positional_embedding": None, "domain_padding": 0.1}
    fno = learning.models.factory.build_fno(n_modes=[2, 2], n_layers=1, **common)
    rno = learning.models.factory.build_rno(n_modes=[2, 2], candidate_activation="tanh", **common)
    uno = learning.models.factory.build_uno(n_layers=5, modes_x=8, modes_y=8, lifting_channel_ratio=3, projection_channel_ratio=1.5, **common)
    default_uno = learning.models.factory.build_uno(n_layers=5, modes_x=8, modes_y=8, **common)

    assert [getattr(model.domain_padding, "domain_padding", None) for model in (fno, rno, uno)] == [0.1, 0.1, 0.1]
    assert [model.positional_embedding for model in (fno, rno, uno)] == [None, None, None]
    assert (uno.lifting.fcs[0].out_channels, uno.projection.fcs[0].out_channels) == (12, 6)
    assert (default_uno.lifting.fcs[0].out_channels, default_uno.projection.fcs[0].out_channels) == (8, 8)


def test_transient_spectral_admission_uses_full_y_x_axes() -> None:
    """Admit full-axis modes and reject values beyond the Train grid."""
    config = {
        "task": "transient_drying",
        "model": {"kind": "fno", "params": {"n_modes": [128, 160]}},
    }
    learning.models.factory.validate_transient_model_spatial_shape(config, (251, 401))
    config["model"]["params"]["n_modes"] = [252, 160]
    with pytest.raises(ValueError, match="modes_y"):
        learning.models.factory.validate_transient_model_spatial_shape(config, (251, 401))
    config["model"]["params"]["n_modes"] = [128, 402]
    with pytest.raises(ValueError, match="modes_x"):
        learning.models.factory.validate_transient_model_spatial_shape(config, (251, 401))


def test_transient_uno_schedule_preserves_y_x_axis_order() -> None:
    """Reject a downsampled UNO block whose [Y, X] modes no longer fit."""
    config = {
        "task": "transient_drying",
        "model": {
            "kind": "uno",
            "params": {
                "modes_y": 16,
                "modes_x": 24,
                "n_layers": 5,
                "mode_ratio": 0.5,
                "uno_scalings": [[1.0, 1.0], [0.5, 0.5], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]],
            },
        },
    }
    learning.models.factory.validate_transient_model_spatial_shape(config, (32, 48))
    config["model"]["params"]["modes_x"] = 49
    with pytest.raises(ValueError, match="modes_x"):
        learning.models.factory.validate_transient_model_spatial_shape(config, (32, 48))


def test_immediate_uno_stride_two_grid_passes_layerwise_preflight() -> None:
    """Admit the maintained m64x64 seven-layer UNO schedule on 126x201."""
    config = {
        "task": "transient_drying",
        "model": {
            "kind": "uno",
            "params": {
                "modes_y": 64,
                "modes_x": 64,
                "n_layers": 7,
                "mode_ratio": 0.495,
                "uno_scalings": [
                    [1.0, 1.0],
                    [0.5, 0.5],
                    [0.5, 0.5],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [2.0, 2.0],
                ],
            },
        },
    }

    learning.models.factory.validate_transient_model_spatial_shape(config, (126, 201))


def test_fno_stride_two_grid_checks_modes_against_effective_axes() -> None:
    """Apply FNO spectral admission to the effective, not canonical, grid."""
    config = {
        "task": "transient_drying",
        "model": {"kind": "fno", "params": {"n_modes": [64, 64]}},
    }
    learning.models.factory.validate_transient_model_spatial_shape(config, (126, 201))

    config["model"]["params"]["n_modes"] = [127, 64]
    with pytest.raises(ValueError, match="modes_y"):
        learning.models.factory.validate_transient_model_spatial_shape(config, (126, 201))


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("fno", {"n_modes": [24, 24]}),
        ("rno", {"n_modes": [24, 24]}),
        (
            "uno",
            {
                "modes_y": 64,
                "modes_x": 64,
                "n_layers": 7,
                "mode_ratio": 0.495,
                "uno_scalings": [
                    [1.0, 1.0],
                    [0.5, 0.5],
                    [0.5, 0.5],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [2.0, 2.0],
                ],
            },
        ),
    ],
)
def test_transient_architectures_share_explicit_cross_resolution_capability(
    kind: str,
    params: dict[str, Any],
) -> None:
    """Resolve every maintained transient architecture from UNKNOWN on the original grid."""
    evidence = learning.models.factory.assess_transient_model_spatial_compatibility(
        {"task": "transient_drying", "model": {"kind": kind, "params": params}},
        training_shape=(126, 201),
        evaluation_shape=(251, 401),
    )

    assert evidence["decision"] == "SUPPORTED_WITH_CONTRACT"
    assert evidence["model_kind"] == kind
    assert evidence["training_shape"] == [126, 201]
    assert evidence["evaluation_shape"] == [251, 401]
    assert evidence["forward_output_shape_proof"] == "pending_synthetic_single_step_probe"


def test_transient_architecture_capability_rejects_requested_grid_before_inference() -> None:
    """Return an actionable unsupported decision when retained modes cannot fit."""
    evidence = learning.models.factory.assess_transient_model_spatial_compatibility(
        {"task": "transient_drying", "model": {"kind": "rno", "params": {"n_modes": [64, 64]}}},
        training_shape=(126, 201),
        evaluation_shape=(32, 48),
    )

    assert evidence["decision"] == "UNSUPPORTED"
    assert evidence["model_kind"] == "rno"
    assert "modes" in evidence["reason"]


@pytest.mark.parametrize("kind", ["fno", "uno"])
def test_one_architecture_label_is_reused_across_config_run_and_display(kind: str) -> None:
    """Stage, physics, seed and optimizer remain distinct from architecture display."""
    raw = configs.direct_config(model_kind=kind)
    config = configs.resolve(raw)
    label = learning.models.naming.architecture_label(config)
    core = learning.models.naming.core_architecture_label(config)
    assert label == core  # The synthetic model carries no non-default switches.
    assert config["run"]["layout"]["model_label"] == core
    assert core in config["run"]["name"]
    assert experiments.config.loader.resolved_model_variant(config) == label
    config["run"]["seed"] += 1
    config["optimizer"]["lr"] *= 2
    config["loss"]["physics"]["enabled"] = True
    assert learning.models.naming.architecture_label(config) == label


def _rno_params(**overrides: Any) -> dict[str, Any]:
    """Return compact valid RNO model parameters."""
    params: dict[str, Any] = {
        "n_modes": [2, 2],
        "hidden_channels": 4,
        "n_layers": 2,
        "candidate_activation": "tanh",
        "positional_embedding": None,
        "domain_padding": 0.1,
    }
    params.update(overrides)
    return params


def test_rno_uses_tanh_candidate_cells_with_unchanged_state_dict_keys() -> None:
    """Convert every block cell while keeping the library parameter layout."""
    from neuralop.models import RNO  # noqa: PLC0415

    model = learning.models.factory.build_rno(in_channels=3, out_channels=2, **_rno_params(positional_embedding="grid"))
    blocks = [module for module in model.modules() if isinstance(module, rno_cell.rno_block.RNOBlock)]
    assert len(blocks) == 2
    assert all(type(block.cell) is rno_cell.TanhCandidateRNOCell for block in blocks)
    plain = RNO(
        n_modes=(2, 2),
        in_channels=3,
        out_channels=2,
        hidden_channels=4,
        n_layers=2,
        rno_skip=False,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        positional_embedding="grid",
        channel_mlp_skip="soft-gating",
        fno_skip="linear",
        return_sequences=False,
    )
    assert list(model.state_dict()) == list(plain.state_dict())
    with pytest.raises(RuntimeError, match="found TanhCandidateRNOCell"):
        rno_cell.use_tanh_candidate_cells(model, n_layers=2)
    with pytest.raises(RuntimeError, match="found 2"):
        rno_cell.use_tanh_candidate_cells(plain, n_layers=3)


def test_tanh_candidate_cell_keeps_hidden_state_bounded() -> None:
    """Hold |h| <= max(1, |h0|) at every step even under large random inputs."""
    torch.manual_seed(0)
    model = learning.models.factory.build_rno(in_channels=3, out_channels=2, **_rno_params(n_layers=1))
    cell = next(module for module in model.modules() if isinstance(module, rno_cell.TanhCandidateRNOCell))
    hidden = 3.0 * torch.randn(2, 4, 6, 6)
    with torch.no_grad():
        for _ in range(256):
            previous = hidden
            hidden = cell(10.0 * torch.randn(2, 4, 6, 6), previous)
            assert bool((hidden.abs() <= previous.abs().clamp(min=1.0) + 1e-6).all())


def test_tanh_candidate_cell_is_the_library_forward_with_only_the_activation_swapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gates, reset and update stay the library's; only SELU becomes tanh."""
    torch.manual_seed(0)
    model = learning.models.factory.build_rno(in_channels=3, out_channels=2, **_rno_params(n_layers=1))
    cell = next(module for module in model.modules() if isinstance(module, rno_cell.TanhCandidateRNOCell))
    inputs, hidden = torch.randn(2, 4, 6, 6), torch.randn(2, 4, 6, 6)
    with torch.no_grad():
        ours = cell(inputs, hidden)
        monkeypatch.setattr(rno_cell.rno_block.F, "selu", torch.tanh)
        library = rno_cell.rno_block.RNOCell.forward(cell, inputs, hidden)
    assert torch.equal(ours, library)


def test_rno_production_step_carry_equals_the_full_sequence_recurrence() -> None:
    """
    Carry a padded RNO state through the production step, truncation and snapshot selection.

    ``predict_step`` per step with a detach at every step must reproduce the library's
    own sequence pass, and per-sample snapshot selection must return the states
    the library reaches after each sample's own step count, domain padding included.
    """
    torch.manual_seed(0)
    model = learning.models.factory.build_rno(in_channels=3, out_channels=4, **_rno_params(domain_padding=0.25)).eval()
    sequence = torch.randn(2, 5, 3, 8, 8)
    with torch.no_grad():
        full, full_hidden = model(sequence, return_hidden_states=True)
        _, early_hidden = model(sequence[:, :3], return_hidden_states=True)
        hidden = None
        snapshots: dict[int, object | None] = {0: None}
        for step in range(sequence.shape[1]):
            prediction, hidden = rollout.predict_step(model, sequence[:, step], model_kind="rno", hidden=hidden)
            hidden = rollout.detach_hidden(hidden)
            snapshots[step + 1] = hidden
    torch.testing.assert_close(prediction, full)
    selected = rollout.select_hidden_per_sample(snapshots, torch.tensor([3, 5]))
    assert isinstance(selected, list)
    for carried, early, late in zip(selected, early_hidden, full_hidden, strict=True):
        torch.testing.assert_close(model.domain_padding.unpad(carried), torch.stack([early[0], late[1]]))


def test_rno_label_omits_the_single_admitted_candidate_activation() -> None:
    """The required tanh candidate names no alternative, so it is no label switch."""
    config = {"model": {"kind": "rno", "params": _rno_params()}}
    assert learning.models.naming.architecture_label(config) == learning.models.naming.core_architecture_label(config)


@pytest.mark.parametrize("candidate_activation", ["selu", None])
def test_rno_rejects_missing_or_non_tanh_candidate_activation(candidate_activation: str | None) -> None:
    """Fail closed on a missing key and on any activation other than tanh."""
    params = _rno_params()
    if candidate_activation is None:
        del params["candidate_activation"]
        match = "missing required key"
    else:
        params["candidate_activation"] = candidate_activation
        match = "candidate_activation must be 'tanh'"
    with pytest.raises(ValueError, match=match):
        learning.models.factory.validate_model_params("rno", params, require_channels=False, operator_dimensionality=2)
    if candidate_activation is not None:
        with pytest.raises(ValueError, match="candidate_activation='tanh'"):
            learning.models.factory.build_rno(in_channels=3, out_channels=2, **params)

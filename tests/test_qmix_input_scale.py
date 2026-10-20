"""Graph input scale, private-state gradients and incompatible checkpoint gates."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.config import load_run_config
from src.env.environment import U2UMECEnvironment
from src.models.ca_gat_mappo import ActorObservationTensorizer
from src.models.factorized_gat_qmix import FactorizedGATQNetwork, stabilize_edge_features
from src.models.qmix_continuation import load_continuation_checkpoint
from src.models.qmix_trainer import QMIXTrainer, checkpoint_bytes, load_checkpoint


@pytest.fixture
def setup(tmp_path):
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    config = load_run_config(Path(__file__).resolve().parents[1] /
        "tests/fixtures/configs/qmix_legacy/fa-gat-qmix-seed42-8k-episode-return-development.json",
        cli_overrides={"training.mappo.training_device": "cpu",
            "output.logs_dir": str(tmp_path / "logs"), "output.plots_dir": str(tmp_path / "plots"),
            "output.dashboard_logs_dir": str(tmp_path / "dashboard_logs")})
    observations = U2UMECEnvironment(config).reset().observations
    batch = ActorObservationTensorizer(config).encode_step(observations)
    generator = torch.Generator().manual_seed(16)
    stress = torch.rand(batch.edge_features.shape, generator=generator) * 1.e6
    batch = replace(batch, edge_features=stress, neighbor_mask=torch.ones_like(batch.neighbor_mask))
    yield config, FactorizedGATQNetwork(config), batch
    torch.set_num_threads(before)


def test_edge_transform_preserves_order_sign_zeros_and_finite_extremes():
    limit = torch.finfo(torch.float32).max
    values = torch.tensor([-limit, -1.e6, -1., 0., 1., 1.e6, limit])
    original = values.clone()
    result = stabilize_edge_features(values)
    assert torch.isfinite(result).all() and (result[1:] > result[:-1]).all()
    assert torch.equal(result.sign(), values.sign()) and result[3] == 0
    assert result.abs().max() < 90 and torch.equal(values, original)


def graph_forward(network, batch, *, compressed):
    agents = network.spec.uav_count
    self_encoded = network.self_encoder(batch.self_features).reshape(agents, -1)
    public_encoded = network.public_encoder(batch.neighbor_public_features).reshape(agents, agents, -1)
    edge = stabilize_edge_features(batch.edge_features) if compressed else batch.edge_features
    graph = network.graph(self_encoded, public_encoded, edge.reshape(agents, agents, -1),
                          batch.neighbor_mask.reshape(agents, agents))
    out, hidden = network.gru(graph[:, None], network.initial_hidden().reshape(1, agents, -1))
    return out.reshape(1, 1, agents, -1), hidden.reshape(1, agents, -1)


def test_production_encode_compresses_edges_once_and_preserves_raw_batch(setup):
    _, network, batch = setup
    raw = batch.edge_features.clone()
    actual, hidden = network.encode(batch, network.initial_hidden())
    expected, expected_hidden = graph_forward(network, batch, compressed=True)
    assert torch.equal(actual, expected) and torch.equal(hidden, expected_hidden)
    assert torch.equal(batch.edge_features, raw)
    for mask in batch.action_masks.values():
        assert mask.dtype == torch.bool


def test_large_edge_ratios_no_longer_suppress_private_queue_sensitivity(setup):
    _, network, batch = setup
    gradients = []
    for compressed in (False, True):
        features = batch.self_features.clone().requires_grad_(True)
        recurrent, _ = graph_forward(network, replace(batch, self_features=features), compressed=compressed)
        q = network.heads["route"](recurrent)
        gradient = torch.autograd.grad((q[..., 2] - q[..., 1]).sum(), features)[0]
        gradients.append(float(gradient[..., 9:89].abs().mean()))
    assert gradients[1] > 1.e-6
    assert gradients[1] > 100 * gradients[0]


@pytest.mark.parametrize("loader", [load_checkpoint, load_continuation_checkpoint])
def test_v3_weights_are_rejected_before_loading_or_continuing(setup, tmp_path, loader):
    config, _, _ = setup
    data = QMIXTrainer(config).payload("periodic")
    data["schema"] = "factorized_gat_qmix.checkpoint.v3"
    path = tmp_path / "obsolete.pt"
    path.write_bytes(checkpoint_bytes(data))
    with pytest.raises(ValueError):
        loader(path)


@pytest.mark.parametrize("loader", [load_checkpoint, load_continuation_checkpoint])
def test_checkpoint_must_declare_current_actor_transform(setup, tmp_path, loader):
    config, _, _ = setup
    data = QMIXTrainer(config).payload("periodic")
    assert data["schema"] == "factorized_gat_qmix.checkpoint.v10"
    data["implementation_contract"].pop("actor_edge_transform")
    path = tmp_path / "missing_transform.pt"
    path.write_bytes(checkpoint_bytes(data))
    with pytest.raises(ValueError, match="actor edge transform"):
        loader(path)

"""Independent exhaustive oracles for the seven-branch QMIX local maximum."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import load_run_config
from src.env.actions import ActionProposal
from src.env.environment import U2UMECEnvironment
from src.env.observation import BRANCH_ORDER
from src.models.ca_gat_mappo import ActorObservationTensorizer
from src.models.factorized_gat_qmix import FactorizedGATQNetwork
from src.models.qmix_trainer import QMIXTrainer, checkpoint_bytes, load_checkpoint


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def setup(tmp_path):
    config = load_run_config(ROOT / "tests/fixtures/configs/qmix_legacy/fa-gat-qmix-seed42-40k.json",
        cli_overrides={"training.mappo.training_device": "cpu", "output.logs_dir": str(tmp_path / "logs"),
                       "output.plots_dir": str(tmp_path / "plots"),
                       "output.dashboard_logs_dir": str(tmp_path / "dashboard_logs")})
    observations = U2UMECEnvironment(config).reset().observations
    network = FactorizedGATQNetwork(config)
    masks = []
    for obs in observations:
        m = obs.action_masks
        route = np.zeros(len(m.route_domain), dtype=bool)
        route[1:3] = True
        tx = np.zeros(len(m.tx_select_domain), dtype=bool)
        tx[[0, 1]] = True
        cpu = np.zeros(len(m.cpu_queue_domain), dtype=bool)
        cpu[:3] = True
        frequencies = np.zeros_like(m.cpu_frequency_energy_mask)
        frequencies[:, 0] = True
        frequencies[0, 1] = True
        frequencies[1, -1] = True
        power = np.zeros(len(m.power_level_domain), dtype=bool)
        power[[0, 1, -1]] = True
        masks.append(replace(m, route_mask=route, tx_select_mask=tx, cpu_queue_mask=cpu,
            cpu_frequency_energy_mask=frequencies, power_energy_mask=power,
            route_branch_active=True, tx_branch_active=True, cpu_queue_branch_active=True))
    batch = ActorObservationTensorizer(config).encode_step(observations)
    return config, network, batch, ((tuple(masks),),)


def enumerate_legal(mask, prefix=None):
    prefix = {} if prefix is None else prefix
    if len(prefix) == len(BRANCH_ORDER):
        yield ActionProposal(uav_id=mask.uav_id, **prefix)
        return
    branch = BRANCH_ORDER[len(prefix)]
    for value, legal in zip(mask.domain_for(branch), mask.mask_for(branch, prefix)):
        if legal:
            yield from enumerate_legal(mask, {**prefix, branch: value})


def set_scores(network, scores):
    with torch.no_grad():
        for branch, head in network.heads.items():
            for parameter in head.parameters():
                parameter.zero_()
            value = torch.tensor(scores[branch])
            if branch == "cpu_frequency":
                value = value.repeat(network.cpu_queue_count)
            head[-1].bias.copy_(value)


def test_cpu_idle_and_tx_idle_must_include_the_whole_service_tuple(setup):
    _, network, batch, contracts = setup
    scores = {b: [0.] * network.spec.action_dimensions[b] for b in BRANCH_ORDER}
    scores["cpu_queue"][:3] = [5., 4., 3.]
    scores["cpu_frequency"] = [-1., 10., -2., 1.]
    scores["tx_select"][:2] = [5., 4.]
    scores["resource_group"] = [0., 1., 1., 1., 1., 2.]
    scores["resource_width"] = [1., 8.]
    scores["power_level"] = [-100., 2., 0., 1.]
    set_scores(network, scores)
    out = network.actions(batch, network.initial_hidden(), contracts)
    for proposal in out.proposals[0][0]:
        assert proposal.cpu_queue == contracts[0][0][0].cpu_queue_domain[1]
        assert proposal.cpu_frequency == .25
        assert proposal.tx_select != "idle"
        # Group 5 has the highest own score but cannot use width 2.
        assert proposal.resource_group == 1 and proposal.resource_width == 2
        assert proposal.power_level == .25


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_local_greedy_equals_exhaustive_joint_action_maximum(setup, seed):
    _, network, batch, contracts = setup
    rng = np.random.default_rng(seed)
    scores = {b: rng.normal(size=network.spec.action_dimensions[b]).tolist() for b in BRANCH_ORDER}
    set_scores(network, scores)
    out = network.actions(batch, network.initial_hidden(), contracts)
    for agent, mask in enumerate(contracts[0][0]):
        maximum = -float("inf")
        for proposal in enumerate_legal(mask):
            active = mask.active_indicators(proposal)
            value = sum(float(out.branch_values[b][0, 0, agent, mask.domain_for(b).index(getattr(proposal, b))])
                        for b, flag in zip(BRANCH_ORDER, active) if flag)
            maximum = max(maximum, value)
        assert float(out.utilities[0, 0, agent].detach()) == pytest.approx(maximum, abs=1.e-6)
        assert mask.is_legal(out.proposals[0][0][agent])
    teacher = network.actions(batch, network.initial_hidden(), contracts, proposals=out.proposals)
    assert torch.equal(teacher.utilities, out.utilities)


def test_negative_service_suffix_can_make_idle_optimal_without_forcing_work(setup):
    _, network, batch, contracts = setup
    scores = {b: [0.] * network.spec.action_dimensions[b] for b in BRANCH_ORDER}
    scores["cpu_queue"][:3] = [1., 10., 10.]
    scores["cpu_frequency"] = [-20.] * 4
    scores["tx_select"][:2] = [1., 10.]
    scores["resource_group"] = [-20.] * 6
    scores["resource_width"] = [-20.] * 2
    scores["power_level"] = [-20.] * 4
    set_scores(network, scores)
    out = network.actions(batch, network.initial_hidden(), contracts)
    assert all(p.cpu_queue == "idle" and p.tx_select == "idle" for p in out.proposals[0][0])
    # Inactive TX resources and CPU frequency add zero, not their negative scores.
    assert torch.equal(out.utilities.detach(), torch.full((1, 1, 4), 2.))


def test_zero_energy_masks_keep_the_existing_zero_frequency_fallback(setup):
    _, network, batch, contracts = setup
    changed = []
    for m in contracts[0][0]:
        energy = np.zeros_like(m.cpu_frequency_energy_mask)
        energy[:, 0] = True
        power = np.zeros_like(m.power_energy_mask)
        power[0] = True
        changed.append(replace(m, cpu_frequency_energy_mask=energy, power_energy_mask=power))
    scores = {b: [0.] * network.spec.action_dimensions[b] for b in BRANCH_ORDER}
    scores["cpu_queue"][1] = 10.
    scores["tx_select"][1] = 10.
    set_scores(network, scores)
    out = network.actions(batch, network.initial_hidden(), ((tuple(changed),),))
    assert all(p.cpu_frequency == 0. and p.power_level == 0. for p in out.proposals[0][0])
    assert all(m.is_legal(p) for m, p in zip(changed, out.proposals[0][0]))


@pytest.mark.parametrize("obsolete_schema", ["factorized_gat_qmix.checkpoint.v2", "factorized_gat_qmix.checkpoint.v3", "factorized_gat_qmix.checkpoint.v4", "factorized_gat_qmix.checkpoint.v5", "factorized_gat_qmix.checkpoint.v6", "factorized_gat_qmix.checkpoint.v7"])
def test_obsolete_checkpoint_schema_is_rejected(setup, tmp_path, obsolete_schema):
    config, _, _, _ = setup
    payload = QMIXTrainer(config).payload("periodic")
    payload["schema"] = obsolete_schema
    path = tmp_path / "obsolete.pt"
    path.write_bytes(checkpoint_bytes(payload))
    with pytest.raises(ValueError, match="not a Factorized"):
        load_checkpoint(path)

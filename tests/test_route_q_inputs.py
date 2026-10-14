"""P0 tensor-only gates. Environment creation/rollout is explicitly blocked."""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.config import ConfigError, load_run_config
from src.models.ca_gat_mappo import CAGATMAPPOActor, MAPPOCentralizedCritic, MAPPOTensorSpec
from src.models.route_option_q import RouteOptionQ, counterfactual_values, prepare_route_q, route_q_loss
from src.models.route_q_inputs import StateField, RouteQInputConditioner, centralized_q_fields
from src.evaluation.route_q_probe import synthetic_fit
from tests.test_route_option_q import tiny_batch

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT/'tests/fixtures/configs'
BASE = CONFIGS/'e14d-v1-route-counterfactual-q-seed42-8k-dev.json'
FIELDS = (StateField('ratio', 0, 4, 'ratio', signed_log=True),
          StateField('mask', 4, 6, 'bool', discrete=True))


@pytest.fixture(autouse=True)
def tensor_only(monkeypatch):
    from src.env.environment import U2UMECEnvironment
    def forbidden(*args, **kwargs):
        raise AssertionError('P0 tests must not instantiate an environment')
    monkeypatch.setattr(U2UMECEnvironment, '__init__', forbidden)
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def make_q(mode='normalized'):
    return RouteOptionQ(6, 11, 16, 42, input_mode=mode, state_fields=FIELDS)


def inputs():
    s = torch.tensor([[1e5, -1e4, 0., 1., 0., 1.], [2e5, 5., 0., 3., 1., 0.]])
    c = torch.randn(2, 6, 11, generator=torch.Generator().manual_seed(42))
    c[..., -3:] = nn.functional.one_hot(torch.arange(6)%3, 3).float()
    legal = torch.tensor([[False, True, True, True, False, True], [False, True, True, False, True, False]])
    return s, c, legal


def test_legacy_exact_reference_and_rng():
    rng = torch.get_rng_state().clone()
    q = make_q('legacy')
    assert torch.equal(rng, torch.get_rng_state())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42+140004)
        old = nn.Sequential(nn.Linear(17, 16), nn.LeakyReLU(), nn.LayerNorm(16), nn.Linear(16, 1))
        nn.init.zeros_(old[-1].weight); nn.init.zeros_(old[-1].bias)
    assert set(q.state_dict()) == set(old.state_dict(prefix='network.'))
    for k, v in old.state_dict().items(): assert torch.equal(v, q.network.state_dict()[k])
    with torch.no_grad(): old[-1].weight.fill_(.01); q.network[-1].weight.copy_(old[-1].weight)
    s, c, legal = inputs()
    expected = torch.zeros_like(legal, dtype=s.dtype)
    expected[legal] = old(torch.cat((s[:, None].expand(2, 6, 6)[legal], c[legal]), -1)).squeeze(-1)
    assert torch.equal(q(s, c, legal), expected)


@pytest.mark.parametrize('mode', ['legacy', 'normalized', 'dual_branch'])
def test_shapes_masks_empty_and_detach(mode):
    q = make_q(mode); s, c, legal = inputs()
    s.requires_grad_(); c.requires_grad_()
    c.data[~legal] = float('nan')
    output = q(s, c, legal)
    assert output.shape == legal.shape and torch.isfinite(output).all()
    assert (output[~legal] == 0).all()
    output.sum().backward()
    assert s.grad is None and c.grad is None
    assert q(s, c, torch.zeros_like(legal)).eq(0).all()


@pytest.mark.parametrize('mode', ['normalized', 'dual_branch'])
def test_discrete_types_and_zero_variance_preserved(mode):
    q = make_q(mode); s, c, legal = inputs()
    c[..., :8] = 0
    a, b = q.preprocess(s, c)
    assert torch.equal(a[..., 4:], s[..., 4:])
    assert torch.equal(b[..., -3:], c[..., -3:])
    assert b[..., :8].eq(0).all()
    c[..., :8] = 7
    _, b = q.preprocess(s, c)
    assert torch.isfinite(b).all() and b[..., :8].eq(1).all()
    assert torch.equal(b[..., -3:], c[..., -3:])


@pytest.mark.parametrize('mode', ['normalized', 'dual_branch'])
def test_extreme_units_and_nonfinite_fail_closed(mode):
    q = make_q(mode); s, c, legal = inputs()
    s[:, :4] = torch.finfo(torch.float32).max
    c[..., :8] = -torch.finfo(torch.float32).max
    assert torch.isfinite(q(s, c, legal)).all()
    for value in (float('nan'), float('inf'), -float('inf')):
        bad = s.clone(); bad[0, 0] = value
        with pytest.raises(ValueError, match='nonfinite'): q(bad, c, legal)
        bad = c.clone(); bad[0, 1, 0] = value
        with pytest.raises(ValueError, match='nonfinite'): q(s, bad, legal)


@pytest.mark.parametrize('mode', ['normalized', 'dual_branch'])
def test_candidate_independence_permutation_and_type_path(mode):
    q = make_q(mode); s, c, legal = inputs()
    with torch.no_grad(): q.network[-1].weight.copy_(torch.linspace(-.2, .2, 16)[None])
    original = q(s, c, legal)
    changed = c.clone(); changed[0, 1, :8] += 2
    altered = q(s, changed, legal)
    assert torch.equal(original[0, 2:], altered[0, 2:])
    assert not torch.equal(original[0, 1], altered[0, 1])
    changed = c.clone(); changed[0, 1, -3:] = c[0, 2, -3:]
    assert not torch.equal(q(s, changed, legal)[0, 1], original[0, 1])
    perm = torch.tensor([5, 0, 2, 1, 4, 3])
    torch.testing.assert_close(q(s, c[:, perm], legal[:, perm]), original[:, perm])
    for count in (2, 3, 4, 6):
        assert q(s, c[:, :count], legal[:, :count]).shape == (2, count)


@pytest.mark.parametrize('mode', ['normalized', 'dual_branch'])
def test_synthetic_candidate_ranking_and_gradients(mode):
    result = synthetic_fit(mode)
    assert result['initial_q_range'] == 0  # expected zero-output cold start
    assert result['pairwise_accuracy'] >= .99
    assert result['rmse'] < .02
    assert result['heldout_pairwise_accuracy'] > .9
    key = 'candidate_projection.0.weight' if mode == 'dual_branch' else 'network.0.weight'
    assert result['second_step_gradients'][key] > 0
    assert result['parameter_update_norms'][key] > 0


@pytest.mark.parametrize('mode', ['normalized', 'dual_branch'])
def test_frozen_credit_chosen_only_and_no_shared_gradients(mode):
    _, policy, task, _ = tiny_batch()
    # Retain factual actions/probabilities, replace only independent Q inputs.
    policy.route_candidate_features = torch.randn(1, 1, 4, 6, 11, requires_grad=True)
    policy.route_candidate_features.data[..., -3:] = nn.functional.one_hot(torch.arange(6)%3, 3).float()
    state = torch.randn(1, 1, 6, requires_grad=True)
    q = make_q(mode)
    with torch.no_grad(): q.network[-1].weight.fill_(.02)
    critic = SimpleNamespace(route_option_q=q, main_critic=nn.Linear(6, 1))
    batch = prepare_route_q(critic, policy, state, task, .7)
    for i, record in enumerate(batch.records):
        assert abs(record['final_route_advantage']-(record['shared_advantage']+.7*float(batch.counterfactual_advantage[i]))) < 1e-6
        assert record['coefficient'] == 0
    b, cf, _ = counterfactual_values(batch.frozen_q, policy.probabilities['route'].reshape(-1, 4, 6)[0, :3],
                                    batch.legal, batch.chosen)
    torch.testing.assert_close(b, batch.baseline); torch.testing.assert_close(cf, batch.counterfactual_advantage)
    snapshots = [t.clone() for t in (batch.frozen_q, batch.baseline, batch.advantage, batch.targets)]
    loss, output = route_q_loss(critic, batch); output.retain_grad(); loss.backward()
    selected = torch.zeros_like(batch.legal).scatter_(-1, batch.chosen[:, None], True)
    assert output.grad[~selected].eq(0).all()
    assert state.grad is None and policy.route_candidate_features.grad is None
    assert all(p.grad is None for p in critic.main_critic.parameters())
    assert all(not t.requires_grad for t in (batch.baseline, batch.advantage, batch.targets))
    for actual, old in zip((batch.frozen_q, batch.baseline, batch.advantage, batch.targets), snapshots):
        assert torch.equal(actual, old)
    task.samples[0]['return'] = -17
    future = prepare_route_q(critic, policy, state, task, .7)
    assert torch.equal(future.state, batch.state) and torch.equal(future.candidates, batch.candidates)
    assert torch.equal(future.frozen_q, batch.frozen_q)


@pytest.mark.parametrize('mode', ['normalized', 'dual_branch'])
def test_fixed_normalizer_checkpoint_and_eval_no_statistics(mode):
    q = make_q(mode); s, c, legal = inputs()
    before = copy.deepcopy(q.state_dict())
    q.eval(); q(s, c, legal); q(s*2, c*3, legal)
    assert all(torch.equal(before[k], v) for k, v in q.state_dict().items())
    memory = io.BytesIO(); torch.save(q.state_dict(), memory); memory.seek(0)
    restored = make_q(mode); restored.load_state_dict(torch.load(memory, weights_only=True))
    assert torch.equal(q(s, c, legal), restored(s, c, legal))
    assert restored.conditioner.schema_version.item() == 1
    assert not any('running' in k for k in before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('mode', ['normalized', 'dual_branch'])
def test_cuda_tensor_only_forward_backward_and_buffers(mode):
    q = make_q(mode).cuda()
    s, c, legal = (t.cuda() for t in inputs())
    with torch.no_grad(): q.network[-1].weight.copy_(torch.linspace(-.2, .2, 16, device='cuda')[None])
    output = q(s, c, legal)
    output[legal].square().mean().backward()
    assert torch.isfinite(output).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in q.parameters())
    assert q.conditioner.gain.device.type == 'cuda'
    assert q.conditioner.gain.dtype == next(q.parameters()).dtype


def test_schema_dynamic_size_and_raw_field_semantics():
    config = load_run_config(BASE)
    for n in (2, 3, 4, 7):
        for r in (1, 5, 20):
            env = replace(config.environment, uav_count=n, ru_count=r)
            fields = centralized_q_fields(SimpleNamespace(environment=env))
            expected = 1+13*n+7*n*n+3*n*r+6*n*n*r+10*n+5*n*n+6*len(env.building_layout)
            assert fields[-1].stop == expected
            for a, b in zip(fields, fields[1:]): assert a.stop == b.start
    fields = centralized_q_fields(config); by_name = {f.name: f for f in fields}
    assert fields[-1].stop == MAPPOTensorSpec.from_config(config).centralized_state_dim
    assert by_name['historical_quality'].stop-by_name['historical_quality'].start == 4*4*20
    assert by_name['csi_valid'].stop-by_name['csi_valid'].start == 16
    assert by_name['historical_denominator'].stop-by_name['historical_denominator'].start == 80


def test_schema_rejects_gaps_discrete_scaling_and_invalid_mode():
    with pytest.raises(ValueError): RouteQInputConditioner(6, 11, FIELDS[:1])
    with pytest.raises(ValueError): RouteQInputConditioner(6, 10, FIELDS)
    with pytest.raises(ValueError): RouteQInputConditioner(6, 11, (StateField('bad', 0, 6, 'bool', 2., discrete=True),))
    with pytest.raises(ValueError): RouteOptionQ(6, 11, 16, 42, input_mode='unknown')


def test_synthetic_entrypoint_needs_no_checkpoint_or_environment(tmp_path):
    import json
    from src.evaluation.route_q_probe import main
    rng = torch.get_rng_state().clone()
    main(['--synthetic-only', '--synthetic-steps', '2', '--output-dir', str(tmp_path)])
    result = json.loads((tmp_path/'route_q_probe.json').read_text(encoding='utf-8'))
    assert set(result['synthetic']) == {'legacy', 'normalized', 'dual_branch'}
    assert torch.equal(rng, torch.get_rng_state())


def test_three_configs_differ_only_in_q_mode_and_output():
    base = load_run_config(BASE)
    raw = base.resolved_dict()
    for mode in ('legacy', 'normalized', 'dual_branch'):
        cfg = load_run_config(CONFIGS/f'e14d-p0-{mode.replace("_", "-")}-seed42-8k-dev.json')
        d = cfg.resolved_dict(); d['training']['mappo'].pop('route_q_input_mode', None)
        d['output'] = raw['output']; assert d == raw
        assert cfg.training.mappo.cpu_anti_starvation_mode == 'deadline_feasibility_v1'
        assert cfg.training.mappo.tx_upper_credit_mode == 'paired_shadow_v1'
    assert base.config_hash == '2cd92efd656e373ad4c563866e7a2c39f14af4e5c6eef89b633c125565adf87d'


@pytest.mark.parametrize('mode', ['legacy', 'normalized', 'dual_branch'])
def test_actor_main_critic_and_rng_identical_across_modes(mode):
    base = load_run_config(BASE)
    config = replace(base, training=replace(base.training, mappo=replace(base.training.mappo, route_q_input_mode=mode)))
    rng = torch.get_rng_state().clone()
    actor0, actor1 = CAGATMAPPOActor(base), CAGATMAPPOActor(config)
    critic0, critic1 = MAPPOCentralizedCritic(base), MAPPOCentralizedCritic(config)
    assert torch.equal(rng, torch.get_rng_state())
    for k, v in actor0.state_dict().items(): assert torch.equal(v, actor1.state_dict()[k])
    for k, v in critic0.state_dict().items():
        if not k.startswith('route_option_q.'): assert torch.equal(v, critic1.state_dict()[k])
    clone = MAPPOCentralizedCritic(config); clone.load_state_dict(critic1.state_dict())
    assert all(torch.equal(v, clone.state_dict()[k]) for k, v in critic1.state_dict().items())


def test_old_real_checkpoint_strict_load_without_evaluation():
    from src.models.ca_gat_mappo_checkpoint import load_checkpoint_payload
    checkpoint = (ROOT/'tests/fixtures/historical/checkpoints/e14d_route_counterfactual_q_seed42_8k/step_4000.pt')
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload = load_checkpoint_payload(checkpoint)
    config = load_run_config(BASE)
    MAPPOCentralizedCritic(config).load_state_dict(payload['model_state']['critic'], strict=True)
    CAGATMAPPOActor(config).load_state_dict(payload['model_state']['actor'], strict=True)
    assert config.config_hash == payload['config_hash']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == sha


def test_configuration_modes_validate_and_disabled_is_not_silent():
    base = load_run_config(BASE)
    for mode, value in (('unknown', base.training.mappo.route_value_mode), ('normalized', 'off')):
        cfg = replace(base, training=replace(base.training, mappo=replace(base.training.mappo,
                      route_q_input_mode=mode, route_value_mode=value)))
        with pytest.raises(ConfigError): cfg.validate()


def test_cpu_feasibility_gate_boundaries_without_environment():
    from src.models.integrated_scheduler import cpu_deadline_feasibility, cpu_idle_deadline_critical
    contract = SimpleNamespace(cpu_queue_domain=('idle', 0), cpu_queue_mask=[True, True],
        cpu_frequency_domain=(0., .5, 1.), cpu_deadline_context=((40e6, 2, 1e9),),
        mask_for=lambda *args: (True, True, True))
    assert cpu_deadline_feasibility(40e6, 2, 1e9, .02) == (2, 0)
    assert cpu_idle_deadline_critical(contract, .02)
    for slack in (1, 3, 45):
        contract.cpu_deadline_context = ((40e6, slack, 1e9),)
        assert not cpu_idle_deadline_critical(contract, .02)
    contract.cpu_deadline_context = ((40e6, 2, 1e9),)
    contract.mask_for = lambda *args: (True, False, False)
    assert not cpu_idle_deadline_critical(contract, .02)
    contract.mask_for = lambda *args: (True, True, True)
    contract.cpu_queue_mask = (True, False)
    assert not cpu_idle_deadline_critical(contract, .02)


@pytest.mark.parametrize('mode', ['normalized', 'dual_branch'])
def test_q_parameters_in_original_critic_optimizer_and_candidate_gradient(mode):
    from src.models.ca_gat_mappo_update import CAGATMAPPORecurrentPPOUpdater
    base = load_run_config(BASE)
    cfg = replace(base, training=replace(base.training, mappo=replace(base.training.mappo,
                  route_q_input_mode=mode, training_device='cpu')))
    actor, critic = CAGATMAPPOActor(cfg), MAPPOCentralizedCritic(cfg)
    updater = CAGATMAPPORecurrentPPOUpdater(actor, critic, cfg)
    optimizer_ids = {id(p) for group in updater.optimizers.critic_optimizer.param_groups for p in group['params']}
    actor_ids = {id(p) for group in updater.optimizers.actor_optimizer.param_groups for p in group['params']}
    assert {id(p) for p in critic.route_option_q.parameters()} <= optimizer_ids
    assert not ({id(p) for p in critic.route_option_q.parameters()} & actor_ids)
    assert len(updater.optimizers.critic_optimizer.param_groups) == 1
    assert updater.optimizers.critic_optimizer.param_groups[0]['lr'] == base.training.mappo.critic_learning_rate
    q = make_q(mode); s, c, legal = inputs()
    with torch.no_grad(): q.network[-1].weight.copy_(torch.linspace(-.2, .2, 16)[None])
    loss = (q(s, c, legal)[legal]-1).square().mean(); loss.backward()
    grad = q.candidate_projection[0].weight.grad if mode == 'dual_branch' else q.network[0].weight.grad[:, 6:]
    assert torch.linalg.vector_norm(grad) > 0

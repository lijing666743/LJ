"""Read-only saved-rollout target audit; no environment or optimizer steps."""
from pathlib import Path
import unittest

import torch

from src.config import load_run_config
from src.models.ca_gat_mappo_checkpoint import restore_active_rollout
from src.models.ca_gat_mappo_gae import compute_rollout_gae, compute_gae_and_returns
from src.models.ca_gat_mappo_rollout import CAGATMAPPORolloutChunk
from src.models.ca_gat_mappo_ppo import compute_configured_ppo_objective_and_loss
from src.models.tx_upper_counterfactual import (
    TXUpperCounterfactualCriticV1, selected_action_q_loss, team_return_target,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'tests/fixtures/configs/e10-v1-tx-counterfactual-credit-seed43-32k.json'
EXPECTED_HASH = '68906491b0320431393759763eed0e3bc51201ec4b64ff4bea253e2a68fd4640'




class E10QTargetSemanticsTests(unittest.TestCase):

    def test_formal_config_is_conserved_role_credit_and_hash_unchanged(self):
        config = load_run_config(CONFIG)
        self.assertEqual(config.training.mappo.agent_credit_mode.value, 'role_decomposed')
        self.assertEqual(config.config_hash, EXPECTED_HASH)




    def test_legacy_scalar_team_return_is_broadcast_once_without_sum(self):
        scalar = torch.tensor([[2., -3.]])
        target = team_return_target(scalar, 4)
        for agent in range(4):
            torch.testing.assert_close(target[..., agent], scalar, rtol=0, atol=0)

    def check_agent_action_alignment(self, device):
        qnet = TXUpperCounterfactualCriticV1(3, 2, 4, 43).to(device)
        with torch.no_grad():
            for parameter in qnet.parameters():
                parameter.zero_()
            qnet.network[-1].bias.copy_(torch.arange(8, device=device, dtype=torch.float32))
        q = qnet(torch.zeros(1, 1, 3, device=device))
        torch.testing.assert_close(q[0, 0], torch.arange(8, device=device).float().reshape(4, 2))
        q.retain_grad()
        # Deliberately distinct labels detect accidental agent/action broadcasting.
        labels = torch.tensor([[[10., 20., 30., 40.]]], device=device, requires_grad=True)
        actions = torch.tensor([[[False, True, False, True]]], device=device)
        active = torch.tensor([[[False, False, True, False]]], device=device)
        loss = selected_action_q_loss(q, actions, labels, active)
        loss.backward()
        expected = torch.zeros_like(q); expected[0, 0, 2, 0] = -1
        torch.testing.assert_close(q.grad, expected, rtol=0, atol=0)
        torch.testing.assert_close(qnet.network[-1].bias.grad, expected.flatten(), rtol=0, atol=0)
        self.assertIsNone(labels.grad)

    def test_agent_action_output_and_selected_label_alignment_cpu(self):
        self.check_agent_action_alignment('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_agent_action_output_and_selected_label_alignment_cuda(self):
        self.check_agent_action_alignment('cuda')

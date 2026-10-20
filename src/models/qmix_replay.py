"""Contiguous episode chunks, restricted-load serialization, and recurrent TD.

Chunks have real lengths and never contain padding or cross an episode. Saved
hidden states are audit data; training reconstructs current-weight prefixes.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, fields

import numpy as np
import torch
from torch import Tensor

from ..config import RunConfig
from ..env.actions import ActionProposal
from ..env.observation import ActionMasks
from .ca_gat_mappo import ActorTensorBatch
from .factorized_gat_qmix import FactorizedQMIX, masked_half_mse
from .qmix_targets import replay_return_targets
from .qmix_recurrent import reconstruct_replay_sequences


def actor_state(batch: ActorTensorBatch) -> dict:
    return {field.name: (
        {k: v.detach().cpu().clone() for k, v in getattr(batch, field.name).items()}
        if isinstance(getattr(batch, field.name), dict)
        else getattr(batch, field.name).detach().cpu().clone()
    ) for field in fields(batch)}


def mask_state(mask: ActionMasks) -> dict:
    return {field.name: (getattr(mask, field.name).tolist()
            if isinstance(getattr(mask, field.name), np.ndarray) else getattr(mask, field.name))
            for field in fields(mask)}


@dataclass
class ReplaySequence:
    actor: ActorTensorBatch
    contracts: tuple
    proposals: tuple
    state: Tensor
    reward: Tensor
    bootstrap: Tensor
    hidden: Tensor
    target_hidden: Tensor
    next_actor: ActorTensorBatch | None
    next_contracts: tuple | None
    next_state: Tensor | None
    episode: int
    first_slot: int
    executed: tuple
    active: Tensor

    @property
    def length(self):
        return self.reward.shape[1]

    def validate(self):
        n = self.length
        if n <= 0 or self.reward.shape != (1, n) or self.bootstrap.shape != (1, n):
            raise ValueError("replay rewards/bootstrap must be [1,real_length]")
        if self.bootstrap.dtype != torch.bool or not torch.isfinite(self.reward).all():
            raise ValueError("invalid replay reward/bootstrap")
        if len(self.contracts) != 1 or len(self.contracts[0]) != n or len(self.proposals[0]) != n:
            raise ValueError("replay time axes differ")
        if len(self.executed) != n or self.state.shape[:2] != (1, n):
            raise ValueError("replay executed/state time axes differ")
        if not self.bootstrap[:, :-1].all():
            raise ValueError("a sequence must not cross an episode boundary")
        has_next = bool(self.bootstrap[0, -1])
        if any((item is not None) != has_next for item in (self.next_actor, self.next_contracts, self.next_state)):
            raise ValueError("last-slot replay must not invent a next decision")
        if self.actor.self_features.shape[:2] != (1, n):
            raise ValueError("replay actor time axes differ")
        if self.active.shape != (*self.actor.self_features.shape[:3], 7):
            raise ValueError("replay active indicators have wrong shape")
        if self.actor.episode_starts[:, 1:].any():
            raise ValueError("replay chunk crosses a recurrent reset")

    def to_state(self):
        self.validate()
        result = {field.name: (getattr(self, field.name).detach().cpu().clone()
                  if isinstance(getattr(self, field.name), Tensor) else copy.deepcopy(getattr(self, field.name)))
                  for field in fields(self)}
        result["actor"] = actor_state(self.actor)
        result["next_actor"] = None if self.next_actor is None else actor_state(self.next_actor)
        result["contracts"] = tuple(tuple(tuple(mask_state(m) for m in step) for step in row) for row in self.contracts)
        result["next_contracts"] = None if self.next_contracts is None else tuple(mask_state(m) for m in self.next_contracts)
        result["proposals"] = tuple(tuple(tuple(vars(p) for p in step) for step in row) for row in self.proposals)
        return result

    @classmethod
    def from_state(cls, state):
        data = copy.deepcopy(state)
        data["actor"] = ActorTensorBatch(**data["actor"])
        data["next_actor"] = None if data["next_actor"] is None else ActorTensorBatch(**data["next_actor"])
        data["contracts"] = tuple(tuple(tuple(ActionMasks(**m) for m in step) for step in row) for row in data["contracts"])
        data["next_contracts"] = None if data["next_contracts"] is None else tuple(ActionMasks(**m) for m in data["next_contracts"])
        data["proposals"] = tuple(tuple(tuple(ActionProposal(**p) for p in step) for step in row) for row in data["proposals"])
        result = cls(**data)
        result.validate()
        return result


class SequenceReplay:
    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self.sequences: list[ReplaySequence] = []
        self.cursor = 0
        self.prefix_cache: dict[tuple[int, int], ActorTensorBatch] = {}
        self.credit_records: dict[int, list[dict]] = {}
        # Independent replay stream; epsilon stream 120 remains separate.
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 120, 1]))

    def add(self, sequence: ReplaySequence):
        sequence.validate()
        owned = ReplaySequence.from_state(sequence.to_state())
        if len(self.sequences) < self.capacity:
            self.sequences.append(owned)
        else:
            outgoing = self.sequences[self.cursor]
            self.prefix_cache[outgoing.episode, outgoing.first_slot] = outgoing.actor
            self.sequences[self.cursor] = owned
        resident = {seq.episode for seq in self.sequences}
        self.prefix_cache = {key: actor for key, actor in self.prefix_cache.items() if key[0] in resident}
        self.credit_records = {key: value for key,value in self.credit_records.items() if key in resident}
        self.cursor = (self.cursor + 1) % self.capacity

    def add_credit_record(self,episode,record):
        records=self.credit_records.setdefault(episode,[])
        if record["slot"]!=len(records):raise ValueError("task credit records must be contiguous episode slots")
        records.append(copy.deepcopy(record))

    def prefix_actors(self, sequence):
        """Retain evicted inputs only while their episode still has replay chunks."""
        actors = {slot: actor for (episode, slot), actor in self.prefix_cache.items()
                  if episode == sequence.episode and slot < sequence.first_slot}
        for item in self.sequences:
            if item.episode == sequence.episode and item.first_slot < sequence.first_slot:
                if item.first_slot in actors:
                    raise ValueError("duplicated recurrent prefix chunk")
                actors[item.first_slot] = item.actor
        covered = 0
        result = []
        for slot, actor in sorted(actors.items()):
            if slot != covered:
                raise ValueError("missing recurrent episode prefix chunk")
            if bool(actor.episode_starts[:, 0].all()) != (slot == 0) or actor.episode_starts[:, 1:].any():
                raise ValueError("recurrent prefix reset differs from its episode offset")
            result.append(actor)
            covered += actor.self_features.shape[1]
        if covered != sequence.first_slot:
            raise ValueError("incomplete recurrent episode prefix")
        return tuple(result)

    def future_sequences(self, sequence):
        """Return only collected resident suffixes; reject gaps or identities."""
        chunks = sorted((item for item in self.sequences
            if item.episode == sequence.episode and item.first_slot >= sequence.first_slot),
            key=lambda item: item.first_slot)
        if not chunks or chunks[0].first_slot != sequence.first_slot:
            raise ValueError("sampled chunk is absent from its replay episode")
        # The sampled replacement carries reconstructed current-weight hidden.
        return (sequence, *chunks[1:])

    def sample(self, count: int):
        if count <= 0 or count > len(self.sequences):
            raise ValueError("replay cannot sample this batch without replacement")
        return tuple(self.sequences[int(i)] for i in self.rng.choice(len(self.sequences), count, replace=False))

    def to_state(self):
        return {"capacity": self.capacity, "cursor": self.cursor,
                "rng": copy.deepcopy(self.rng.bit_generator.state),
                "sequences": [seq.to_state() for seq in self.sequences],
                "credit_records": copy.deepcopy(self.credit_records),
                "prefix_cache": [{"episode": episode, "first_slot": slot, "actor": actor_state(actor)}
                    for (episode, slot), actor in sorted(self.prefix_cache.items())]}

    @classmethod
    def from_state(cls, state):
        result = cls(state["capacity"], 0)
        result.sequences = [ReplaySequence.from_state(item) for item in state["sequences"]]
        result.cursor = state["cursor"]
        result.rng.bit_generator.state = copy.deepcopy(state["rng"])
        result.credit_records=copy.deepcopy(state.get("credit_records",{}))
        for item in state.get("prefix_cache", []):
            key = (item["episode"], item["first_slot"])
            if key in result.prefix_cache or key[0] not in {seq.episode for seq in result.sequences}:
                raise ValueError("invalid recurrent prefix cache identity")
            result.prefix_cache[key] = ActorTensorBatch(**copy.deepcopy(item["actor"]))
        if len(result.sequences) > result.capacity or not 0 <= result.cursor < result.capacity:
            raise ValueError("invalid replay checkpoint geometry")
        return result


class QMIXLearner:
    def __init__(self, config: RunConfig, device="cpu"):
        self.config = config
        self.online = FactorizedQMIX(config).to(device)
        self.target = copy.deepcopy(self.online).requires_grad_(False).eval()
        parameters=[p for p in self.online.parameters() if p.requires_grad]
        if config.method_id=="c_feature_credit_qmix":
            prefixes=tuple("agent.backbone."+prefix for prefix in self.online.agent.backbone.route_parameter_prefixes)
            route=[p for name,p in self.online.named_parameters() if p.requires_grad and name.startswith(prefixes)]
            route_ids={id(p) for p in route}
            parameters=[{"params":[p for p in parameters if id(p) not in route_ids]},
                        {"params":route,"lr":config.training.qmix.learning_rate*config.training.mappo.route_head_lr_scale}]
        self.optimizer = torch.optim.Adam(parameters, lr=config.training.qmix.learning_rate)
        self.updates = 0
        self.device = torch.device(device)

    def update(self, sequences: tuple[ReplaySequence, ...], *, replay=None) -> dict:
        if not sequences:
            raise ValueError("QMIX update requires a sequence minibatch")
        sequences, context_slots = reconstruct_replay_sequences(self.online, self.target, sequences, replay)
        enhanced=self.config.method_id=="c_feature_credit_qmix"
        role_targets=None
        if enhanced:
            from .c_feature_qmix_credit import credit_return_targets
            components,return_stats=credit_return_targets(self.online,self.target,sequences,replay,self.device,self.config.training.qmix.gamma)
            targets=tuple(item[0] for item in components);role_targets=tuple(item[1] for item in components)
        else:
            targets, return_stats = replay_return_targets(self.online, self.target, sequences,
                self.device, self.config.training.qmix.gamma, replay=replay)
        total_count = sum(seq.length for seq in sequences)
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        mean_abs_td = 0.0
        q_taken_mean = target_mean = 0.0
        credit_stats={}
        for sample_index,(seq, target) in enumerate(zip(sequences, targets)):
            seq.validate()
            actor = seq.actor.to(self.device)
            output = self.online.agent.actions(actor, seq.hidden.to(self.device), seq.contracts, proposals=seq.proposals)
            if not torch.equal(output.active.cpu(), seq.active):
                raise ValueError("replay active branches differ from the saved contract")
            prediction = self.online.mixer(output.utilities, seq.state.to(self.device))
            loss = masked_half_mse(prediction, target, torch.ones_like(prediction, dtype=torch.bool))
            extra_stats={}
            if enhanced:
                from .c_feature_qmix_credit import auxiliary_losses
                auxiliary,extra_stats=auxiliary_losses(self.online,self.target,seq,output,replay,
                    self.config.training.qmix.gamma,role_targets[sample_index],self.device)
                extra_stats["team_td_loss"]=float(loss.detach())
                loss=loss+auxiliary
            # Exact real-transition mean, including short terminal chunks;
            # sequential backward bounds GPU memory without changing the batch.
            weight = seq.length / total_count
            (loss * weight).backward()
            total_loss += float(loss.detach()) * weight
            mean_abs_td += float((prediction.detach() - target).abs().mean()) * weight
            q_taken_mean += float(prediction.detach().mean()) * weight
            target_mean += float(target.mean()) * weight
            for key,value in extra_stats.items():
                credit_stats[key]=credit_stats.get(key,0)+(value if key.endswith("samples") else value*weight)
        grad = torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        copied = self.updates % self.config.training.qmix.target_update_interval_updates == 0
        if copied:
            self.target.load_state_dict(self.online.state_dict())
        return {"update": self.updates, "loss": total_loss, "mean_abs_td": mean_abs_td,
                "gradient_norm": float(grad), "real_transitions": total_count,
                "target_updated": copied, "q_taken_mean": q_taken_mean,
                "td_target_mean": target_mean, "recurrent_context_slots": context_slots,
                **return_stats,**credit_stats}

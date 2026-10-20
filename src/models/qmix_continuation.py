"""Explicit full-state QMIX budget extension without changing its learning code.

A final checkpoint at a whole-episode boundary contains all replay and optimizer
state; no old steps are replayed. Only the current learning implementation and
checkpoint schema may be continued; old input-processing versions are rejected.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from ..config import _construct_run_config
from .factorized_gat_qmix import ACTOR_EDGE_TRANSFORM, CPU_UTILITY_CONTRACT, EXPLORATION_CONTRACT, METHOD_ID,QMIX_METHOD_IDS
from .qmix_replay import SequenceReplay
from .qmix_trainer import CHECKPOINT_SCHEMA, QMIXTrainer, implementation_fingerprints,validate_checkpoint_contract
from .qmix_targets import LEARNING_TARGET_CONTRACT
from .qmix_recurrent import RECURRENT_CONTEXT_CONTRACT


CONTINUITY_KEYS = ("online", "target", "optimizer", "replay", "epsilon_rng", "steps",
                   "next_episode", "update_count", "episodes", "updates", "credit_collection_state")


def continuation_fingerprint():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_digest(value):
    """Device-independent, exact content hash for nested restricted states."""
    digest = hashlib.sha256()

    def feed(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor" + str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(b"array" + item.dtype.str.encode() + str(item.shape).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda x: (type(x).__name__, repr(x))):
                feed(key)
                feed(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + str(len(item)).encode())
            for child in item:
                feed(child)
        else:
            digest.update(type(item).__name__.encode() + repr(item).encode() + b";")

    feed(value)
    return digest.hexdigest()


def source_config(data):
    config = _construct_run_config({k: v for k, v in data["config_snapshot"].items() if k != "_metadata"})
    config.validate()
    if config.config_hash != data["config_hash"]:
        raise ValueError("QMIX source configuration hash is inconsistent")
    return config


def load_continuation_checkpoint(path):
    data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if data.get("schema") != CHECKPOINT_SCHEMA or data.get("method_id") not in QMIX_METHOD_IDS:
        raise ValueError("continuation requires a current QMIX checkpoint schema")
    if data.get("implementation_source_sha256") != implementation_fingerprints():
        raise ValueError("QMIX core implementation differs from the source checkpoint")
    validate_checkpoint_contract(data)
    provenance = data.get("budget_extension")
    if provenance is not None:
        if (provenance.get("schema") != "qmix_budget_extension.v1"
                or provenance.get("continuation_source_sha256") != continuation_fingerprint()
                or not provenance.get("ancestors")):
            raise ValueError("QMIX continuation implementation/provenance differs")
    config = source_config(data)
    if provenance is not None:
        previous = None
        for ancestor in provenance["ancestors"]:
            ancestor_path = Path(ancestor["path"])
            if not ancestor_path.is_file() or file_sha256(ancestor_path) != ancestor["sha256"]:
                raise ValueError("QMIX ancestor checkpoint is missing or its identity changed")
            ancestor_config = source_config(ancestor)
            if ancestor["steps"] != ancestor_config.training.qmix.max_training_environment_steps:
                raise ValueError("QMIX ancestor is not its declared budget end")
            if previous is not None:
                validate_budget_extension(previous, ancestor_config)
            previous = ancestor_config
        validate_budget_extension(previous, config)
    q, horizon = config.training.qmix, config.environment.episode_horizon
    if (type(data["steps"]) is not int or not 0 < data["steps"] <= q.max_training_environment_steps
            or data["steps"] != data["next_episode"] * horizon):
        raise ValueError("QMIX source is not a complete episode boundary")
    expected_kind = "final" if data["steps"] == q.max_training_environment_steps else "periodic"
    if data["kind"] != expected_kind or data["steps"] % q.checkpoint_interval_steps:
        raise ValueError("QMIX source checkpoint kind/interval is inconsistent")
    chunks = data["next_episode"] * math.ceil(horizon / q.recurrent_sequence_length_slots)
    if (data["update_count"] != max(0, chunks - q.sequence_batch_size + 1)
            or len(data["episodes"]) != data["next_episode"]
            or len(data["updates"]) != data["update_count"]
            or len(data["replay"]["sequences"]) != min(chunks, q.replay_capacity_sequences)
            or data["replay"]["cursor"] != chunks % q.replay_capacity_sequences):
        raise ValueError("QMIX source replay/update/episode counters are inconsistent")
    return data


def validate_budget_extension(source, target):
    """Only QMIX episode/step caps and three independent output roots may change."""
    source.validate()
    target.validate()

    def scientific(config):
        fields = copy.deepcopy(config.resolved_dict())
        for name in ("max_training_episodes", "max_training_environment_steps"):
            fields["training"]["qmix"].pop(name)
        for name in ("logs_dir", "plots_dir", "dashboard_logs_dir"):
            fields["output"].pop(name)
        return fields

    if scientific(source) != scientific(target):
        raise ValueError("QMIX budget extension cannot change scientific settings, seed or device")
    old, new = source.training.qmix, target.training.qmix
    if (new.max_training_environment_steps <= old.max_training_environment_steps
            or new.max_training_episodes <= old.max_training_episodes
            or new.max_training_episodes * target.environment.episode_horizon != new.max_training_environment_steps):
        raise ValueError("QMIX extension must increase both whole-episode budget caps")
    previous = [Path(getattr(source.output, key)).resolve() for key in ("logs_dir", "plots_dir", "dashboard_logs_dir")]
    current = [Path(getattr(target.output, key)).resolve() for key in ("logs_dir", "plots_dir", "dashboard_logs_dir")]
    if len(set(current)) != 3:
        raise ValueError("QMIX extension output roots must be distinct")
    for directory in current:
        if any(part.lower() in {".agents", "knowledge", ".git"} for part in directory.parts):
            raise ValueError("QMIX extension cannot write into a protected directory")
        if any(directory == old_path or directory.is_relative_to(old_path) or old_path.is_relative_to(directory)
               for old_path in previous):
            raise ValueError("QMIX extension output roots must not overlap previous outputs")


class QMIXContinuationTrainer(QMIXTrainer):
    def __init__(self, config, *, device=None):
        super().__init__(config, device=device)
        self.budget_extension = None
        self.restore_verification = None

    def payload(self, kind):
        result = super().payload(kind)
        if self.budget_extension is not None:
            result["budget_extension"] = copy.deepcopy(self.budget_extension)
        return result

    @classmethod
    def from_checkpoint(cls, config, path, *, extend=False, device=None):
        data = load_continuation_checkpoint(path)
        source = source_config(data)
        if extend:
            if data["kind"] != "final":
                raise ValueError("QMIX budget extension requires the completed source final checkpoint")
            validate_budget_extension(source, config)
            for ancestor in data.get("budget_extension", {}).get("ancestors", []):
                validate_budget_extension(source_config(ancestor), config)
        elif data["kind"] != "periodic" or data["config_hash"] != config.config_hash:
            raise ValueError("exact QMIX resume requires a periodic checkpoint of the identical configuration")
        requested_device = str(torch.device(device or config.training.mappo.training_device))
        if data["runtime"] != {"torch": str(torch.__version__), "device": requested_device}:
            raise ValueError("QMIX full-state continuation requires the same torch runtime and device")
        before = state_digest({key: data[key] for key in CONTINUITY_KEYS})
        trainer = cls(config, device=device)
        trainer.learner.online.load_state_dict(data["online"], strict=True)
        trainer.learner.target.load_state_dict(data["target"], strict=True)
        trainer.learner.optimizer.load_state_dict(data["optimizer"])
        trainer.replay = SequenceReplay.from_state(data["replay"])
        trainer.epsilon_rng.bit_generator.state = copy.deepcopy(data["epsilon_rng"])
        trainer.steps, trainer.next_episode = data["steps"], data["next_episode"]
        trainer.learner.updates = data["update_count"]
        trainer.episodes, trainer.updates = copy.deepcopy(data["episodes"]), copy.deepcopy(data["updates"])
        trainer.credit_collection_state=copy.deepcopy(data["credit_collection_state"])
        provenance = copy.deepcopy(data.get("budget_extension"))
        if extend:
            ancestors = [] if provenance is None else provenance["ancestors"]
            ancestors.append({"path": str(Path(path).resolve()), "sha256": file_sha256(path),
                              "steps": data["steps"], "config_snapshot": copy.deepcopy(data["config_snapshot"]),
                              "config_hash": data["config_hash"]})
            provenance = {"schema": "qmix_budget_extension.v1",
                          "continuation_source_sha256": continuation_fingerprint(), "ancestors": ancestors,
                          "restored_state_sha256": before, "restored_environment_steps": data["steps"],
                          "restored_optimizer_updates": data["update_count"], "replayed_environment_steps": 0}
        trainer.budget_extension = provenance
        restored = trainer.payload("periodic")
        after = state_digest({key: restored[key] for key in CONTINUITY_KEYS})
        if before != after:
            raise ValueError("QMIX restored model/optimizer/replay/RNG/history state differs from source")
        trainer.restore_verification = {"full_state_sha256": after, "source_steps": trainer.steps,
            "source_optimizer_updates": trainer.learner.updates, "environment_steps_executed": 0,
            "optimizer_updates_executed": 0, "target_steps": config.training.qmix.max_training_environment_steps,
            "replayed_environment_steps": 0}
        return trainer

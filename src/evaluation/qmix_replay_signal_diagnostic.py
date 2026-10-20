"""Bounded current-prefix TD/gradient probes without optimizer or learning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..artifacts import _filesystem_path, atomic_write_text_group
from ..config import load_run_config
from ..models.qmix_continuation import load_continuation_checkpoint, state_digest
from ..models.qmix_replay import SequenceReplay
from .actor_loader import checkpoint_sha256
from .qmix_limited_diagnostic import gradient_probe, load_system


def run(config_path, output_path, progress=print):
    config = load_run_config(config_path)
    output_path = Path(output_path)
    if _filesystem_path(output_path).exists():
        raise FileExistsError("refusing to overwrite replay diagnostic evidence")
    final = Path(config.artifact_paths()["final_checkpoint"])
    final_hash = checkpoint_sha256(final)
    payload = load_continuation_checkpoint(final)
    if payload["config_hash"] != config.config_hash or config.seed != 42:
        raise ValueError("diagnostic requires the declared seed42 source configuration")
    replay = SequenceReplay.from_state(payload["replay"])
    boundary = sum(sequence.episode < 8 for sequence in replay.sequences)
    if payload["steps"] != 8000 or boundary != 128 or len(replay.sequences) != 256:
        raise ValueError("this bounded diagnostic requires the complete seed42 8k replay")
    rng = np.random.default_rng(20261017)
    strata = {"first_4k": sorted(rng.choice(boundary, 8, replace=False).tolist()),
              "last_4k": sorted(rng.choice(np.arange(boundary, len(replay.sequences)), 8, replace=False).tolist())}
    records = []
    for steps in (4000, 8000):
        checkpoint = final if steps == 8000 else final.with_name("step_004000.pt")
        checkpoint_hash = checkpoint_sha256(checkpoint)
        saved = payload if steps == 8000 else load_continuation_checkpoint(checkpoint)
        system, target = load_system(config, saved)
        before, target_before = state_digest(system.state_dict()), state_digest(target.state_dict())
        for name, indices in strata.items():
            probes = [gradient_probe(system, target, replay.sequences[index], config.training.qmix.gamma, replay=replay)
                      for index in indices]
            records.append({"training_steps": steps, "checkpoint_sha256": checkpoint_hash,
                            "stratum": name, "indices": indices, "probes": probes})
            progress(json.dumps({"stage": "current_prefix_backward_only", "training_steps": steps,
                                 "stratum": name, "chunks": len(probes), "optimizer_updates": 0}), flush=True)
        if before != state_digest(system.state_dict()) or target_before != state_digest(target.state_dict()):
            raise ValueError("backward-only probe changed model weights")
        if checkpoint_hash != checkpoint_sha256(checkpoint):
            raise ValueError("probe changed source checkpoint")
    if final_hash != checkpoint_sha256(final):
        raise ValueError("replay source checkpoint changed")
    result = {"schema": "qmix_current_prefix_signal_diagnostic.v1", "probes": records,
              "chunk_model_pairs": 32, "sampled_chunks": 16, "sampling_seed": 20261017,
              "optimizer_created": False, "optimizer_updates": 0, "parameters_checkpoints_unchanged": True,
              "claim_boundary": "Fixed sampled real chunks from the 8k replay, evaluated with both checkpoint models and their targets; current-weight full prefix reconstruction. Includes late off-policy data for the 4k model. Finite nonzero sampled gradients do not establish good credit assignment or optimal learning."}
    atomic_write_text_group([(output_path, json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))],
                           group_name="QMIX replay signal diagnostic")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.config, args.output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

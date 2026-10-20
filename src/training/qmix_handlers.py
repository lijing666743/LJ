"""Explicit registry adapters for independent QMIX training and evaluation."""
from __future__ import annotations

from ..artifacts import preflight_formal_training_artifacts
from ..execution import validate_execution_context


def interactive_qmix(mode, *, input_fn, output_fn):
    from ..cli import _prompt_optional, _prompt_int, _prompt_choice, _dispatch
    from ..config import ConfigError, load_run_config
    from ..execution import ExecutionContext
    path = _prompt_optional(input_fn, output_fn, "QMIX config path (required; source: project-doc): ")
    if not path:
        output_fn("Configuration error: an explicit QMIX config is required")
        return 2
    try:
        base = load_run_config(path)
        if base.method_id not in {"factorized_action_gat_qmix","c_feature_credit_qmix"}:
            raise ConfigError("QMIX menu requires a QMIX configuration")
        seed = _prompt_int(input_fn, output_fn,
            f"Master seed [integer >=0] (default: {base.seed}, source: config): ", base.seed, minimum=0)
        device = _prompt_choice(input_fn, output_fn,
            f"Device [cpu/cuda] (default: {base.training.mappo.training_device}, source: config): ",
            ("cpu", "cuda"), base.training.mappo.training_device)
        config = load_run_config(path, interactive_overrides={
            "mode": mode, "seed": seed, "training.mappo.training_device": device})
        context = None
        if mode == "evaluation":
            checkpoint = _prompt_optional(input_fn, output_fn, "QMIX final checkpoint path (required): ")
            context = ExecutionContext.from_evaluation_path(checkpoint or None, device)
        elif mode == "rl":
            checkpoint = _prompt_optional(input_fn, output_fn,
                "QMIX resume checkpoint [optional; empty=fresh initialization]: ")
            if checkpoint:
                restore_mode = _prompt_choice(input_fn, output_fn,
                    "Restore mode [exact/extend] (default: exact): ", ("exact", "extend"), "exact")
                context = ExecutionContext(resume_from=checkpoint, extend_training_budget=restore_mode == "extend")
        validate_execution_context(config, context)
        return _dispatch(config, execution_context=context, output_fn=output_fn)
    except (ValueError, OSError) as exc:
        output_fn(f"Configuration error: {exc}")
        return 2


def qmix_training_handler(config, *, execution_context=None):
    from ..registry import RunResult
    from ..models.qmix_trainer import QMIXTrainer, write_training_artifacts
    context = validate_execution_context(config, execution_context)
    try:
        preflight_formal_training_artifacts(config.artifact_paths(),
            resume=context.resume_from is not None and not context.extend_training_budget)
        if context.resume_from is not None:
            from ..models.qmix_continuation import QMIXContinuationTrainer
            trainer = QMIXContinuationTrainer.from_checkpoint(config, context.resume_from,
                extend=context.extend_training_budget)
        else:
            trainer = QMIXTrainer(config)
        if context.check_resume:
            import json
            return RunResult("completed", config.run_id, config.mode, config.method_id,
                "QMIX resume_check_passed: " + json.dumps(trainer.restore_verification, sort_keys=True))
        trainer.train(write_checkpoints=True)
        artifacts = write_training_artifacts(trainer)
    except Exception as exc:
        return RunResult("failed", config.run_id, config.mode, config.method_id,
                         f"QMIX training did not complete: {exc}")
    return RunResult("completed", config.run_id, config.mode, config.method_id,
                     f"QMIX training completed: transitions={trainer.steps}, optimizer_updates={trainer.learner.updates}; diagnostics only",
                     (*artifacts, config.artifact_paths()["final_checkpoint"]))


def qmix_evaluation_handler(config, *, execution_context=None):
    from ..registry import RunResult
    from ..evaluation.qmix_runner import QMIXEvaluationRunner
    context = validate_execution_context(config, execution_context)
    try:
        result = QMIXEvaluationRunner(config, context.evaluate_from, device=context.evaluation_device).run()
    except Exception as exc:
        return RunResult("failed", config.run_id, config.mode, config.method_id,
                         f"QMIX frozen evaluation did not complete: {exc}")
    return RunResult("completed", result["manifest"]["evaluation_run_id"], config.mode, config.method_id,
                     f"QMIX frozen evaluation completed: episodes={len(result['episodes'])}, network_updates=0",
                     result["artifacts"])

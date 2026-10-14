"""Offline route-option alignment metrics for diagnostic JSONL sidecars.

The reconstructed delay is an actor-safe diagnostic proxy, not an exact
counterfactual optimum.  This module never participates in policy execution.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, Sequence

from ..models.ca_gat_mappo_route_telemetry import (
    reconstruct_route_diagnostic_utility,
)


DEFAULT_ROUTE_ALIGNMENT_WINDOWS: Mapping[str, tuple[int | None, int | None]] = {
    "overall": (None, None),
    "early_0_10k": (0, 10_000),
    "middle_10k_20k": (10_000, 20_000),
    "late_20k_32k": (20_000, 32_000),
    "final_20_percent_25_6k_32k": (25_600, 32_000),
}


class RouteOptionAlignmentError(ValueError):
    """Raised when a route diagnostic artifact cannot be analyzed safely."""


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Sequence[float]) -> float | None:
    return fmean(values) if values else None


def _scale(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": _mean(values),
        "std": pstdev(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "range": max(values) - min(values) if values else None,
    }


def _in_window(step: int, bounds: tuple[int | None, int | None]) -> bool:
    lower, upper = bounds
    return (lower is None or step >= lower) and (upper is None or step < upper)


def _window_summary(
    header: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    *,
    tie_tolerance: float,
) -> dict[str, Any]:
    decisive_pairs = 0
    agreeing_pairs = 0
    utility_ties = 0
    probability_ties = 0
    top1_eligible = 0
    top1_matches = 0
    probability_advantages: list[float] = []
    source1_candidate2_minus0: list[float] = []
    source1_candidate2_minus3: list[float] = []
    local_probabilities: list[float] = []
    legal_remote_masses: list[float] = []
    local_logits: list[float] = []
    defer_logits: list[float] = []
    legal_remote_logits: list[float] = []
    u1_to_u2_legal = 0
    u1_to_u2_selected = 0
    reconstructed_samples = 0

    for sample in samples:
        try:
            utility = reconstruct_route_diagnostic_utility(sample, header)
        except (KeyError, TypeError, ValueError):
            continue
        reconstructed_samples += 1
        policy = sample.get("policy", {})
        if not isinstance(policy, Mapping):
            continue
        entries = policy.get("route_entries", ())
        if not isinstance(entries, Iterable):
            entries = ()
        for entry in entries:
            if isinstance(entry, Mapping):
                action = entry.get("action")
                logit = _finite_float(entry.get("raw_logit"))
                if logit is None:
                    continue
                if action == "local":
                    local_logits.append(logit)
                elif action == "defer":
                    defer_logits.append(logit)
                elif (
                    isinstance(action, int)
                    and not isinstance(action, bool)
                    and bool(entry.get("legal"))
                ):
                    legal_remote_logits.append(logit)

        local_probability = _finite_float(policy.get("local_probability"))
        if local_probability is not None:
            local_probabilities.append(local_probability)

        probabilities: dict[int, float] = {}
        for item in policy.get("remote_probability_by_candidate", ()):
            if not isinstance(item, Mapping) or not bool(item.get("legal")):
                continue
            candidate = item.get("candidate_uav")
            probability = _finite_float(item.get("probability"))
            if (
                isinstance(candidate, int)
                and not isinstance(candidate, bool)
                and probability is not None
            ):
                probabilities[candidate] = probability
        if probabilities:
            legal_remote_masses.append(math.fsum(probabilities.values()))

        times: dict[int, float] = {}
        for item in utility.get("remote_by_candidate", ()):
            if not isinstance(item, Mapping) or item.get("status") != "valid":
                continue
            candidate = item.get("candidate_uav")
            total_time = _finite_float(item.get("T_remote_total_est_s"))
            if (
                isinstance(candidate, int)
                and not isinstance(candidate, bool)
                and total_time is not None
                and candidate in probabilities
            ):
                times[candidate] = total_time

        candidates = sorted(times)
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1 :]:
                time_delta = times[left] - times[right]
                probability_delta = probabilities[left] - probabilities[right]
                if abs(time_delta) <= tie_tolerance:
                    utility_ties += 1
                elif abs(probability_delta) <= tie_tolerance:
                    probability_ties += 1
                else:
                    decisive_pairs += 1
                    agreeing_pairs += int(time_delta * probability_delta < 0.0)

        if candidates:
            minimum_time = min(times[candidate] for candidate in candidates)
            maximum_probability = max(probabilities[candidate] for candidate in candidates)
            best_times = [
                candidate
                for candidate in candidates
                if abs(times[candidate] - minimum_time) <= tie_tolerance
            ]
            best_probabilities = [
                candidate
                for candidate in candidates
                if abs(probabilities[candidate] - maximum_probability)
                <= tie_tolerance
            ]
            if len(best_times) == 1 and len(best_probabilities) == 1:
                top1_eligible += 1
                top1_matches += int(best_times[0] == best_probabilities[0])
            if len(best_times) == 1 and len(candidates) > 1:
                best = best_times[0]
                others = [
                    probabilities[candidate]
                    for candidate in candidates
                    if candidate != best
                ]
                probability_advantages.append(probabilities[best] - fmean(others))

        source_uav = sample.get("source_uav")
        if source_uav == 1 and {0, 2}.issubset(times):
            if times[2] < times[0] - tie_tolerance:
                source1_candidate2_minus0.append(probabilities[2] - probabilities[0])
        if source_uav == 1 and {2, 3}.issubset(times):
            if times[2] < times[3] - tie_tolerance:
                source1_candidate2_minus3.append(probabilities[2] - probabilities[3])
        if source_uav == 1 and 2 in probabilities:
            u1_to_u2_legal += 1
            u1_to_u2_selected += int(policy.get("chosen_remote") == 2)

    return {
        "sample_count": len(samples),
        "reconstructed_sample_count": reconstructed_samples,
        "destination_ranking_accuracy": {
            "agreeing_pair_count": agreeing_pairs,
            "decisive_pair_count": decisive_pairs,
            "utility_tie_count": utility_ties,
            "probability_tie_count": probability_ties,
            "accuracy": (
                agreeing_pairs / decisive_pairs if decisive_pairs else None
            ),
        },
        "top1_alignment": {
            "match_count": top1_matches,
            "eligible_sample_count": top1_eligible,
            "accuracy": top1_matches / top1_eligible if top1_eligible else None,
        },
        "best_candidate_probability_advantage": {
            "count": len(probability_advantages),
            "mean": _mean(probability_advantages),
        },
        "source_uav_1_candidate_2_proxy_better": {
            "p2_minus_p0_eligible_sample_count": len(source1_candidate2_minus0),
            "mean_p2_minus_p0": _mean(source1_candidate2_minus0),
            "p2_minus_p3_eligible_sample_count": len(source1_candidate2_minus3),
            "mean_p2_minus_p3": _mean(source1_candidate2_minus3),
        },
        "policy": {
            "mean_local_probability": _mean(local_probabilities),
            "mean_legal_remote_probability_mass": _mean(legal_remote_masses),
            "uav1_to_uav2_legal_count": u1_to_u2_legal,
            "uav1_to_uav2_selected_count": u1_to_u2_selected,
            "uav1_to_uav2_selection_rate_when_legal": (
                u1_to_u2_selected / u1_to_u2_legal if u1_to_u2_legal else None
            ),
        },
        "raw_logit_scale": {
            "local": _scale(local_logits),
            "defer": _scale(defer_logits),
            "legal_remote": _scale(legal_remote_logits),
        },
    }


def analyze_route_option_alignment(
    header: Mapping[str, Any],
    samples: Iterable[Mapping[str, Any]],
    *,
    tie_tolerance: float = 1.0e-12,
    windows: Mapping[str, tuple[int | None, int | None]] | None = None,
) -> dict[str, Any]:
    """Compute DRA and policy/logit diagnostics over fixed training windows."""

    if not isinstance(header, Mapping) or header.get("record_type") != "schema":
        raise RouteOptionAlignmentError("header must be a route diagnostic schema")
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise RouteOptionAlignmentError("tie_tolerance must be finite and non-negative")
    copied = tuple(
        sample
        for sample in samples
        if isinstance(sample, Mapping)
        and sample.get("record_type") == "route_diagnostic_sample"
    )
    selected_windows = DEFAULT_ROUTE_ALIGNMENT_WINDOWS if windows is None else windows
    summaries: dict[str, Any] = {}
    for name, bounds in selected_windows.items():
        if (
            not isinstance(name, str)
            or not isinstance(bounds, tuple)
            or len(bounds) != 2
        ):
            raise RouteOptionAlignmentError("window definitions must be named pairs")
        window_samples = tuple(
            sample
            for sample in copied
            if isinstance(sample.get("global_environment_step"), int)
            and not isinstance(sample.get("global_environment_step"), bool)
            and _in_window(int(sample["global_environment_step"]), bounds)
        )
        summaries[name] = _window_summary(
            header,
            window_samples,
            tie_tolerance=tie_tolerance,
        )
    return {
        "analysis": "route_option_alignment_v1",
        "utility_interpretation": (
            "actor-safe reconstructed delay proxy; not an exact counterfactual optimum"
        ),
        "utility_proxy_fields": {
            "task": [
                "remaining_bits",
                "remaining_cycles",
                "remaining_deadline_slots",
            ],
            "source": [
                "max_cpu_frequency_hz",
                "cpu_coefficient",
                "max_transmit_power_w",
                "residual_energy_j",
                "local_cpu_backlog.remaining_cycles",
            ],
            "candidate": [
                "legal_mask",
                "helper_public CPU/energy/backlog",
                "source_tx_queue.remaining_bits",
                "link_history.historical_quality_by_ru and validity",
            ],
            "future_information_used": False,
        },
        "tie_tolerance": tie_tolerance,
        "windows": summaries,
    }


def load_route_diagnostic_samples(
    path: str | Path,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Load one schema header and its samples from a diagnostic JSONL file."""

    artifact = Path(path)
    header: dict[str, Any] | None = None
    samples: list[dict[str, Any]] = []
    with artifact.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RouteOptionAlignmentError(
                    f"invalid JSON at line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise RouteOptionAlignmentError(
                    f"line {line_number} is not a JSON object"
                )
            if record.get("record_type") == "schema":
                if header is not None:
                    raise RouteOptionAlignmentError("artifact contains multiple headers")
                header = record
            elif record.get("record_type") == "route_diagnostic_sample":
                samples.append(record)
    if header is None:
        raise RouteOptionAlignmentError("artifact has no schema header")
    return header, tuple(samples)


__all__ = [
    "DEFAULT_ROUTE_ALIGNMENT_WINDOWS",
    "RouteOptionAlignmentError",
    "analyze_route_option_alignment",
    "load_route_diagnostic_samples",
]

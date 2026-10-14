"""Diagnostic Package 3.1 production-path Return/TD distribution gates."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from src.models.ca_gat_mappo_route_telemetry import (
    ROUTE_DISTRIBUTION_GROUPS,
    RouteTelemetry,
    collect_route_telemetry,
    summarize_distribution,
)
from src.training_artifacts import (
    _base_record,
    _csv_text,
    _json_lines,
    read_training_metrics_csv_text,
    write_cagat_mappo_training_artifacts,
)
from tests.test_ca_gat_mappo_route_telemetry import (
    active_route_fixture,
    evaluate_route_steps,
)
from tests.test_ca_gat_mappo_training_artifacts import training_result


_DISTRIBUTION_SUFFIXES = (
    "valid_sample_count",
    "mean",
    "std",
    "median",
    "p25",
    "p75",
    "positive_fraction",
    "negative_fraction",
)


def _collect_production_telemetry(
    advantages: tuple[float, ...],
    returns: tuple[float, ...],
    td_residuals: tuple[float, ...],
    route_kinds: tuple[str, ...],
    *,
    use_reset_observations: bool = False,
):
    config, environment, reset, observations = active_route_fixture()
    if use_reset_observations:
        observations = tuple(reset.observations)
    observation_steps = tuple(observations for _ in route_kinds)
    kinds = tuple(
        tuple(kind for _ in observations)
        for kind in route_kinds
    )
    actor, _, masks, proposals, policy = evaluate_route_steps(
        config,
        environment,
        observation_steps,
        kinds,
    )
    telemetry = collect_route_telemetry(
        policy=policy,
        action_mask_batch=masks,
        proposals=proposals,
        advantage=torch.tensor([advantages], dtype=torch.float32),
        return_target=torch.tensor([returns], dtype=torch.float32),
        sequence_valid_mask=torch.ones(
            (1, len(route_kinds)),
            dtype=torch.bool,
        ),
        route_head=actor.action_heads["route"],
        td_residual=torch.tensor([td_residuals], dtype=torch.float32),
    )
    return config, telemetry


def _telemetry_record(config, telemetry: RouteTelemetry) -> dict[str, object]:
    record = _base_record(config, "insufficient-horizon")
    record.update({
        "record_type": "ppo_epoch",
        "telemetry_scope": "per_ppo_epoch",
        "series_index": 1,
        "route_telemetry_enabled": True,
    })
    telemetry_record = telemetry.record()
    telemetry_record["route_telemetry_schema_version"] = telemetry_record.pop(
        "schema_version"
    )
    record.update(telemetry_record)
    return record


def _assert_distribution(
    test_case: unittest.TestCase,
    record: dict[str, object],
    prefix: str,
    *,
    count: int,
    mean: float,
    median: float,
    p25: float,
    p75: float,
    positive_fraction: float,
    negative_fraction: float,
) -> None:
    expected = {
        "valid_sample_count": count,
        "mean": mean,
        "median": median,
        "p25": p25,
        "p75": p75,
        "positive_fraction": positive_fraction,
        "negative_fraction": negative_fraction,
    }
    for suffix in _DISTRIBUTION_SUFFIXES:
        test_case.assertIn(f"{prefix}_{suffix}", record)
    test_case.assertEqual(record[f"{prefix}_valid_sample_count"], count)
    for suffix in (
        "mean",
        "median",
        "p25",
        "p75",
        "positive_fraction",
        "negative_fraction",
    ):
        test_case.assertAlmostEqual(
            float(record[f"{prefix}_{suffix}"]),
            expected[suffix],
        )


class ReturnAndTDDistributionCompletionTests(unittest.TestCase):
    def test_production_collector_record_has_complete_route_groups(self) -> None:
        _, telemetry = _collect_production_telemetry(
            (-2.0, 0.0, 3.0),
            (-4.0, 0.0, 6.0),
            (-1.0, 0.0, 2.0),
            ("local", "remote", "defer"),
        )
        record = telemetry.record()

        expected_groups = {
            "route_active": (12, 2.0 / 3.0, 0.0, -4.0, 6.0, 1.0 / 3.0, 1.0 / 3.0),
            "local_route": (4, -4.0, -4.0, -4.0, -4.0, 0.0, 1.0),
            "remote_route": (4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "defer_route": (4, 6.0, 6.0, 6.0, 6.0, 1.0, 0.0),
            "legal_remote_local": (4, -4.0, -4.0, -4.0, -4.0, 0.0, 1.0),
            "legal_remote_remote": (4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        }
        for group, values in expected_groups.items():
            _assert_distribution(
                self,
                record,
                f"{group}_return_target",
                count=values[0],
                mean=values[1],
                median=values[2],
                p25=values[3],
                p75=values[4],
                positive_fraction=values[5],
                negative_fraction=values[6],
            )

        expected_td = {
            "route_active": (12, 1.0 / 3.0, 0.0, -1.0, 2.0, 1.0 / 3.0, 1.0 / 3.0),
            "local_route": (4, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0),
            "remote_route": (4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "defer_route": (4, 2.0, 2.0, 2.0, 2.0, 1.0, 0.0),
            "legal_remote_local": (4, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0),
            "legal_remote_remote": (4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        }
        for group, values in expected_td.items():
            _assert_distribution(
                self,
                record,
                f"{group}_td_residual",
                count=values[0],
                mean=values[1],
                median=values[2],
                p25=values[3],
                p75=values[4],
                positive_fraction=values[5],
                negative_fraction=values[6],
            )

    def test_single_finite_sample_uses_complete_distribution_semantics(self) -> None:
        summary = summarize_distribution(
            torch.tensor([7.0, float("nan"), float("inf"), float("-inf")])
        )
        self.assertEqual(summary.count, 1)
        self.assertEqual(summary.mean, 7.0)
        self.assertEqual(summary.std, 0.0)
        self.assertEqual(summary.median, 7.0)
        self.assertEqual(summary.p25, 7.0)
        self.assertEqual(summary.p75, 7.0)
        self.assertEqual(summary.positive_fraction, 1.0)
        self.assertEqual(summary.negative_fraction, 0.0)

    def test_zero_route_active_samples_are_na_in_json_and_blank_in_csv(self) -> None:
        config, telemetry = _collect_production_telemetry(
            (5.0,),
            (9.0,),
            (-3.0,),
            ("idle",),
            use_reset_observations=True,
        )
        self.assertEqual(telemetry.route_branch_active_count, 0)
        record = _telemetry_record(config, telemetry)
        decoded = json.loads(_json_lines([record]))
        csv_row = next(csv.DictReader(io.StringIO(_csv_text([record]))))
        for group in ROUTE_DISTRIBUTION_GROUPS:
            for signal in ("return_target", "td_residual"):
                prefix = f"{group}_{signal}"
                self.assertEqual(decoded[f"{prefix}_valid_sample_count"], 0)
                for suffix in _DISTRIBUTION_SUFFIXES[1:]:
                    self.assertIsNone(decoded[f"{prefix}_{suffix}"])
                    self.assertEqual(csv_row[f"{prefix}_{suffix}"], "")

    def test_real_writer_persists_return_and_td_quantiles_and_fractions(self) -> None:
        config, telemetry = _collect_production_telemetry(
            (-2.0, 0.0, 3.0),
            (-4.0, 0.0, 6.0),
            (-1.0, 0.0, 2.0),
            ("local", "remote", "defer"),
        )
        base_training = training_result()
        first_update = base_training.updates[0]
        epochs = tuple(
            replace(
                epoch,
                route_telemetry=telemetry if index == 0 else epoch.route_telemetry,
            )
            for index, epoch in enumerate(first_update.output.epoch_diagnostics)
        )
        update = replace(
            first_update,
            output=replace(first_update.output, epoch_diagnostics=epochs),
        )
        training = replace(
            base_training,
            updates=(update,) + base_training.updates[1:],
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            isolated = replace(
                config,
                output=replace(
                    config.output,
                    logs_dir=str(root / "logs"),
                    dashboard_logs_dir=str(root / "dashboard_logs"),
                    plots_dir=str(root / "plots"),
                ),
            )
            write_cagat_mappo_training_artifacts(isolated, training)
            paths = isolated.artifact_paths()
            raw_records = [
                json.loads(line)
                for line in Path(paths["raw_metrics"]).read_text(encoding="utf-8").splitlines()
            ]
            raw_epoch = next(
                item
                for item in raw_records
                if item.get("record_type") == "ppo_epoch"
                and item.get("route_telemetry_enabled") is True
            )
            aggregate = json.loads(
                Path(paths["aggregate_metrics"]).read_text(encoding="utf-8")
            )
            aggregate_epoch = aggregate["route_telemetry"]["final_summary"][
                "latest_ppo_epoch"
            ]
            with Path(paths["dashboard_csv"]).open(
                encoding="utf-8", newline=""
            ) as handle:
                csv_epoch = next(
                    row
                    for row in csv.DictReader(handle)
                    if row["record_type"] == "ppo_epoch"
                    and row["route_telemetry_enabled"] == "True"
                )

            fields = (
                "median",
                "p25",
                "p75",
                "positive_fraction",
                "negative_fraction",
            )
            for signal in ("return_target", "td_residual"):
                prefix = f"route_active_{signal}"
                for suffix in fields:
                    name = f"{prefix}_{suffix}"
                    self.assertIn(name, raw_epoch)
                    self.assertIn(name, aggregate_epoch)
                    self.assertIn(name, csv_epoch)
                    self.assertEqual(raw_epoch[name], aggregate_epoch[name])
                    self.assertAlmostEqual(float(csv_epoch[name]), raw_epoch[name])

    def test_schema_v2_csv_remains_readable_with_new_columns_missing(self) -> None:
        legacy = (
            "diagnostics_schema_version,record_type,"
            "route_active_return_target_median\n"
            "2,ppo_epoch,1.0\n"
        )
        row = read_training_metrics_csv_text(legacy)[0]
        self.assertEqual(row["diagnostics_schema_version"], "2")
        self.assertEqual(row["route_active_return_target_median"], "1.0")
        self.assertEqual(row["route_active_return_target_p25"], "")
        self.assertEqual(row["route_active_td_residual_p75"], "")


if __name__ == "__main__":
    unittest.main()

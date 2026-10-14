"""Passive task/TX/CPU observations; no environment or random-stream mutation.

WAIT means no positive service has ever occurred in the current stage;
SERVICE means at least one positive service, including the expiry slot.
CPU queues include local and remote tasks, at slot start. Frequency is in Hz.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


SERVICE_TELEMETRY_SCHEMA = "evaluation_service_telemetry.v1"


def snapshot_tasks(tasks: Iterable[Any]) -> dict[int, dict[str, Any]]:
    """Copy scalar task snapshots without retaining live task references."""
    return {task.task_id: task.snapshot() for task in tasks}


class EvaluationServiceTelemetryCollector:
    """Join existing service events to copied task records after each step."""

    def __init__(
        self, *, evaluation_run_id: str, method_id: str,
        evaluation_seed: int, episode_id: int,
    ) -> None:
        self.identity = {
            "schema_version": SERVICE_TELEMETRY_SCHEMA,
            "evaluation_run_id": evaluation_run_id,
            "method_id": method_id,
            "evaluation_seed": evaluation_seed,
            "episode_id": episode_id,
        }
        self.tasks: dict[int, dict[str, Any]] = {}
        self.cpu_slots: list[dict[str, Any]] = []
        self.tx_slots: list[dict[str, Any]] = []

    def observe_step(
        self, *, before: Mapping[int, Mapping[str, Any]],
        after: Mapping[int, Mapping[str, Any]], step_info: Mapping[str, Any],
    ) -> None:
        slot = int(step_info["slot"])
        for task_id, task in after.items():
            record = self.tasks.setdefault(task_id, {
                **self.identity, "task_id": task_id,
                "source_uav": task["source_uav"],
                "generation_slot": task["arrival_slot"],
                "deadline_slot": task["deadline_slot"],
                "input_bits": task["data_bits"],
                "required_cycles": task["cpu_cycles"],
                "tx_first_service_slot": None, "cpu_first_service_slot": None,
                "done_slot": None, "expired_slot": None,
                "expire_stage": None, "terminal_stage": None,
            })
            record.update(
                destination_uav=task["destination"],
                route_bind_slot=task["binding_slot"],
                tx_complete_slot=(task["cpu_entry_slot"] if
                                  task["destination"] != task["source_uav"] else None),
                remaining_bits=task["remaining_bits"],
                remaining_cycles=task["remaining_cycles"],
            )

        bits_by_task: dict[int, float] = defaultdict(float)
        links = {item["sender_uav"]: item for item in step_info["service"]["links"]}
        for link in links.values():
            for service in link["task_services"]:
                task_id, amount = service["task_id"], float(service["amount"])
                bits_by_task[task_id] += amount
                if amount > 0 and self.tasks[task_id]["tx_first_service_slot"] is None:
                    self.tasks[task_id]["tx_first_service_slot"] = slot
        cpu = {item["executor_uav"]: item for item in step_info["service"]["cpu"]}
        for item in cpu.values():
            task_id = item["task_id"]
            if task_id is not None and item["service_cycles"] > 0:
                if self.tasks[task_id]["cpu_first_service_slot"] is None:
                    self.tasks[task_id]["cpu_first_service_slot"] = slot

        rejections = {item["uav_id"]: item["reason"] for item in step_info["rejection"]}
        proposals = {item["uav_id"]: item for item in step_info["proposal"]}
        debits = {item["uav_id"]: item for item in step_info["energy"]["debits"]}
        for executed in step_info["executed"]:
            uav = executed["uav_id"]
            cpu_tasks = [t for t in before.values() if t["outcome"] == "none"
                         and t["status"] in {"local", "cpu"} and t["destination"] == uav]
            service = cpu.get(uav, {})
            served_cycles = float(service.get("service_cycles", 0.0))
            selected_task = executed["cpu"]["head_task_id"]
            self.cpu_slots.append({
                **self.identity, "slot": slot, "uav_id": uav,
                "cpu_queue_length": len(cpu_tasks),
                "cpu_queue_cycles": sum(t["remaining_cycles"] for t in cpu_tasks),
                "selected_task_id": selected_task,
                "served_task_id": service.get("task_id") if served_cycles > 0 else None,
                "cpu_frequency": executed["cpu"]["executed_frequency_hz"],
                "served_cycles": served_cycles,
                "cpu_energy_j": debits[uav]["cpu_energy_j"],
                "residual_energy_j": debits[uav]["residual_after_j"],
            })
            communication = executed["communication"]
            link = links.get(uav, {})
            tx_tasks = [t for t in before.values() if t["outcome"] == "none"
                        and t["status"] == "tx" and t["source_uav"] == uav]
            task_ids = {communication["head_task_id"]} | {
                t["task_id"] for t in link.get("task_services", ())}
            task_ids.discard(None)
            for task_id in sorted(task_ids) if task_ids else (None,):
                self.tx_slots.append({
                    **self.identity, "slot": slot, "uav_id": uav,
                    "destination_uav": communication["receiver_uav"],
                    "task_id": task_id,
                    "link_summary": task_id == communication["head_task_id"],
                    "transmit_proposed": proposals[uav]["tx_select"] != "idle",
                    "accepted": communication["accepted"],
                    "rejected": uav in rejections,
                    "rejection_reason": rejections.get(uav),
                    "transmitted_bits": bits_by_task.get(task_id, 0.0),
                    "link_transmitted_bits": float(link.get("service_bits", 0.0)),
                    "remaining_bits": after[task_id]["remaining_bits"] if task_id is not None else None,
                    "tx_backlog_tasks": len(tx_tasks),
                    "tx_backlog_bits": sum(t["remaining_bits"] for t in tx_tasks),
                    "tx_queue_length": sum(t["destination"] == communication["receiver_uav"] for t in tx_tasks),
                    "executed_power_w": communication["executed_power_w"],
                    "executed_ru_indices": list(communication["executed_ru_indices"]),
                    "effective_rate_bps": link.get("effective_rate_bps"),
                    "tx_energy_j": debits[uav]["transmit_energy_j"],
                })

        # Infer the stage after same-slot binding/TX transition, before settlement.
        # Terminal status itself no longer carries the previous physical stage.
        for task_id, task in after.items():
            record = self.tasks[task_id]
            outcome = task["outcome"].upper()
            record["final_status"] = outcome if outcome != "NONE" else task["status"].upper()
            if outcome in {"EXPIRED", "TRUNCATED"} and record["terminal_stage"] is None:
                if task["binding_slot"] is None:
                    stage = "UNBOUND"
                elif task["destination"] == task["source_uav"] or task["cpu_entry_slot"] is not None:
                    stage = "CPU_SERVICE" if record["cpu_first_service_slot"] is not None else "CPU_WAIT"
                else:
                    stage = "TX_SERVICE" if record["tx_first_service_slot"] is not None else "TX_WAIT"
                record["terminal_stage"] = stage
                if outcome == "EXPIRED":
                    record["expired_slot"] = slot
                    record["expire_stage"] = stage
            if outcome == "DONE":
                record["done_slot"] = task["completion_slot"]

    def task_records(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(dict(self.tasks[key]) for key in sorted(self.tasks))

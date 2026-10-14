"""Summarize observed one-shot Route oracle results; no environment calls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .route_counterfactual_oracle import best_set, energy_tie, write_json, write_jsonl


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "median": None, "P25": None, "P75": None}
    return {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "P25": float(np.percentile(values,25)), "P75": float(np.percentile(values,75))}


def anchored_classification(rows, mode="primary"):
    """Use ONE near-best set for all options; pairwise ties can be non-transitive.

    The underlying thresholds and best_set were committed before rollouts.
    This only makes the classification agree with the already-recorded global
    oracle_best_routes, without changing any environment observation/result.
    """
    best = best_set(rows, mode)
    if not best:
        return "F"
    local = any(x["route"] == "local" for x in best)
    remote = any(isinstance(x["route"],int) for x in best)
    return "T" if local and remote else "L" if local else "R"


def analyze(root):
    root = Path(root)
    decisions = read_rows(root/"actor_vs_oracle.jsonl")
    branches = read_rows(root/"branch_results.jsonl")
    grouped = defaultdict(list)
    for row in branches:
        grouped[row["cohort"], row["task_id"]].append(row)
    corrections = []
    for decision in decisions:
        rows = grouped[decision["cohort"],decision["task_id"]]
        original = dict(decision["classes"])
        decision["pairwise_classes"] = original
        decision["classes"] = {mode:anchored_classification(rows,mode) for mode in original}
        if original != decision["classes"]:
            corrections.append({"cohort":decision["cohort"],"task_id":decision["task_id"],"pairwise":original,"anchored":decision["classes"]})
        assert decision["oracle_best_routes"] == [x["route"] for x in best_set(rows)]
    write_jsonl(root/"actor_vs_oracle_classification_checked.jsonl",decisions)
    write_json(root/"classification_anchor_checks.json",{
        "thresholds_changed":False,"environment_results_changed":False,
        "rule":"class from pre-existing globally anchored best_set; raw pairwise labels retained",
        "corrections":corrections,
    })
    result = {}
    feature_summary = {}
    individual = []
    for cohort in ("8k", "4k_local"):
        cohort_decisions = [x for x in decisions if x["cohort"] == cohort]
        features = {x["task_id"]:x for x in read_rows(root/f"{cohort}_decision_features.jsonl")}
        expected = 117 if cohort == "8k" else 20
        if len(cohort_decisions) != expected:
            raise ValueError(f"incomplete cohort {cohort}")
        modes = {mode:dict(Counter(x["classes"][mode] for x in cohort_decisions)) for mode in ("primary","strict_completion","latency_only","energy_only")}
        cases, feature_rows = [], defaultdict(list)
        for decision in cohort_decisions:
            rows = grouped[cohort, decision["task_id"]]
            local = next(x for x in rows if x["route"] == "local")
            remote_rows = [x for x in rows if isinstance(x["route"],int)]
            done_remotes = best_set(remote_rows)
            remote = min(done_remotes,key=lambda x:(x["latency_slots"],x["task_service_energy_j"])) if done_remotes else remote_rows[0]
            factual = next(x for x in rows if x["route"] == decision["actor_route"])
            gains = remote["system"]["DONE"]-factual["system"]["DONE"]
            f = features[decision["task_id"]]
            remote_features = f["remotes"][str(remote["route"])]
            source = f["source_resources"]
            flat_features = {
                "required_cycles":f["required_cycles"], "input_bits":f["input_bits"], "slack_slots":f["slack_slots"],
                "source_cpu_queue_length":source["cpu_queue_length"], "source_cpu_queue_cycles":source["cpu_queue_cycles"],
                "source_energy_j":source["residual_energy_j"], "source_cpu_hz":source["cpu_capacity_hz"],
                "best_remote_cpu_queue_length":remote_features["cpu_queue_length"],
                "best_remote_cpu_queue_cycles":remote_features["cpu_queue_cycles"],
                "best_remote_energy_j":remote_features["residual_energy_j"], "best_remote_cpu_hz":remote_features["cpu_capacity_hz"],
                "best_remote_stale_gain":remote_features["stale_gain_mean"],
                "best_remote_rate_bps_proxy":remote_features["noise_limited_rate_bps_proxy"],
                "best_remote_tx_slots_proxy":remote_features["tx_time_slots_proxy"],
                "best_remote_tx_queue_compute_slots_proxy":remote_features["tx_queue_compute_slots_proxy"],
                "local_compute_slots_proxy":f["local_compute_slots_proxy"],
                "local_queue_compute_slots_proxy":f["local_queue_compute_slots_proxy"],
                "source_to_remote_cpu_ratio":source["cpu_capacity_hz"]/remote_features["cpu_capacity_hz"],
            }
            if not done_remotes:
                # Protocol refuses to call any failed candidate "best".
                # Retain the raw failed reference outcome, but exclude its
                # arbitrary identity from best-Remote feature statistics.
                for key in flat_features:
                    if key.startswith("best_remote_") or key == "source_to_remote_cpu_ratio":
                        flat_features[key] = None
            feature_rows[decision["classes"]["primary"]].append(flat_features)
            case = {
                **decision, "features":flat_features,
                "local":{k:local[k] for k in ("final_status","done_slot","expired_slot","latency_slots","deadline_margin_slots","task_cpu_energy_j","task_tx_energy_j","task_service_energy_j","cpu_wait_slots","cpu_service_elapsed_slots","system","other_changed_outcomes")},
                "best_remote_anchor":{k:remote[k] for k in ("route","final_status","done_slot","expired_slot","latency_slots","deadline_margin_slots","task_cpu_energy_j","task_tx_energy_j","task_service_energy_j","cpu_wait_slots","cpu_service_elapsed_slots","system","other_changed_outcomes")},
                "factual_status":factual["final_status"],
                "remote_reference_is_oracle_best":bool(done_remotes),
                "best_remote_vs_factual_system_done_delta": gains,
                "best_remote_task_vs_local_system_done_delta":remote["system"]["DONE"]-local["system"]["DONE"],
                "system_local_minus_best_remote_done":local["system"]["DONE"]-max(x["system"]["DONE"] for x in remote_rows),
                "system_best_remote_routes_done_only":[x["route"] for x in remote_rows if x["system"]["DONE"] == max(y["system"]["DONE"] for y in remote_rows)],
            }
            cases.append(case)
            individual.append(case)
        remote_best = [x for x in cohort_decisions if x["classes"]["primary"] == "R"]
        remote_actor_rows = [x for x in remote_best if isinstance(x["actor_route"],int)]
        destination_correct = sum(x["actor_route"] in x["oracle_best_routes"] for x in remote_actor_rows)
        local_rows = [next(x for x in grouped[cohort,d["task_id"]] if x["route"] == "local") for d in cohort_decisions]
        factual_fail_remote_rescue = [x for x in cases if x["factual_status"] != "DONE" and x["best_remote_anchor"]["final_status"] == "DONE"]
        factual_fail_local_rescue = [x for x in cases if x["factual_status"] != "DONE" and x["local"]["final_status"] == "DONE"]
        local_beats_all_remote_completion = [x for x in cases if x["local"]["final_status"] == "DONE" and x["best_remote_anchor"]["final_status"] != "DONE"]
        remote_beats_local_completion = [x for x in cases if x["local"]["final_status"] != "DONE" and x["best_remote_anchor"]["final_status"] == "DONE"]
        local_cases = [x for x in cases if x["classes"]["primary"] == "L"]
        metrics = {
            "n":len(cohort_decisions), "branch_count":sum(len(grouped[cohort,d["task_id"]]) for d in cohort_decisions),
            "classifications":modes,
            "raw_pairwise_classifications":{mode:dict(Counter(x["pairwise_classes"][mode] for x in cohort_decisions)) for mode in ("primary","strict_completion","latency_only","energy_only")},
            "actor_exact_option_agreement_count":sum(x["actor_agrees"] for x in cohort_decisions),
            "actor_exact_option_agreement_denominator_done_possible":sum(bool(x["oracle_best_routes"]) for x in cohort_decisions),
            "actor_top_level_compatible_count":sum(x["classes"]["primary"] in ({"L","T"} if x["actor_route"] == "local" else {"R","T"}) for x in cohort_decisions),
            "remote_oracle_tasks":len(remote_best), "remote_destination_correct":destination_correct,
            "remote_destination_denominator":len(remote_actor_rows),
            "remote_destination_accuracy":None if not remote_actor_rows else destination_correct/len(remote_actor_rows),
            "local_branch_outcomes":dict(Counter(x["final_status"] for x in local_rows)),
            "local_beats_all_remote_completion_ids":[x["task_id"] for x in local_beats_all_remote_completion],
            "remote_beats_local_completion_ids":[x["task_id"] for x in remote_beats_local_completion],
            "factual_fail_remote_rescue_ids":[x["task_id"] for x in factual_fail_remote_rescue],
            "remote_rescued_factual_status":dict(Counter(x["factual_status"] for x in factual_fail_remote_rescue)),
            "factual_fail_local_rescue_ids":[x["task_id"] for x in factual_fail_local_rescue],
            "primary_local_better_ids":[x["task_id"] for x in local_cases],
            "local_better_system_gain_distribution":dict(Counter(x["system_local_minus_best_remote_done"] for x in local_cases)),
            "system_local_minus_best_remote_done_distribution":dict(Counter(x["system_local_minus_best_remote_done"] for x in cases)),
            "system_primary_best_contains_local":sum("local" in x["system_best_routes"] for x in cases),
            "system_primary_best_only_local":sum(x["system_best_routes"] == ["local"] for x in cases),
            "system_actor_agreement":sum(x["actor_route"] in x["system_best_routes"] for x in cases),
            "local_minus_best_remote_latency_on_both_done":stats([x["local"]["latency_slots"]-x["best_remote_anchor"]["latency_slots"] for x in cases if x["local"]["final_status"] == x["best_remote_anchor"]["final_status"] == "DONE"]),
            "local_minus_best_remote_task_energy_on_both_done":stats([x["local"]["task_service_energy_j"]-x["best_remote_anchor"]["task_service_energy_j"] for x in cases if x["local"]["final_status"] == x["best_remote_anchor"]["final_status"] == "DONE"]),
            "all_states_have_legal_remote":all(bool(features[d["task_id"]]["remotes"]) for d in cohort_decisions),
        }
        result[cohort] = metrics
        feature_summary[cohort] = {label:{key:stats([row[key] for row in group]) for key in group[0]} for label,group in feature_rows.items()}
    write_json(root/"oracle_summary.json",result)
    write_json(root/"feature_group_statistics.json",feature_summary)
    write_json(root/"per_task_paired_analysis.json",individual)
    write_json(root/"four_k_local_paired_analysis.json",[x for x in individual if x["cohort"] == "4k_local"])
    write_json(root/"remote_destination_accuracy.json",{k:{f:v for f,v in r.items() if "destination" in f or "remote_rescue" in f or f == "remote_oracle_tasks"} for k,r in result.items()})
    with (root/"per_task_paired_analysis.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        columns=["cohort","task_id","source_uav","slot","actor_route","primary_class","strict_completion_class","local_status","remote_status","best_remote_route","local_latency_slots","remote_latency_slots","local_energy_j","remote_energy_j","local_system_DONE","remote_system_DONE","system_local_minus_best_remote_done"]
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        for case in individual:
            writer.writerow(dict(zip(columns,[case["cohort"],case["task_id"],case["source_uav"],case["slot"],case["actor_route"],case["classes"]["primary"],case["classes"]["strict_completion"],case["local"]["final_status"],case["best_remote_anchor"]["final_status"],case["best_remote_anchor"]["route"],case["local"]["latency_slots"],case["best_remote_anchor"]["latency_slots"],case["local"]["task_service_energy_j"],case["best_remote_anchor"]["task_service_energy_j"],case["local"]["system"]["DONE"],case["best_remote_anchor"]["system"]["DONE"],case["system_local_minus_best_remote_done"]])))
    return result,feature_summary,individual


def markdown_table(headers, rows):
    return "|"+"|".join(headers)+"|\n|"+"|".join("---" for _ in headers)+"|\n"+"\n".join("|"+"|".join(map(str,row))+"|" for row in rows)+"\n"


def write_report(root, result, features, cases, report_path):
    root = Path(root).resolve()
    a, b = result["8k"],result["4k_local"]
    evidence = lambda name: f"[{name}]({(root/name).as_posix()})"
    main_counts = a["classifications"]["primary"]
    table = markdown_table(["117 个首次绑定状态","数量","比例"],[
        [label,main_counts.get(key,0),f"{main_counts.get(key,0)/117:.2%}"]
        for key,label in (("L","Local clearly better"),("R","Remote clearly better"),("T","Near tie"),("F","Both fail"))])
    sensitivity = markdown_table(["预设标准","Local 优","Remote 优","Tie","Both fail"],[
        [name,*[a["classifications"][key].get(label,0) for label in ("L","R","T","F")]]
        for key,name in (("strict_completion","仅按时完成"),("latency_only","完成→延迟"),("energy_only","完成→能耗"),("primary","完成→延迟→能耗；主判定"))])
    four_k = markdown_table(["4k 的 20 个实际 Local","主判定","仅按时完成"],[
        [name,b["classifications"]["primary"].get(label,0),b["classifications"]["strict_completion"].get(label,0)]
        for label,name in (("L","Local 更优"),("R","Remote 更优"),("T","Tie"),("F","Both fail"))])
    keys=[("required_cycles",1e9,"任务计算量 Gcycles"),("input_bits",1e6,"任务大小 Mbit"),("slack_slots",1,"绑定后 hard-deadline 服务窗口 slots"),
          ("source_cpu_queue_length",1,"源 CPU queue length"),("source_cpu_queue_cycles",1e9,"源 CPU queue Gcycles"),("source_energy_j",1,"源剩余能量 J"),
          ("best_remote_cpu_queue_length",1,"目标 CPU queue length"),("best_remote_cpu_queue_cycles",1e9,"目标 CPU queue Gcycles"),("best_remote_energy_j",1,"目标剩余能量 J"),
          ("source_cpu_hz",1e9,"源 CPU GHz"),("best_remote_cpu_hz",1e9,"最佳 Remote CPU GHz"),("source_to_remote_cpu_ratio",1,"源/目标 CPU 容量比"),
          ("best_remote_stale_gain",1,"stale |h|²"),("best_remote_rate_bps_proxy",1e6,"noise-limited 速率代理 Mbit/s"),("best_remote_tx_slots_proxy",1,"TX 时间代理 slots"),
          ("local_compute_slots_proxy",1,"Local 满频计算代理 slots"),("local_queue_compute_slots_proxy",1,"Local backlog+计算代理 slots"),("best_remote_tx_queue_compute_slots_proxy",1,"Remote TX+backlog+计算代理 slots")]
    def fmt(value, scale):
        if not value["n"]:
            return "不可获得"
        numbers=[value[k]/scale for k in ("mean","median","P25","P75")]
        if scale == 1 and max(abs(v) for v in numbers)<1e-5:
            return f"{numbers[0]:.3e} / {numbers[1]:.3e} [{numbers[2]:.3e}, {numbers[3]:.3e}]"
        return f"{numbers[0]:.3f} / {numbers[1]:.3f} [{numbers[2]:.3f}, {numbers[3]:.3f}]"
    feature_table=markdown_table(["字段：mean / median [P25,P75]","Local-best，n=21；Remote字段n=20","Remote-best，n=92"],[
        [name,fmt(features["8k"]["L"][key],scale),fmt(features["8k"]["R"][key],scale)] for key,scale,name in keys])
    text=f"""# P1b：eval1044 Local / Remote 一次性成对 Oracle 诊断

## 一、结论：选择 C，优先处理 Remote destination ranking

最有证据的是 **C：顶层 Remote 多数合理，但 destination 错选是当前更直接的完成损失来源**。P1a-8k factual 仍为 94 DONE / 19 EXPIRED / 4 TRUNCATED，Local=0、Remote=117、Defer=0。本轮没有训练出或替换任何策略。

在所有 117 个真实首次绑定状态中，主判定支持 Remote 的有 **92/117=78.63%**，Local 更优 **21/117=17.95%**；因此不能宣称 Local 在此场景毫无价值，也不足以仅凭 Local=0 判定顶层 actor 普遍失效。更直接的证据是：**17/19 个 factual EXPIRED，以及 1 个 TRUNCATED，改选另一合法 Remote 即可让目标任务 DONE**。Local 独有的按时完成优势只有 **task68，1/117=0.85%**。

按预设任务 Oracle 的 Remote-best 92 个状态，actor 目的地正确 **39/92=42.39%**。其中一部分不一致只是延迟/能耗偏好差异；上面的 17 个过期救回才是更强的完成证据。**不建议以强制 Local、增加 Local 奖励或 Local 比例为下一步目标；优先辨别合法 Remote 候选的真实价值与系统外部性。** 本轮到此停止，没有执行任何后续机制修改。

## 二、方法、固定身份和可靠性

- 主 cohort：P1a final.pt，8k；固定 seed1044、remote_stress_v2、500 slots、CPU float32、actor.eval()、no_grad/inference_mode、原 masked deterministic argmax。
- 补充 cohort：step_4000.pt 的 20 个 factual Local 绑定状态，分支继续使用同一个 **4k checkpoint**。不是把 4k 状态交给 8k 策略。
- 每个状态包含 Local 及全部当前合法 Remote；不加入 Defer。只替换目标 UAV 当步 Route 字段，其他 UAV 以及该 UAV 的 TX/CPU 字段保持 factual 值；后续全部动作交回对应冻结 actor。
- 复用已有 clone_for_shadow() 完整对象图校验；克隆包含 lifecycle/queues/resources、CSI/历史、mobility/channel/traffic RNG 和 recurrent hidden。跨进程传输后再次核对完整状态与 RNG fingerprint。
- 所有分支推进到 episode 结束。逐 slot 比较 mobility、真实 channel/shadowing、stale CSI、arrival 与全部环境 RNG；干扰、队列、能耗和后续动作允许随干预变化。
- 共 **483** 个候选结果：8k **413**、4k **70**。其中 137 个 factual 候选复用已记录的真实 suffix，346 个替代候选实际推进 **88,760** 次 shadow transitions；另做两次完整无变化克隆复跑。
- 两次 factual 回放均与既有 task_lifecycle 逐字段一致；8k=94/19/4，4k=62/53/2。两者外生 trace SHA-256 均为 `11423f92499e671a3e7da9126bbbff9e7030eb4af260b02c86ab07b5bf3da28c`。
- final.pt SHA-256：`d303878ed57dcfca0ee9d04354ab678046fbf22b3546aef38a853b53bae7a062`；step_4000.pt：`edfd24a233d52f1c2d1db3b63675a20e76ca9ea0b6ec2f2bb4e863b233647f55`。
- actor digest、requires_grad/eval 状态、checkpoint 和原源码/config 均核验不变；未构造训练 optimizer，未调用 PPO 更新。详见 {evidence('read_only_integrity.json')}、{evidence('8k_no_intervention_replay_gate.json')}、{evidence('4k_local_no_intervention_replay_gate.json')}。

这是 **固定策略下的一步 Route 反事实**，不是全局最优调度。不同任务的分支分别从各自 factual 状态出发，不能把逐任务最优结果相加，宣称一个实际策略已达到 112/117 或 113/117。

## 三、预先固定的分类规则

阈值在第一次分支前保存：延迟差不超过 **1 slot=20ms**；能耗差不超过 `max(0.001J, 1%×较大能耗)`。先保留 DONE 候选，再以全体候选最早完成 slot 为统一锚点保留 1-slot 内候选，最后以其中最低能耗为锚点保留容差内候选。集合仅含 Local→L，仅含 Remote→R，同时含两者→T，均未完成→F。全部失败时不宣称某个选项最优。

原采集工具的 pairwise 分类在 **task34** 上出现容差非传递链：22 L / 91 R；统一到工具预先实现的全候选 best_set 后为 **21 L / 92 R**。本报告使用后者；阈值、实际 branch 结果和既有全候选 oracle_best_routes 均未改变。原始标签保留，核验见 {evidence('classification_anchor_checks.json')}、{evidence('actor_vs_oracle_classification_checked.jsonl')}。其余三种标准及 4k 分类没有此差异。

## 四、117 任务分类与 actor 一致率

{table}

{sensitivity}

仅按完成：Local 独有 DONE **1**，Remote 独有 DONE **26**，两者 DONE **86**，全部失败 **4**。逐任务独立 Local 分支中 87 DONE / 25 EXPIRED / 5 TRUNCATED；这些是 117 个不同根状态的目标结果，**不是 Local-only policy 的 episode 指标**。

{markdown_table(['Actor 顶层','Oracle Local','Oracle Remote','Tie','Both fail'],[['Remote',21,92,0,4]])}

顶层相容率 **92/117=78.63%**；排除 F 为 92/113=81.42%。完整选项（含目的地）一致 **39/117=33.33%**，排除 F 为 39/113=34.51%。实际 actor 仍是 0 Local / 117 Remote / 0 Defer；反事实 Local 不应计作 actor 学到了 Local。

## 五、Remote destination：主要可辨识损失

{markdown_table(['指标','结果'],[['Remote-best 状态',92],['actor 目的地属于全候选最优集合',39],['目的地一致率','42.39%'],['换 Remote 可救回 factual EXPIRED','17/19'],['换 Remote 可救回 factual TRUNCATED','1/4'],['换 Remote 可救回全部未完成目标','18/23'],['Local 可救回 factual 未完成目标',14],['其中其他 Remote 也能救回',13],['只有 Local 能完成、所有 Remote 均失败','task68']])}

可换 Remote 救回的 task_id：{a['factual_fail_remote_rescue_ids']}。这是固定后续策略下目标任务的成对证据，不是“18 个都能同时救回”的已运行结果。

例 **task114**：源 UAV1，slot478 绑定，actor→UAV3 后 TRUNCATED；改→UAV0 后 slot496 DONE，Local 仍 TRUNCATED。该单次分支全 episode 从 94 DONE 变为 95 DONE，是直接的 destination 错选证据。

例 **task5**：源 UAV2，actor→UAV3 后 EXPIRED；改 Local 在 slot46 DONE，改 UAV0 在 slot62 DONE。Local 对目标更快（17 vs 33 slots），但全 episode DONE 是 **93 vs 96**，factual 为94。不能因为 Local 更快就把该任务当成系统应优先 Local 的训练标签。

## 六、Oracle-2：系统结果与任务结果必须分开

每个根状态重新比较全 episode DONE 最大、EXPIRED 最少、CPU+TX 能耗最低（同一预设能耗容差）。与该状态下所有 Remote 中最高系统 DONE 比较，Local：

{markdown_table(['Local 系统 DONE 对比 best Remote','状态数'],[['更高',12],['相同',38],['更低',67]])}

12 个严格系统 Local 优势状态均多 **1** DONE；其 task_id 为 `[1,17,31,32,33,37,48,50,87,95,99,107]`。系统 Oracle 最优集合含 Local **42/117**，仅 Local **12/117**；factual actor 属于系统最优集合 **70/117=59.83%**。

关键区别：**21 个任务级 Local-best 状态中，Local 系统 DONE 超过 best Remote 的为0，持平8，更少13**。反过来，12 个严格系统 Local 优势状态的目标任务本身都属于任务级 Remote-best。说明任务自身的速度/终局与资源竞争及后续策略响应并不等价；不能把任务 Oracle 或 chosen-task target 直接当成系统最优真值。

task68 是唯一 Local 独有目标完成：Local slot320 DONE，所有 Remote 目标均失败；但 Local 和 Remote-UAV1 的**系统**结果都为95 DONE/18 EXPIRED/4 TRUNCATED，其他改变终局的任务不同。详见每候选 `other_changed_outcomes`。

## 七、Local-best / Remote-best 状态差异

以下均为对应 factual 决策时的已存在状态；分组由真实反事实结果确定。Remote 特征使用任务级最佳 Remote 代表；所有 Remote 均失败的 task68 没有最佳 Remote，其相关字段从分组统计中排除，故 Local-best 的 Remote 字段只有20个有效样本。

{feature_table}

Local-best 源 UAV0/1/2/3 为 **5/5/5/6**；Remote-best 为 **20/45/4/23**。较明确的组间差异是 Local-best 任务计算量中位数 **0.367G vs 0.541Gcycles**，源/目标 CPU 容量比中位数 **{features['8k']['L']['source_to_remote_cpu_ratio']['median']:.3f} vs {features['8k']['R']['source_to_remote_cpu_ratio']['median']:.3f}**，Local 满频计算时间中位数 **7.83 vs 15.84 slots**。源 queue 长度中位数均为1、队列和能量分布重叠明显，不能声称“只要源 queue 空就 Local”，也不能从本表推出差链路造成 Local 优势的因果结论。

时间定义：表中服务窗口是 `deadline_slot-route_slot`；绑定当 slot 不服务，从下一 slot 起、deadline slot 可服务，所以正好有这么多个可能服务 slot。环境 slot-start `deadline_slack()` 则为本列+1。实际 latency 为 `done_slot-generation_slot`，与环境 e2e_delay 定义一致。

计算代理使用当前 backlog/capacity，假设满频且没有抢占/竞争；TX 代理使用当时 stale CSI、满功率、全部 RU、仅噪声 Shannon 速率，**忽略实际 RU 分配、TX queue、干扰、outage 门限及 arbitration**，不是真实完成时间或 feasibility 证明。Remote-best 的速率/时间代理仅91/92有效，Local-best 的最佳 Remote 代理20/21有效；未读取未来 channel 拟合或作为 actor 输入。CPU queue/resources 是诊断可读的完整当前物理状态；同时保留 actor_observation_snapshot，不能假定 actor 已观测到全部远端即时量。

进一步严谨性：主 Local-best 中 task68 没有成功 Remote，因此其所选 Remote 参考不能用于声称“最佳 Remote”特征；完整 JSON 显式保留 best_remote_routes=[]。所有 F 状态同理。

## 八、4k 的20个实际 Local

factual Local 为 **16 DONE / 4 EXPIRED / 0 TRUNCATED**；都存在合法 Remote。

{four_k}

8 个 Local 确实任务级更优，10 个 Remote 更优，2 个全部失败；并非所有 Local 都是错误探索，也不能把 Local=20 直接判为学会了正确协同。只看完成，两类各自独有优势均为2，14个都能完成。

Local 独有完成为 task8、task22；Remote 独有为 task15、task21。4k 的 source0 task8 在 slot61 绑定，8k 在57；task22 为107 vs103；4k 绑定后 deadline 服务窗口为18/10，8k为22/14。因此这组更晚绑定、不同队列及不同 continuation 的结果不能直接转用为8k策略的同状态标签。见 {evidence('four_k_local_paired_analysis.json')}。

## 九、测试、范围与交付

- 最终相关测试：**18 passed，5 subtests passed，3 deselected，0 failed**。排除包含 trainer/resume/优化行为的三项，未运行可能启动训练更新的完整套件。首次测试默认系统 TEMP 不可写，重定向到仓库日志内临时目录后通过。
- 核验覆盖克隆对象图、跨进程传输一致、共同随机流、一次 Route-only 覆盖、非目标动作不变、分支 hidden 隔离、冻结 actor 续接、terminal/lifecycle 一致与 near-tie 全局锚定。两次全 factual 回放额外核对每个 joint action 和全部 task lifecycle；CPU/TX 正服务真实推进，Local DONE 有 CPU service、Remote DONE 有 TX complete。
- 串行 pilot 的前3个状态、11个候选与并行最终结果逐字段相同（排除隔离输出配置导致不同的 fingerprint），见 {evidence('parallel_serial_equivalence.json')}。
- 483 候选中10个 Remote 的 task TX energy 为多任务链路按 bits 分摊代理，其他 Remote为精确归因；Local 无 TX 能耗。能耗是 active CPU+TX，不能推广为包含飞行能耗的总能耗。完成/延迟结论不依赖该能耗分摊。
- 仅新增独立 evaluation/analysis 工具和测试，并按仓库约定更新 README 宏观进度。未修改网络、Q、credit、reward、探索、mask、CPU gate、TX 或环境 dynamics、既有配置/checkpoint。
- 权威原始数据：{evidence('branch_results.jsonl')}、{evidence('branch_results.csv')}；任务比较：{evidence('per_task_paired_analysis.csv')}；分类汇总：{evidence('oracle_summary.json')}；特征统计：{evidence('feature_group_statistics.json')}。临时串行及不可序列化的并行尝试只用于工程校验，移至 `logs/development/route_oracle_p1b_engineering/`，不混入483个最终候选。

**下一步优先研究 Remote destination 的 candidate-specific 价值辨别，并明确任务价值与系统外部性的差别。当前数据不支持为追求 Local 比例而重写顶层 Route，也不支持把场景宣称为绝对 Remote 最优。以上仅为固定 eval1044 的开发诊断，不是多场景、多种子或论文正式结论。**
"""
    Path(report_path).write_text(text,encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",required=True)
    parser.add_argument("--report",default=None)
    args = parser.parse_args(argv)
    result, features, cases = analyze(args.output_root)
    if args.report:
        write_report(args.output_root,result,features,cases,args.report)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

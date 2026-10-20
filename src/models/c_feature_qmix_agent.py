"""C-owned feature/scoring code, independently trained as exact agent Q utilities.

Scores are Q components, not PPO probabilities. Route, complete TX atoms and
joint CPU atoms factor; exhaustive TX suffix evaluation gives the exact legal
maximum of their sum. CPU support matches C's positive-frequency controller.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
from torch import nn

from ..env.actions import ActionProposal
from ..env.observation import BRANCH_ORDER
from .ca_gat_mappo import CAGATMAPPOActor
from .factorized_gat_qmix import QActions
from .integrated_scheduler import cpu_idle_deadline_critical


METHOD_ID = "c_feature_credit_qmix"
ARCHITECTURE_CONTRACT = {
    "name": "c_actor_features_exact_group_q_v1", "physical_route_dimension": 35,
    "route_private_backbone": True, "raw_graph_edges": True,
    "tx": "c_binary_destination_and_conditional_resource_scores_exact_suffix",
    "cpu": "c_joint_positive_frequency_and_last_chance_idle_gate",
    "state_value": "shared_GRU_linear_zero_initialized_action_independent",
    "parameters": "independent_random_initialization_no_C_weights",
}


@dataclass(frozen=True)
class CFeatureQActions(QActions):
    credit_values: dict


class CFeatureQNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        m = config.training.mappo
        q = config.training.qmix
        if any(getattr(q,name)!=getattr(m,name) for name in
               ("encoder_hidden_dimension","gat_layer_count","attention_head_count","gru_hidden_dimension")):
            raise ValueError("C-feature QMIX encoder dimensions must match C")
        if (config.reward.credit_assignment_mode!="responsibility_terminal"
                or m.agent_credit_mode!="role_decomposed" or q.gamma!=m.gamma
                or q.recurrent_sequence_length_slots>128):
            raise ValueError("C-feature QMIX requires conserved C responsibility credit, gamma and <=128 real gradient slots")
        if (m.route_feature_mode != "physical_relations_v1" or not m.route_private_backbone
                or m.route_decoder_mode != "unified_node_v1"
                or m.cpu_scheduler_mode != "deadline_joint_v1"
                or m.tx_upper_decoder_mode != "private_candidate_binary_v1"
                or m.cpu_anti_starvation_mode != "deadline_feasibility_v1"):
            raise ValueError("C-feature QMIX requires the complete matched C actor configuration")
        self.backbone = CAGATMAPPOActor(config)
        self.spec = self.backbone.spec
        # Role returns can be nonzero even when this slot has no active branch.
        # A state-only baseline preserves every legal-action argmax and keeps
        # the C backbone initialization and the process RNG exactly unchanged.
        with torch.random.fork_rng(devices=[]):
            self.value_head = nn.Linear(self.spec.gru_hidden_dimension, 1)
            nn.init.zeros_(self.value_head.weight)
            nn.init.zeros_(self.value_head.bias)
        self.cpu_queue_count = self.spec.uav_count
        self.cpu_frequency_count = len(config.action.cpu_frequency_levels)
        self.config = config

    @property
    def heads(self):
        b = self.backbone
        return {key: (b.cpu_joint_scheduler.idle_scorer if key == "cpu_queue" else
                      b.cpu_joint_scheduler.scorer if key == "cpu_frequency" else
                      b.action_heads[key]) for key in BRANCH_ORDER}

    @property
    def gru(self): return self.backbone.gru
    @property
    def graph(self): return self.backbone.graph_encoder
    @property
    def self_encoder(self): return self.backbone.self_encoder

    def initial_hidden(self, batch_size=1):
        return self.backbone.initial_hidden(batch_size)

    def _features(self, batch, hidden):
        batch.validate(self.spec)
        h = self.spec.gru_hidden_dimension
        if hidden.shape != (batch.self_features.shape[0], self.spec.uav_count, 2*h):
            raise ValueError("C-feature QMIX requires separate shared/Route hidden lanes")
        shared = self.backbone._backbone_features(batch, hidden[..., :h])
        route = self.backbone._backbone_features(batch, hidden[..., h:], private=True)
        return shared, route, torch.cat((shared[1], route[1]), -1)

    def encode(self, batch, hidden):
        shared, _, outgoing = self._features(batch, hidden)
        return shared[0], outgoing

    def tx_tables(self, recurrent, batch, first):
        """Vectorized C conditional heads for every TX prefix, with raw Q scores."""
        b = self.backbone
        shape, h = recurrent.shape[:3], recurrent.shape[-1]
        d = self.spec.action_dimensions
        def embedding(name):
            return b.branch_embeddings[name](torch.arange(d[name], device=recurrent.device))
        tx, rg, width = (embedding(key) for key in ("tx_select", "resource_group", "resource_width"))
        nt, ng, nw = len(tx), len(rg), len(width)
        base = recurrent.unsqueeze(-2).expand(*shape, nt, h)
        te = tx.view(1,1,1,nt,h).expand_as(base)
        groups = b.action_heads["resource_group"](torch.cat((base,te),-1))
        base2 = base.unsqueeze(-2).expand(*shape,nt,ng,h)
        te2 = te.unsqueeze(-2).expand_as(base2)
        ge2 = rg.view(1,1,1,1,ng,h).expand_as(base2)
        widths = b.action_heads["resource_width"](torch.cat((base2,te2,ge2),-1))
        base3 = base2.unsqueeze(-2).expand(*shape,nt,ng,nw,h)
        te3, ge3 = (value.unsqueeze(-2).expand_as(base3) for value in (te2,ge2))
        we3 = width.view(1,1,1,1,1,nw,h).expand_as(base3)
        powers = b.action_heads["power_level"](torch.cat((base3,te3,ge3,we3),-1))
        head = b.action_heads["tx_select"]
        queues = batch.self_features[...,25:25+8*self.spec.uav_count].reshape(*shape,self.spec.uav_count,8)
        candidates = [head._destinations(value) for value in
                      (queues,batch.neighbor_public_features,batch.edge_features)]
        rec = recurrent.unsqueeze(-2).expand(*shape,self.spec.uav_count-1,h)
        destinations = head.destination_head(head.candidate_encoder(torch.cat((rec,*candidates),-1))).squeeze(-1)
        binary = b.tx_upper_private(batch,batch.action_masks["tx_select"][...,1:])
        selections = torch.cat((binary[...,:1],binary[...,1:]+destinations),-1)
        atoms = [(0,0,first.resource_width_domain.index(first.canonical_width),
                  first.power_level_domain.index(first.canonical_power))]
        atoms += [(t,g,w,p) for t,g,w,p in product(range(1,nt),range(1,ng),range(nw),range(d["power_level"]))
                  if first.resource_group_domain[g]+first.resource_width_domain[w]-1 <= first.resource_group_count]
        atom_indices = torch.tensor(atoms,device=recurrent.device,dtype=torch.long)
        ti,gi,wi,pi = atom_indices.unbind(-1)
        suffix = groups[...,ti,gi]+widths[...,ti,gi,wi]+powers[...,ti,gi,wi,pi]
        scores = selections[...,ti]+torch.where(ti>0,suffix,torch.zeros_like(suffix))
        return selections,groups,widths,powers,binary,atom_indices,scores

    def actions(self, batch, hidden, contracts, *, proposals=None, epsilon=0., rng=None,
                force_transmit_source=None):
        if not 0 <= epsilon <= 1 or (epsilon and rng is None):
            raise ValueError("epsilon exploration requires an independent RNG")
        shared, route, outgoing = self._features(batch,hidden)
        rec = shared[0];shape = rec.shape[:3]
        flat = [mask for row in contracts for step in row for mask in step]
        if len(flat) != int(np.prod(shape)) or any(mask.uav_id != i%shape[2] for i,mask in enumerate(flat)):
            raise ValueError("Q contracts require stable [batch,time,agent] axes")
        first = flat[0]
        route_q = self.backbone.route_logits(route[0],route[5],batch.edge_features,
            cooperative_features=route[2],local_endpoint_features=route[3],
            candidate_endpoint_features=route[4],self_features=batch.self_features,
            route_action_mask=batch.action_masks["route"])
        tx,g,w,p,binary,atoms,tx_scores = self.tx_tables(rec,batch,first)
        device = rec.device
        tx_mask = torch.tensor(np.stack([mask.tx_select_mask for mask in flat]),device=device).reshape(*shape,-1)
        power_mask = torch.tensor(np.stack([mask.mask_for("power_level",
            {"tx_select":mask.tx_select_domain[1],"resource_group":1}) for mask in flat]),device=device).reshape(*shape,-1)
        tx_support = tx_mask[...,atoms[:,0]] & ((atoms[:,0]==0) | power_mask[...,atoms[:,3]])
        if force_transmit_source is not None:
            tx_support = tx_support.clone()
            tx_support[...,force_transmit_source,0] = False
        cpu_q,_ = self.backbone.cpu_joint_scheduler(rec,batch)
        positive = [i for i,f in enumerate(first.cpu_frequency_domain) if f>0]
        cpu_atoms = [(0,first.cpu_frequency_domain.index(first.canonical_cpu_frequency))]
        cpu_atoms += list(product(range(1,len(first.cpu_queue_domain)),positive))
        cpu_rows = []
        for mask in flat:
            row = [bool(mask.cpu_queue_mask[0])]
            for qi,queue in enumerate(mask.cpu_queue_domain[1:],1):
                fm = mask.mask_for("cpu_frequency",{"cpu_queue":queue})
                row.extend(bool(mask.cpu_queue_mask[qi] and fm[k]) for k in positive)
            if cpu_idle_deadline_critical(mask,self.backbone.cpu_joint_scheduler.delta_t):row[0] = False
            cpu_rows.append(row)
        cpu_support = torch.tensor(cpu_rows,device=device,dtype=torch.bool).reshape_as(cpu_q)
        rm = torch.tensor(np.stack([mask.route_mask for mask in flat]),device=device).reshape_as(route_q)
        if not tx_support.any(-1).all() or not cpu_support.any(-1).all():
            raise ValueError("empty C-matched controller support")
        ri = route_q.masked_fill(~rm,-torch.inf).argmax(-1).reshape(-1).tolist()
        ta = tx_scores.masked_fill(~tx_support,-torch.inf).argmax(-1).reshape(-1).tolist()
        ca = cpu_q.masked_fill(~cpu_support,-torch.inf).argmax(-1).reshape(-1).tolist()
        explored = [bool(rng.random()<epsilon) for _ in flat] if proposals is None and epsilon else [False]*len(flat)
        teachers = None if proposals is None else [v for row in proposals for step in row for v in step]
        atom_list = atoms.tolist();atom_lookup = {tuple(v):i for i,v in enumerate(atom_list)}
        cpu_lookup = {pair:i for i,pair in enumerate(cpu_atoms)}
        chosen=[];indices={key:[] for key in BRANCH_ORDER};masks={key:[] for key in BRANCH_ORDER}
        for i,mask in enumerate(flat):
            if teachers is not None:
                proposal = teachers[i]
                if not mask.is_legal(proposal):raise ValueError("teacher violates original action contract")
                values = [mask.domain_for(key).index(mask._normalise(key,getattr(proposal,key))) for key in BRANCH_ORDER]
                ri[i]=values[0];ta[i]=atom_lookup[tuple(values[1:5])]
                if tuple(values[5:]) not in cpu_lookup:raise ValueError("teacher outside C positive CPU support")
                ca[i]=cpu_lookup[tuple(values[5:])]
            elif explored[i]:
                ri[i]=int(rng.choice(np.flatnonzero(mask.route_mask)))
                # Agent-level epsilon; conditionally uniform original TX tree.
                context={};tx_indices=[]
                for key in BRANCH_ORDER[1:5]:
                    legal=np.flatnonzero(mask.mask_for(key,context))
                    index=int(rng.choice(legal));tx_indices.append(index)
                    context[key]=mask.domain_for(key)[index]
                ta[i]=atom_lookup[tuple(tx_indices)]
                ca[i]=int(rng.choice(np.flatnonzero(cpu_rows[i])))
            if not bool(cpu_rows[i][ca[i]]):raise ValueError("teacher/exploration violates C CPU last-chance support")
            values=[ri[i],*atom_list[ta[i]],*cpu_atoms[ca[i]]]
            proposal=ActionProposal(uav_id=mask.uav_id,**{key:mask.domain_for(key)[index] for key,index in zip(BRANCH_ORDER,values)})
            if not mask.is_legal(proposal):raise ValueError("Q decoder produced an illegal proposal")
            chosen.append(proposal)
            for key,index in zip(BRANCH_ORDER,values):
                indices[key].append(index)
                branch_mask=mask.mask_for(key,proposal).copy()
                if key=="cpu_queue":
                    branch_mask[0]=cpu_rows[i][0]
                    for qi in range(1,len(branch_mask)):
                        branch_mask[qi]=any(cpu_rows[i][1+(qi-1)*len(positive):1+qi*len(positive)])
                if key=="cpu_frequency" and values[5]>0:
                    branch_mask &= np.asarray(mask.cpu_frequency_domain)>0
                masks[key].append(branch_mask)
        ix={key:torch.tensor(value,device=device).reshape(shape) for key,value in indices.items()}
        n=int(np.prod(shape)); ar=torch.arange(n,device=device)
        def selected_prefix(value,*keys):
            prefix=tuple(ix[key].reshape(-1) for key in keys)
            return value.reshape(n,*value.shape[3:])[ar,*prefix].reshape(*shape,-1)
        frequency=cpu_q.new_zeros((*shape,self.spec.uav_count,len(first.cpu_frequency_domain)))
        frequency[...,positive]=cpu_q[...,1:].reshape(*shape,self.spec.uav_count,len(positive))
        queue=cpu_q.new_zeros((*shape,len(first.cpu_queue_domain)));queue[...,0]=cpu_q[...,0]
        fq=frequency.reshape(n,self.spec.uav_count,-1)[ar,(ix["cpu_queue"].reshape(-1)-1).clamp_min(0)].reshape(*shape,-1)
        fq=torch.where((ix["cpu_queue"]>0).unsqueeze(-1),fq,torch.zeros_like(fq))
        branch={"route":route_q,"tx_select":tx,"resource_group":selected_prefix(g,"tx_select"),
                "resource_width":selected_prefix(w,"tx_select","resource_group"),
                "power_level":selected_prefix(p,"tx_select","resource_group","resource_width"),
                "cpu_queue":queue,"cpu_frequency":fq}
        active=torch.tensor([m.active_indicators(v) for m,v in zip(flat,chosen)],device=device).reshape(*shape,7)
        state_value=self.value_head(rec).squeeze(-1)
        utility=state_value+sum(branch[key].gather(-1,ix[key].unsqueeze(-1)).squeeze(-1)*active[...,j] for j,key in enumerate(BRANCH_ORDER))
        grid=tuple(tuple(tuple(chosen[(b*shape[1]+t)*shape[2]:(b*shape[1]+t+1)*shape[2]]) for t in range(shape[1])) for b in range(shape[0]))
        if not torch.isfinite(utility).all():raise ValueError("nonfinite C-feature Q utility")
        return CFeatureQActions(grid,utility,outgoing,ix,
            {key:torch.tensor(np.stack(v),device=device).reshape(*shape,-1) for key,v in masks.items()},
            active,branch,torch.tensor(explored,device=device).reshape(shape),frequency,
            {"state_value":state_value,"route":route_q,"tx_upper":binary,"tx_atom_scores":tx_scores,
             "tx_atom_indices":atoms,"tx_support":tx_support,"cpu_atom_scores":cpu_q,
             "cpu_support":cpu_support})

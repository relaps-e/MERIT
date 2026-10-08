# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax


class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, node_dict, edge_dict, n_heads, dropout=0.5):
        super(GraphTransformerLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )
        self.relation_msg = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )
        self.relation_weight_scale = nn.Parameter(torch.zeros(self.num_relations, self.n_heads))
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            active_relations = {}
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]
                if sub_graph.num_edges() == 0:
                    continue

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]
                active_relations[etype] = e_id

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata["k"] = k
                sub_graph.dstdata["q"] = q
                sub_graph.srcdata["v_%d" % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u("q", "k", "t"))
                attn_score = sub_graph.edata.pop("t").sum(-1) * relation_pri / self.sqrt_dk
                edge_weight = sub_graph.edata["weight"].float().view(-1, 1)
                edge_weight = torch.nan_to_num(edge_weight, nan=0.0, posinf=1.0, neginf=0.0)
                edge_weight = torch.clamp(edge_weight, min=0.0)
                attn_score = attn_score + torch.log1p(edge_weight) * self.relation_weight_scale[
                    e_id
                ].view(1, -1)
                attn_score = edge_softmax(sub_graph, attn_score, norm_by="dst")

                sub_graph.edata["t"] = attn_score.unsqueeze(-1)

            if active_relations:
                G.multi_update_all(
                    {
                        etype: (fn.u_mul_e("v_%d" % e_id, "t", "m"), fn.sum("m", "t"))
                        for etype, e_id in active_relations.items()
                    },
                    cross_reducer="mean",
                )

            new_h = {}
            for ntype in G.ntypes:
                """
                Step 3: Target-specific Aggregation
                x = norm( W[node_type] * gelu( Agg(x) ) + x )
                """
                n_id = node_dict[ntype]
                if "t" not in G.nodes[ntype].data:
                    new_h[ntype] = h[ntype]
                    continue
                alpha = torch.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data["t"].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                new_h[ntype] = self.norms[n_id](trans_out)
            return new_h


class HeterogeneousGraphTransformer(nn.Module):
    def __init__(self, node_dict, edge_dict, n_inp, n_hid, n_out, n_layers, n_heads):
        super().__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws = nn.ModuleList()
        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp, n_hid))
        for _ in range(n_layers):
            self.gcs.append(GraphTransformerLayer(n_hid, n_hid, node_dict, edge_dict, n_heads))
        self.out_layers = nn.ModuleList()
        for _ in range(len(node_dict)):
            self.out_layers.append(nn.Linear(n_hid, n_out))

    def forward(self, G, inp):
        h = {}
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id](inp[ntype]))
        for i in range(self.n_layers):
            h = self.gcs[i](G, h)

        out_dict = {}
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            out_dict[ntype] = self.out_layers[n_id](h[ntype])
        return out_dict

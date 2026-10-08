"""Neighbor perturbations and embedding invariance learning."""

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


class EIL:
    """Neighbor perturbations and embedding invariance learning."""

    def _prepare_eil(self, M_train):
        """Cluster source descriptors for neighbor replacement."""
        if not self.use_eil:
            return
        values = np.asarray(M_train, dtype=np.float64)
        mean = np.nanmean(values, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        values = np.where(np.isfinite(values), values, mean[None, :])
        scale = np.std(values, axis=0)
        scale[scale < 1e-08] = 1.0
        values = (values - np.mean(values, axis=0)) / scale
        n_clusters = min(self.eil_num_clusters, max(1, values.shape[0]))
        self._eil_clusters = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(
            values
        )

    def _perturb_neighbors(self, G_cpu):
        """Sample edge dropout or cluster based neighbor replacement."""
        src, dst = G_cpu.edges(etype="T2T_sim")
        src_np = src.numpy().astype(np.int64, copy=True)
        dst_np = dst.numpy().astype(np.int64, copy=True)
        weight_np = G_cpu.edges["T2T_sim"].data["weight"].numpy().copy()
        alpha = max(0.0, min(1.0, float(self.eil_edge_dropout)))
        strategy = "dropout" if np.random.random() < 0.5 else "replacement"
        if strategy == "dropout":
            keep = np.ones(len(src_np), dtype=bool)
            for task_id in np.unique(src_np):
                indices = np.where(src_np == task_id)[0]
                drop_count = min(
                    int(np.floor(alpha * len(indices))),
                    max(0, len(indices) - self.eil_min_keep_t2t),
                )
                if drop_count:
                    keep[np.random.choice(indices, size=drop_count, replace=False)] = False
            return (src_np[keep], dst_np[keep], weight_np[keep])
        labels = self._eil_clusters
        strength = self.eil_replacement_strength
        if strength == "random":
            strength = "mild" if np.random.random() < 0.5 else "strong"
        all_tasks = np.arange(len(labels), dtype=np.int64)
        for task_id in np.unique(src_np):
            indices = np.where(src_np == task_id)[0]
            top_count = min(self.eil_neighbor_pool_size, len(indices))
            if top_count == 0:
                continue
            top = indices[np.argsort(-weight_np[indices], kind="stable")[:top_count]]
            replace_count = int(np.floor(alpha * top_count))
            if alpha > 0 and replace_count == 0:
                replace_count = 1
            replace_count = min(replace_count, len(top))
            if replace_count == 0:
                continue
            chosen = np.random.choice(top, size=replace_count, replace=False)
            occupied = set((int(value) for value in dst_np[indices]))
            for edge_index in chosen:
                if strength == "mild":
                    candidates = all_tasks[labels == labels[task_id]]
                else:
                    candidates = all_tasks[labels != labels[task_id]]
                candidates = np.asarray(
                    [
                        int(value)
                        for value in candidates
                        if int(value) != int(task_id) and int(value) not in occupied
                    ],
                    dtype=np.int64,
                )
                if candidates.size == 0:
                    continue
                old = int(dst_np[edge_index])
                new = int(np.random.choice(candidates))
                dst_np[edge_index] = new
                occupied.discard(old)
                occupied.add(new)
        return (src_np, dst_np, weight_np)

    def _generate_eil_graph(self, G_original):
        G_cpu = G_original.to("cpu")
        graph_data, edge_weights = ({}, {})
        for relation in G_cpu.canonical_etypes:
            etype = relation[1]
            if etype == "T2T_sim":
                src, dst, weights = self._perturb_neighbors(G_cpu)
                graph_data[relation] = (
                    torch.as_tensor(src, dtype=torch.long),
                    torch.as_tensor(dst, dtype=torch.long),
                )
                edge_weights[etype] = torch.as_tensor(weights, dtype=torch.float32)
            else:
                graph_data[relation] = G_cpu.edges(etype=etype)
                edge_weights[etype] = G_cpu.edges[etype].data["weight"].clone()
        G_aug = dgl.heterograph(
            graph_data,
            num_nodes_dict={
                "task": G_cpu.number_of_nodes("task"),
                "model": G_cpu.number_of_nodes("model"),
            },
        )
        for node_type in G_cpu.ntypes:
            G_aug.nodes[node_type].data["feat"] = G_cpu.nodes[node_type].data["feat"].clone()
        for etype, weights in edge_weights.items():
            G_aug.edges[etype].data["weight"] = weights
        return G_aug.to(self.device)

    def _infonce_loss(self, z_orig, z_aug):
        z_orig = F.normalize(z_orig, dim=1)
        z_aug = F.normalize(z_aug, dim=1)
        logits = torch.mm(z_orig, z_aug.T) / self.eil_temperature
        labels = torch.arange(z_orig.shape[0], device=z_orig.device)
        return F.cross_entropy(logits, labels)

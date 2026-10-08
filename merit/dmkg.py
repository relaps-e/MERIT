"""Dataset-model knowledge graph construction and target insertion."""

import copy

import dgl
import numpy as np
import torch
from scipy.stats import kendalltau


class DMKG:
    """Dataset-model knowledge graph construction and target insertion."""

    def _compute_kendall_correlation(self, P, mode="task", min_common=3):
        """Pairwise Kendall correlation over shared observed entries."""
        values = P if mode == "task" else P.T
        n = len(values)
        corr_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                common = ~np.isnan(values[i]) & ~np.isnan(values[j])
                tau = 0.0
                if common.sum() >= min_common:
                    tau, _ = kendalltau(values[i, common], values[j, common])
                    tau = 0.0 if np.isnan(tau) else tau
                corr_matrix[i, j] = tau
                corr_matrix[j, i] = tau
            corr_matrix[i, i] = 1.0
        return corr_matrix

    def _build_tmg_graph(self, task_features, model_features, P_train):
        """Build dataset and model relations with observed performance edges."""
        n_tasks, n_models = P_train.shape
        t2t_corr = self._compute_kendall_correlation(P_train, mode="task")
        t2t_src, t2t_dst, t2t_weights = ([], [], [])
        for i in range(n_tasks):
            similarities = t2t_corr[i].copy()
            similarities[i] = -np.inf
            topk_indices = (
                np.argsort(similarities)[-self.k_similar_tasks :]
                if self.k_similar_tasks
                else np.empty(0, dtype=np.int64)
            )
            for j in topk_indices:
                if similarities[j] > 0:
                    t2t_src.append(i)
                    t2t_dst.append(j)
                    t2t_weights.append(similarities[j])
        m2m_src, m2m_dst, m2m_weights = ([], [], [])
        if not self.skip_m2m_edges and n_models <= 5000:
            m2m_corr = self._compute_kendall_correlation(P_train, mode="model")
            for i in range(n_models):
                similarities = m2m_corr[i].copy()
                similarities[i] = -np.inf
                topk_indices = (
                    np.argsort(similarities)[-self.k_similar_models :]
                    if self.k_similar_models
                    else np.empty(0, dtype=np.int64)
                )
                for j in topk_indices:
                    if similarities[j] > 0:
                        m2m_src.append(i)
                        m2m_dst.append(j)
                        m2m_weights.append(similarities[j])
        t2m_src, t2m_dst, t2m_weights = ([], [], [])
        m2t_src, m2t_dst, m2t_weights = ([], [], [])
        for i in range(n_tasks):
            task_perfs = P_train[i, :]
            valid_mask = ~np.isnan(task_perfs)
            valid_indices = np.where(valid_mask)[0]
            valid_perfs = task_perfs[valid_mask]
            if len(valid_indices) == 0:
                continue
            n_valid = len(valid_indices)
            sorted_order = np.argsort(valid_perfs)
            ranks = np.zeros(n_valid)
            ranks[sorted_order] = np.arange(n_valid)
            if n_valid > 1:
                percentiles = ranks / (n_valid - 1)
            else:
                percentiles = np.array([1.0])
            for local_idx, j in enumerate(valid_indices):
                edge_weight = percentiles[local_idx] ** self.edge_weight_alpha
                t2m_src.append(i)
                t2m_dst.append(j)
                t2m_weights.append(edge_weight)
                m2t_src.append(j)
                m2t_dst.append(i)
                m2t_weights.append(edge_weight)
        graph_data = {}
        graph_data["task", "T2T_sim", "task"] = (
            torch.LongTensor(t2t_src),
            torch.LongTensor(t2t_dst),
        )
        if len(m2m_src) > 0:
            graph_data["model", "M2M_sim", "model"] = (
                torch.LongTensor(m2m_src),
                torch.LongTensor(m2m_dst),
            )
        if len(t2m_src) > 0:
            graph_data["task", "T2M_perf", "model"] = (
                torch.LongTensor(t2m_src),
                torch.LongTensor(t2m_dst),
            )
        if len(m2t_src) > 0:
            graph_data["model", "M2T_perf", "task"] = (
                torch.LongTensor(m2t_src),
                torch.LongTensor(m2t_dst),
            )
        G = dgl.heterograph(graph_data, num_nodes_dict={"task": n_tasks, "model": n_models})
        G.nodes["task"].data["feat"] = torch.FloatTensor(task_features)
        G.nodes["model"].data["feat"] = torch.FloatTensor(model_features)
        if "T2T_sim" in G.etypes:
            G.edges["T2T_sim"].data["weight"] = torch.FloatTensor(t2t_weights)
        if "M2M_sim" in G.etypes:
            G.edges["M2M_sim"].data["weight"] = torch.FloatTensor(m2m_weights)
        if "T2M_perf" in G.etypes:
            G.edges["T2M_perf"].data["weight"] = torch.FloatTensor(t2m_weights)
        if "M2T_perf" in G.etypes:
            G.edges["M2T_perf"].data["weight"] = torch.FloatTensor(m2t_weights)
        return G

    def _similarity_features(self, M_train, M_other):
        train = np.asarray(M_train, dtype=np.float64)
        other = np.asarray(M_other, dtype=np.float64)
        train_view = np.nan_to_num(train, nan=0.0, posinf=0.0, neginf=0.0)
        other_view = np.nan_to_num(other, nan=0.0, posinf=0.0, neginf=0.0)
        return (train_view, other_view)

    def _augment_graph_with_test_nodes(
        self, target_task_features, topk_train_indices, topk_similarities
    ):
        """Attach target datasets through weighted bidirectional neighbor edges."""
        target_task_features = np.asarray(target_task_features, dtype=np.float32)
        neighbors = np.asarray(topk_train_indices)
        weights = np.asarray(topk_similarities, dtype=np.float32)
        n_test = target_task_features.shape[0]
        n_train = self.G_train.number_of_nodes("task")
        G_test = copy.deepcopy(self.G_train.to("cpu"))
        historical_features = G_test.nodes["task"].data["feat"]
        G_test.add_nodes(
            n_test,
            ntype="task",
            data={"feat": torch.as_tensor(target_task_features, dtype=historical_features.dtype)},
        )
        historical_ids = neighbors.reshape(-1)
        target_ids = np.repeat(np.arange(n_train, n_train + n_test), neighbors.shape[1])
        edge_weights = weights.reshape(-1)
        if historical_ids.size:
            G_test.add_edges(
                torch.LongTensor(np.concatenate([historical_ids, target_ids])),
                torch.LongTensor(np.concatenate([target_ids, historical_ids])),
                data={"weight": torch.FloatTensor(np.concatenate([edge_weights, edge_weights]))},
                etype="T2T_sim",
            )
        return G_test

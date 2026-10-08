"""Gaussian evidence recovery and recovered performance edges."""

import numpy as np
import torch
import torch.nn.functional as F


class ERL:
    """Gaussian evidence recovery and recovered performance edges."""

    def _prepare_erl_state(self, P_train):
        """Initialize Gaussian recovery for missing source performance entries."""
        P_obs = np.asarray(P_train, dtype=np.float64)
        observed = np.isfinite(P_obs)
        missing = ~observed
        if not missing.any():
            self._erl_missing_mask = None
            return
        normalized = P_obs.copy()
        finite_values = normalized[observed]
        if finite_values.size and (finite_values.min() < 0.0 or finite_values.max() > 1.0):
            for row in normalized:
                valid = np.isfinite(row)
                if not valid.any():
                    continue
                lo, hi = (float(row[valid].min()), float(row[valid].max()))
                row[valid] = (row[valid] - lo) / (hi - lo) if hi > lo else 0.0
        self._erl_missing_mask = missing
        self._erl_added_mask = np.zeros_like(missing, dtype=bool)
        self._erl_observed_values = normalized
        self._erl_posterior_mean = None
        self._erl_edges = []
        self._erl_observed_edges = int(observed.sum())
        self._erl_max_edges = int(missing.sum())

    def _erl_target_edge_count(self, epoch):
        if self._erl_missing_mask is None or self._erl_max_edges <= 0:
            return 0
        elapsed = max(0, int(epoch) - self.erl_warmup_epochs)
        round_index = elapsed // self.erl_interval
        initial = int(round(self._erl_observed_edges * self.erl_initial_budget_ratio))
        growth = int(round(self._erl_observed_edges * self.erl_budget_growth_ratio))
        return min(self._erl_max_edges, max(0, initial + round_index * growth))

    def _select_recovered_edges(self, P_pred, edge_budget):
        """Update Gaussian posterior means and select confident missing edges."""
        if self._erl_missing_mask is None:
            return []
        edge_budget = int(edge_budget)
        if edge_budget <= 0:
            return []
        score = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(P_pred, dtype=np.float64), -60.0, 60.0)))
        observed = np.isfinite(self._erl_observed_values)
        posterior = score.copy()
        posterior[observed] = (
            self.erl_sigma2 * score[observed] + self.erl_tau2 * self._erl_observed_values[observed]
        ) / (self.erl_sigma2 + self.erl_tau2)
        self._erl_posterior_mean = posterior
        eligible = (
            self._erl_missing_mask
            & ~self._erl_added_mask
            & np.isfinite(score)
            & (score >= self.erl_min_confidence)
        )
        coords = np.argwhere(eligible)
        if coords.size == 0:
            return []
        values = score[eligible]
        order = np.argsort(-values, kind="stable")[:edge_budget]
        return [
            (int(coords[idx, 0]), int(coords[idx, 1]), float(values[idx]), float(values[idx]))
            for idx in order
        ]

    def _graph_with_recovered_edges(self):
        if not self._erl_edges:
            return self.G_train
        G = self.G_train.to("cpu").clone()
        t_src = torch.LongTensor([edge[0] for edge in self._erl_edges])
        m_dst = torch.LongTensor([edge[1] for edge in self._erl_edges])
        weights = torch.FloatTensor([edge[2] for edge in self._erl_edges])
        G.add_edges(t_src, m_dst, data={"weight": weights}, etype="T2M_perf")
        G.add_edges(m_dst, t_src, data={"weight": weights}, etype="M2T_perf")
        return G.to(self.device)

    def _erl_soft_loss(self, h_task, h_model):
        if self._erl_posterior_mean is None:
            return h_task.sum() * 0.0
        scores = torch.sigmoid(torch.mm(self.bilinear(h_task), h_model.T))
        target = torch.as_tensor(self._erl_posterior_mean, dtype=scores.dtype, device=scores.device)
        return F.mse_loss(scores, target)

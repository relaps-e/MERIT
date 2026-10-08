"""MERIT model assembly, training and candidate prediction."""

import gc

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

from .dmkg import DMKG
from .eil import EIL
from .encoder import HeterogeneousGraphTransformer
from .erl import ERL
from .losses import listmle_loss


class MERIT(DMKG, ERL, EIL):
    """Model selection with graph encoding, evidence recovery and invariance learning."""

    def __init__(
        self,
        hid_dim=32,
        n_layers=2,
        n_heads=4,
        k_nn=10,
        k_similar_tasks=10,
        k_similar_models=20,
        edge_weight_alpha=3.0,
        epochs=500,
        lr=0.001,
        patience=20,
        skip_m2m_edges=False,
        device="cpu",
        use_eil=False,
        lambda_eil=0.015,
        eil_temperature=0.5,
        eil_projection_dim=64,
        eil_embed_consistency=1.0,
        eil_edge_dropout=0.1,
        eil_min_keep_t2t=2,
        eil_warmup_epochs=0,
        eil_interval=1,
        use_erl=False,
        lambda_erl=0.03,
        erl_warmup_epochs=60,
        erl_interval=25,
        erl_min_confidence=0.78,
        erl_sigma2=1.0,
        erl_tau2=1.0,
        erl_initial_budget_ratio=0.003,
        erl_budget_growth_ratio=0.003,
        eil_neighbor_pool_size=10,
        eil_num_clusters=8,
        eil_replacement_strength="random",
    ):
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.k_nn = k_nn
        self.k_similar_tasks = k_similar_tasks
        self.k_similar_models = k_similar_models
        self.edge_weight_alpha = edge_weight_alpha
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.device = torch.device(device) if isinstance(device, str) else device
        self.skip_m2m_edges = skip_m2m_edges
        self.use_eil = bool(use_eil)
        self.lambda_eil = float(lambda_eil)
        self.eil_temperature = float(eil_temperature)
        self.eil_projection_dim = int(eil_projection_dim)
        self.eil_embed_consistency = float(eil_embed_consistency)
        self.eil_edge_dropout = float(eil_edge_dropout)
        self.eil_min_keep_t2t = int(eil_min_keep_t2t)
        self.eil_warmup_epochs = int(eil_warmup_epochs)
        self.eil_interval = max(1, int(eil_interval))
        self.use_erl = bool(use_erl)
        self.lambda_erl = float(lambda_erl)
        self.erl_warmup_epochs = int(erl_warmup_epochs)
        self.erl_interval = max(1, int(erl_interval))
        self.erl_min_confidence = float(erl_min_confidence)
        self.erl_sigma2 = float(erl_sigma2)
        self.erl_tau2 = float(erl_tau2)
        self.erl_initial_budget_ratio = max(0.0, float(erl_initial_budget_ratio))
        self.erl_budget_growth_ratio = max(0.0, float(erl_budget_growth_ratio))
        self.eil_neighbor_pool_size = max(1, int(eil_neighbor_pool_size))
        self.eil_num_clusters = max(1, int(eil_num_clusters))
        self.eil_replacement_strength = str(eil_replacement_strength).lower()
        self.M_train = None
        self.source_task_features = None
        self.target_task_features = None
        self.model_features = None
        self.G_train = None
        self.encoder = None
        self.bilinear = None
        self.proj_head = None
        self.node_dict = None
        self.edge_dict = None
        self._erl_missing_mask = None
        self._erl_added_mask = None
        self._erl_edges = []
        self._erl_observed_edges = 0
        self._erl_max_edges = 0
        self._erl_observed_values = None
        self._erl_posterior_mean = None
        self._eil_clusters = None

    def __str__(self):
        return f"MERIT(layers={self.n_layers}, heads={self.n_heads}, neighbors={self.k_nn})"

    def set_node_features(self, features):
        """Set source, target and model features constructed from source performance."""
        self.source_task_features = np.asarray(features["source_task"], dtype=np.float32)
        self.target_task_features = np.asarray(features["target_task"], dtype=np.float32)
        self.model_features = np.asarray(features["model"], dtype=np.float32)

    def fit(self, M_train, P_train):
        """Train on source metadata and observed performance."""
        n_tasks, n_models = P_train.shape
        self.G_train = self._build_tmg_graph(
            self.source_task_features, self.model_features, P_train
        )
        self.G_train = self.G_train.to(self.device)
        self.node_dict = {ntype: i for i, ntype in enumerate(self.G_train.ntypes)}
        self.edge_dict = {etype: i for i, etype in enumerate(self.G_train.etypes)}
        task_feat_dim = self.source_task_features.shape[1]
        model_feat_dim = self.model_features.shape[1]
        n_inp = max(task_feat_dim, model_feat_dim)
        self.encoder = HeterogeneousGraphTransformer(
            node_dict=self.node_dict,
            edge_dict=self.edge_dict,
            n_inp=n_inp,
            n_hid=self.hid_dim,
            n_out=self.hid_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
        ).to(self.device)
        self.bilinear = nn.Linear(self.hid_dim, self.hid_dim).to(self.device)
        if self.use_eil and self.eil_embed_consistency > 0:
            self.proj_head = nn.Sequential(
                nn.Linear(self.hid_dim, self.hid_dim),
                nn.ReLU(),
                nn.Linear(self.hid_dim, self.eil_projection_dim),
            ).to(self.device)
        params = list(self.encoder.parameters()) + list(self.bilinear.parameters())
        if self.proj_head is not None:
            params += list(self.proj_head.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
        task_feat_tensor = torch.FloatTensor(self.source_task_features).to(self.device)
        model_feat_tensor = torch.FloatTensor(self.model_features).to(self.device)
        if task_feat_dim < n_inp:
            padding = torch.zeros(n_tasks, n_inp - task_feat_dim).to(self.device)
            task_feat_tensor = torch.cat([task_feat_tensor, padding], dim=1)
        if model_feat_dim < n_inp:
            padding = torch.zeros(n_models, n_inp - model_feat_dim).to(self.device)
            model_feat_tensor = torch.cat([model_feat_tensor, padding], dim=1)
        inp = {"task": task_feat_tensor, "model": model_feat_tensor}
        P_target = torch.FloatTensor(P_train).to(self.device)
        if self.use_erl:
            self._prepare_erl_state(P_train)
        self._prepare_eil(M_train)
        best_loss = float("inf")
        patience_counter = 0
        self.encoder.train()
        self.bilinear.train()
        if self.proj_head is not None:
            self.proj_head.train()
        active_graph = self.G_train
        for epoch in range(self.epochs):
            if (
                self.use_erl
                and self._erl_missing_mask is not None
                and (epoch >= self.erl_warmup_epochs)
                and ((epoch - self.erl_warmup_epochs) % self.erl_interval == 0)
            ):
                self.encoder.eval()
                self.bilinear.eval()
                with torch.no_grad():
                    probe = self.encoder(active_graph, inp)
                    probe_task = self.bilinear(probe["task"])
                    probe_pred = torch.mm(probe_task, probe["model"].T).cpu().numpy()
                target_edges = self._erl_target_edge_count(epoch)
                edge_budget = max(0, target_edges - len(self._erl_edges))
                new_edges = self._select_recovered_edges(probe_pred, edge_budget)
                if new_edges:
                    for task_id, model_id, _, _ in new_edges:
                        self._erl_added_mask[task_id, model_id] = True
                    self._erl_edges.extend(new_edges)
                    active_graph = self._graph_with_recovered_edges()
                self.encoder.train()
                self.bilinear.train()
            optimizer.zero_grad(set_to_none=True)
            out_dict = self.encoder(active_graph, inp)
            h_task = out_dict["task"]
            h_model = out_dict["model"]
            h_task_proj = self.bilinear(h_task)
            P_pred = torch.mm(h_task_proj, h_model.T)
            loss = listmle_loss(P_pred, P_target)
            if self.use_erl and (self._erl_edges or self._erl_posterior_mean is not None):
                loss = loss + self.lambda_erl * self._erl_soft_loss(h_task, h_model)
            if (
                self.use_eil
                and epoch >= self.eil_warmup_epochs
                and ((epoch - self.eil_warmup_epochs) % self.eil_interval == 0)
            ):
                ramp = min(
                    1.0, (epoch - self.eil_warmup_epochs + 1) / max(1, self.eil_warmup_epochs)
                )
                eil_weight = self.lambda_eil * ramp
            else:
                eil_weight = 0.0
            if eil_weight > 0:
                G_aug = self._generate_eil_graph(active_graph)
                out_aug = self.encoder(G_aug, inp)
                if self.eil_embed_consistency > 0:
                    original_projection = self.proj_head(h_task)
                    loss_eil_embed = self._infonce_loss(
                        original_projection, self.proj_head(out_aug["task"])
                    )
                else:
                    loss_eil_embed = torch.tensor(0.0, device=P_pred.device)
                loss_eil = self.eil_embed_consistency * loss_eil_embed
                loss = loss + eil_weight * loss_eil
            loss.backward()
            optimizer.step()
            epoch_loss = loss.item()
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
            if eil_weight > 0:
                del G_aug, out_aug, loss_eil_embed, loss_eil
            del out_dict, h_task, h_model, h_task_proj, P_pred, loss
            if (epoch + 1) % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        self.G_train = active_graph
        self.M_train = M_train

    def predict(self, M_test):
        """Predict target candidate scores with the trained graph encoder."""
        target_task_features = self.target_task_features
        M_train_view, M_test_view = self._similarity_features(self.M_train, M_test)
        similarities = cosine_similarity(M_test_view, M_train_view)
        topk_indices = np.argsort(similarities, axis=1)[:, -self.k_nn :]
        neighbor_similarities = np.take_along_axis(similarities, topk_indices, axis=1)
        G_test = self._augment_graph_with_test_nodes(
            target_task_features, topk_indices, neighbor_similarities
        )
        G_test = G_test.to(self.device)
        n_train = self.G_train.number_of_nodes("task")
        n_models = self.G_train.number_of_nodes("model")
        task_feat_dim = target_task_features.shape[1]
        model_feat_dim = self.model_features.shape[1]
        n_inp = max(task_feat_dim, model_feat_dim)
        all_task_features = np.vstack([self.source_task_features, target_task_features])
        task_feat_tensor = torch.FloatTensor(all_task_features).to(self.device)
        model_feat_tensor = torch.FloatTensor(self.model_features).to(self.device)
        if task_feat_dim < n_inp:
            padding = torch.zeros(n_train + n_test, n_inp - task_feat_dim).to(self.device)
            task_feat_tensor = torch.cat([task_feat_tensor, padding], dim=1)
        if model_feat_dim < n_inp:
            padding = torch.zeros(n_models, n_inp - model_feat_dim).to(self.device)
            model_feat_tensor = torch.cat([model_feat_tensor, padding], dim=1)
        inp = {"task": task_feat_tensor, "model": model_feat_tensor}
        self.encoder.eval()
        self.bilinear.eval()
        with torch.no_grad():
            out_dict = self.encoder(G_test, inp)
            h_task_all = out_dict["task"]
            h_model = out_dict["model"]
            h_test = h_task_all[n_train:]
            h_test_proj = self.bilinear(h_test)
            P_hat = torch.mm(h_test_proj, h_model.T)
            P_hat = P_hat.cpu().numpy()
        return P_hat

"""
MYUNGBIN
LATTICE BASE
1. Add get_item_item_graph (Ultra form)
2. Add get_user_user_graph (Ultra form)
3. Multi-modal graph = learned_adj
4. InfoNCE (user / item)    
"""

import os
import random
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_knn_neighbourhood, build_sim, compute_normalized_laplacian


class RADIAN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(RADIAN, self).__init__(config, dataset)

        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.weight_size = config["weight_size"]
        self.knn_k = config["knn_k"]
        self.lambda_coeff = config["lambda_coeff"]
        self.cf_model = config["cf_model"]
        self.n_layers = config["n_layers"]
        self.reg_weight = config["reg_weight"]
        self.cl_weights = config["cl_weights"]
        self.build_item_graph = True

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.norm_adj = self.get_adj_mat()
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)
        self.item_adj = None
        self.learned_adj = None

        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        if config["cf_model"] == "ngcf":
            self.GC_Linear_list = nn.ModuleList()
            self.Bi_Linear_list = nn.ModuleList()
            self.dropout_list = nn.ModuleList()
            dropout_list = config["mess_dropout"]
            for i in range(self.n_ui_layers):
                self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.dropout_list.append(nn.Dropout(dropout_list[i]))

        dataset_path = os.path.abspath(config["data_path"] + config["dataset"])
        image_adj_file = os.path.join(dataset_path, "image_adj_{}.pt".format(self.knn_k))
        text_adj_file = os.path.join(dataset_path, "text_adj_{}.pt".format(self.knn_k))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_neighbourhood(image_adj, topk=self.knn_k)
                image_adj = compute_normalized_laplacian(image_adj)
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
                text_adj = compute_normalized_laplacian(text_adj)
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)

        self.item_item_adj = self.get_item_item_graph(top_k=20).to(self.device)
        self.user_user_adj = self.get_user_user_graph(top_k=20).to(self.device)

    def pre_epoch_processing(self):
        self.build_item_graph = True

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[: self.n_users, self.n_users :] = R
        adj_mat[self.n_users :, : self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_item_item_graph(self, top_k=10):
        # Step 1: co-occurrence matrix A^TA
        item_item_matrix = self.interaction_matrix.transpose().dot(self.interaction_matrix).tocoo()

        row = item_item_matrix.row
        col = item_item_matrix.col
        data = item_item_matrix.data
        num_items = self.interaction_matrix.shape[1]

        # Step 2: calc degree vecotr g_i
        degree = np.array(item_item_matrix.sum(axis=1)).flatten()
        item_item_diag = item_item_matrix.diagonal()

        # Step 3: ignore self-connections
        mask = row != col
        row, col, data = row[mask], col[mask], data[mask]
        g_i = degree[row]
        g_j = degree[col]
        G_ii = item_item_diag[row]

        # Step 4: calc w_{i, j}
        omega = data / (g_i - G_ii + 1e-10) * np.sqrt(g_i / (g_j + 1e-10))

        # Step 5: Top-K
        top_k_graph = defaultdict(list)
        for i, j, w in zip(row, col, omega):
            top_k_graph[i].append((j, w))

        final_rows, final_cols, final_data = [], [], []
        for i, neighbors in top_k_graph.items():
            neighbors = sorted(neighbors, key=lambda x: -x[1])[:top_k]
            for j, w in neighbors:
                final_rows.append(i)
                final_cols.append(j)
                final_data.append(w)

        indices = torch.LongTensor([final_rows, final_cols])  # (2, nnz)
        values = torch.FloatTensor(final_data)  # (nnz,)
        shape = torch.Size([num_items, num_items])  # (N, N)

        item_item_adj = torch.sparse.FloatTensor(indices, values, shape)
        return item_item_adj

    def get_user_user_graph(self, top_k=10):
        # Step 1: co-occurrence matrix AA^T
        user_user_matrix = self.interaction_matrix.dot(self.interaction_matrix.transpose()).tocoo()

        row = user_user_matrix.row
        col = user_user_matrix.col
        data = user_user_matrix.data
        num_users = self.interaction_matrix.shape[0]

        # Step 2: calc degree vecotr g_i
        degree = np.array(user_user_matrix.sum(axis=1)).flatten()
        user_user_diag = user_user_matrix.diagonal()

        # Step 3: ignore self-connections
        mask = row != col
        row, col, data = row[mask], col[mask], data[mask]
        g_i = degree[row]
        g_j = degree[col]
        G_ii = user_user_diag[row]

        # Step 4: calc w_{i, j}
        omega = data / (g_i - G_ii + 1e-10) * np.sqrt(g_i / (g_j + 1e-10))

        # Step 5: Top-K
        top_k_graph = defaultdict(list)
        for i, j, w in zip(row, col, omega):
            top_k_graph[i].append((j, w))

        final_rows, final_cols, final_data = [], [], []
        for i, neighbors in top_k_graph.items():
            neighbors = sorted(neighbors, key=lambda x: -x[1])[:top_k]
            for j, w in neighbors:
                final_rows.append(i)
                final_cols.append(j)
                final_data.append(w)

        indices = torch.LongTensor([final_rows, final_cols])  # (2, nnz)
        values = torch.FloatTensor(final_data)  # (nnz,)
        shape = torch.Size([num_users, num_users])  # (N, N)

        user_user_adj = torch.sparse.FloatTensor(indices, values, shape)
        return user_user_adj

    def forward(self, adj, build_item_graph=False, train=False):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
        if build_item_graph:
            weight = self.softmax(self.modal_weight)

            if self.v_feat is not None:
                self.image_adj = build_sim(image_feats)
                self.image_adj = build_knn_neighbourhood(self.image_adj, topk=self.knn_k)
                learned_adj = self.image_adj
                original_adj = self.image_original_adj
            if self.t_feat is not None:
                self.text_adj = build_sim(text_feats)
                self.text_adj = build_knn_neighbourhood(self.text_adj, topk=self.knn_k)
                learned_adj = self.text_adj
                original_adj = self.text_original_adj
            if self.v_feat is not None and self.t_feat is not None:
                learned_adj = weight[0] * self.image_adj + weight[1] * self.text_adj
                original_adj = weight[0] * self.image_original_adj + weight[1] * self.text_original_adj
            learned_adj = compute_normalized_laplacian(learned_adj)
            self.learned_adj = learned_adj
            if self.item_adj is not None:
                del self.item_adj
            self.item_adj = (1 - self.lambda_coeff) * learned_adj + self.lambda_coeff * original_adj
        else:
            self.item_adj = self.item_adj.detach()
            self.learned_adj = self.learned_adj.detach()
            
        # [modality graph / item id embedding] item-item graph propagation
        h = self.item_id_embedding.weight
        for _ in range(self.n_layers):
            h = torch.mm(self.learned_adj, h)
            
        # [co-interaction graph / id embedding] item-item graph propagation
        ii_embeddings = self.item_id_embedding.weight
        for _ in range(self.n_layers):
            ii_embeddings = torch.sparse.mm(self.item_item_adj, ii_embeddings)

        # [co-interaction graph / id embedding] user-user graph propagation
        uu_embeddings = self.user_embedding.weight
        for _ in range(self.n_layers):
            uu_embeddings = torch.sparse.mm(self.user_user_adj, uu_embeddings)

        if self.cf_model == "lightgcn":
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            if train:
                return u_g_embeddings, i_g_embeddings, h, ii_embeddings, uu_embeddings
            return u_g_embeddings, i_g_embeddings

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1.0 / 2 * (users**2).sum() + 1.0 / 2 * (pos_items**2).sum() + 1.0 / 2 * (neg_items**2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))  # in-batch negative sampling
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, mm_embeddings, ii_embeddings, uu_embeddings = self.forward(
            self.norm_adj, build_item_graph=self.build_item_graph, train=True
        )
        self.build_item_graph = False

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
        )
        infoNCE_loss = self.InfoNCE(mm_embeddings[pos_items], ii_embeddings[pos_items], 0.2) + self.InfoNCE(ua_embeddings[users], uu_embeddings[users], 0.2)

        return batch_mf_loss + batch_emb_loss + batch_reg_loss + (self.cl_weights * infoNCE_loss)

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj, build_item_graph=True)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores


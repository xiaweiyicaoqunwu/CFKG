import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import math

class RGCN_RotatE(nn.Module):
    def __init__(self, num_nodes, num_rels, hidden_dim, 
                 pretrained_node_emb=None, pretrained_rel_emb=None, # 新增预训练参数
                 num_bases=2, dropout_rate=0.3, margin=9.0):
        super(RGCN_RotatE, self).__init__()
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.hidden_dim = hidden_dim 
        self.margin = margin
        
        # 1. 节点 Embedding 初始化
        if pretrained_node_emb is not None:
            # 大模型生成的 256 维嵌入直接作为实部+虚部 (2 * hidden_dim)
            self.node_emb = nn.Embedding.from_pretrained(pretrained_node_emb, freeze=False)
            input_dim = pretrained_node_emb.shape[1]
        else:
            self.node_emb = nn.Embedding(num_nodes, 2 * hidden_dim)
            nn.init.xavier_uniform_(self.node_emb.weight)
            input_dim = 2 * hidden_dim
        
        # 2. R-GCN 编码器 (输入维度根据预训练向量动态调整)
        self.rgcn1 = RGCNConv(input_dim, 2 * hidden_dim, num_rels, num_bases=num_bases)
        self.rgcn2 = RGCNConv(2 * hidden_dim, 2 * hidden_dim, num_rels, num_bases=num_bases)
        
        # 3. Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # 4. 关系 Embedding 初始化
        if pretrained_rel_emb is not None:
            # 注意：RotatE 的关系是相位 theta，范围在 [-pi, pi]
            # 大模型生成的向量是稠密向量，需要通过 Tanh 映射到 [-1, 1] 再乘以 pi
            # 这里的关系嵌入维度应为 hidden_dim
            rel_data = pretrained_rel_emb
            # 如果大模型返回维度不对，可以取其前 hidden_dim 位或进行线性变换
            if rel_data.shape[1] > hidden_dim:
                rel_data = rel_data[:, :hidden_dim]
            
            # 将通用嵌入映射为相位
            rel_phase = torch.tanh(rel_data) * math.pi
            self.rel_emb = nn.Embedding.from_pretrained(rel_phase, freeze=False)
        else:
            self.rel_emb = nn.Embedding(num_rels, hidden_dim)
            nn.init.uniform_(self.rel_emb.weight, -math.pi, math.pi)

    def forward(self, edge_index, edge_type, h_idx, r_idx, t_idx):
        # --- Encoder: R-GCN ---
        x = self.node_emb.weight
        
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.rgcn2(x, edge_index, edge_type)
        
        # --- Decoder: RotatE Scoring ---
        head_emb = x[h_idx]
        tail_emb = x[t_idx]
        
        # 提取关系相位 (Batch, hidden_dim)
        relation_phases = self.rel_emb(r_idx)
        
        return self.rotate_score(head_emb, relation_phases, tail_emb)

    def rotate_score(self, h, r_phase, t):
        # 1. 拆分实体嵌入为实部和虚部
        h_re, h_im = torch.chunk(h, 2, dim=-1)
        t_re, t_im = torch.chunk(t, 2, dim=-1)
        
        # 2. 将相位转换为复数 (欧拉公式)
        r_re = torch.cos(r_phase)
        r_im = torch.sin(r_phase)
        
        # 3. 计算旋转 (Hadamard 积)
        h_prime_re = h_re * r_re - h_im * r_im
        h_prime_im = h_re * r_im + h_im * r_re
        
        # 4. 计算 L2 距离
        score_re = h_prime_re - t_re
        score_im = h_prime_im - t_im
        
        score = torch.stack([score_re, score_im], dim=0)
        dist = score.norm(p=2, dim=0).sum(dim=-1) 
        
        return self.margin - dist
    
    def get_full_embeddings(self, edge_index, edge_type):
        """推理工具函数"""
        x = self.node_emb.weight
        x = F.relu(self.rgcn1(x, edge_index, edge_type))
        x = self.dropout(x)
        x = self.rgcn2(x, edge_index, edge_type)
        return x
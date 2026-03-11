import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class RGCN_DistMult(nn.Module):
    def __init__(self, num_nodes, num_rels, hidden_dim, pretrained_node_emb=None, pretrained_rel_emb=None, num_bases=2, dropout_rate=0.3):
        super(RGCN_DistMult, self).__init__()
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.hidden_dim = hidden_dim 
        
        # 1. 节点 Embedding
        if pretrained_node_emb is not None:
            # 确保传入的 Tensor 维度与 hidden_dim * 2 匹配
            # 如果不匹配，你需要在这里加一个 Linear 层做投影，或者调整 config 中的 hidden_dim
            self.node_emb = nn.Embedding.from_pretrained(pretrained_node_emb, freeze=False)
            print("Loaded pretrained node embeddings.")
        else:
            self.node_emb = nn.Embedding(num_nodes, 2 * hidden_dim)
            nn.init.xavier_uniform_(self.node_emb.weight)
        
        # 2. R-GCN 编码器
        # 输入输出维度改为 hidden_dim
        input_dim = self.node_emb.weight.shape[1]
        self.rgcn1 = RGCNConv(input_dim, 2 * hidden_dim, num_rels, num_bases=num_bases)
        self.rgcn2 = RGCNConv(2 * hidden_dim, 2 * hidden_dim, num_rels, num_bases=num_bases)
        
        # 3. Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # 4. 关系 Embedding
        if pretrained_rel_emb is not None:
            self.rel_emb = nn.Embedding.from_pretrained(pretrained_rel_emb, freeze=False)
            print("Loaded pretrained relation embeddings.")
        else:
            self.rel_emb = nn.Embedding(num_rels, 2 * hidden_dim)
            nn.init.xavier_uniform_(self.rel_emb.weight)
        

    def forward(self, edge_index, edge_type, h_idx, r_idx, t_idx):
        # --- Encoder: R-GCN ---
        x = self.node_emb.weight
        
        # 第一层卷积 + 激活 + Dropout
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第二层卷积
        x = self.rgcn2(x, edge_index, edge_type)
        # 此时 x 的维度是 (num_nodes, hidden_dim)
        
        # --- Decoder: DistMult Scoring ---
        head_emb = x[h_idx]            # (Batch, hidden_dim)
        tail_emb = x[t_idx]            # (Batch, hidden_dim)
        relation_emb = self.rel_emb(r_idx) # (Batch, hidden_dim)
        
        return self.distmult_score(head_emb, relation_emb, tail_emb)

    def distmult_score(self, h, r, t):
        """
        DistMult Score Function: <h, r, t>
        计算对应元素的乘积，然后沿最后一个维度求和。
        """
        # h, r, t shape: [Batch, hidden_dim]
        # element-wise multiply: h * r * t
        score = h * r * t
        
        # sum over embedding dimension
        return score.sum(dim=-1)

    def get_full_embeddings(self, edge_index, edge_type):
        """推理时使用，保持与训练相同的结构"""
        x = self.node_emb.weight
        x = F.relu(self.rgcn1(x, edge_index, edge_type))
        x = self.dropout(x)
        x = self.rgcn2(x, edge_index, edge_type)
        return x
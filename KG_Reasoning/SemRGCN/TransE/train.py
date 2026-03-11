import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import pickle
import numpy as np
import os

# 引入模型
from model import RGCN_TransE

# --- 配置 ---
DATA_DIR = "./TransE/data"
HIDDEN_DIM = 128
BATCH_SIZE = 1024
EPOCHS = 100
LR = 0.005 # 通常 Self-Adv Loss 需要稍微调低一点 LR 或细心调参
WEIGHT_DECAY = 1e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 新增/修改的超参数 ===
NEG_K = 64              # 负采样数量 (Self-Adv 通常需要较多的负样本，如 32, 64)
ADVERSARIAL_TEMP = 1.0  # 自对抗温度系数 alpha (控制对 Hard Negative 的关注度)


def load_dataset():
    with open(os.path.join(DATA_DIR, 'dataset.pkl'), 'rb') as f:
        return pickle.load(f)


def train():
    data = load_dataset()
    num_nodes = data['num_nodes']
    num_rels = data['num_rels']
    train_triplets = data['train']

    # --- 加载预训练嵌入 ---
    node_emb_path = os.path.join(DATA_DIR, 'pretrained_node_emb.npy')
    rel_emb_path = os.path.join(DATA_DIR, 'pretrained_rel_emb.npy')

    pretrained_node_emb = None
    pretrained_rel_emb = None
    
    if os.path.exists(node_emb_path) and os.path.exists(rel_emb_path):
        print("发现预训练嵌入，正在加载...")
        p_node = np.load(node_emb_path)
        p_rel = np.load(rel_emb_path)
        
        # 转换为 FloatTensor
        pretrained_node_emb = torch.FloatTensor(p_node).to(DEVICE)
        pretrained_rel_emb = torch.FloatTensor(p_rel).to(DEVICE)
        
        # 检查维度匹配
        expected_dim = 2 * HIDDEN_DIM
        if pretrained_node_emb.shape[1] != expected_dim:
            print(f"警告: 预训练维度 ({pretrained_node_emb.shape[1]}) 与模型设定 ({expected_dim}) 不匹配!")
            print("建议修改 HIDDEN_DIM 或重新生成嵌入。")
            # 这里你可以选择报错退出，或者加一个线性层处理，这里简单起见假设用户会改对
    else:
        print("未找到预训练嵌入，将使用随机初始化。")
    
    # --- 图结构 ---
    src = torch.tensor(train_triplets[:, 0], dtype=torch.long)
    dst = torch.tensor(train_triplets[:, 2], dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0).to(DEVICE)
    edge_type = torch.tensor(train_triplets[:, 1], dtype=torch.long).to(DEVICE)
    
    # --- 模型 ---
    model = RGCN_TransE(num_nodes, num_rels, HIDDEN_DIM, pretrained_node_emb=pretrained_node_emb, pretrained_rel_emb=pretrained_rel_emb).to(DEVICE)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print(f"开始训练... 设备: {DEVICE}")
    print(f"图结构: {num_nodes} 节点, {len(train_triplets)} 边")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        idxs = np.random.permutation(len(train_triplets))
        
        for i in range(0, len(train_triplets), BATCH_SIZE):
            batch_idx = idxs[i:i+BATCH_SIZE]
            batch = train_triplets[batch_idx]
            
            # (Batch_Size,)
            h = torch.tensor(batch[:, 0], dtype=torch.long).to(DEVICE)
            r = torch.tensor(batch[:, 1], dtype=torch.long).to(DEVICE)
            t = torch.tensor(batch[:, 2], dtype=torch.long).to(DEVICE)
            
            # =========================================================
            # 1. 计算正样本得分
            # =========================================================
            # pos_scores shape: [Batch_Size]
            # model 返回的是 (margin - distance)
            pos_scores = model(edge_index, edge_type, h, r, t)
            
            # =========================================================
            # 2. 负采样 (Batch_Size, NEG_K)
            # =========================================================
            current_batch_size = len(h)
            
            neg_t = torch.randint(
                0, num_nodes,
                (current_batch_size, NEG_K),
                device=DEVICE
            )
            
            # 扩展 h, r 以匹配负样本维度: [Batch, NEG_K]
            h_expand = h.unsqueeze(1).expand(-1, NEG_K).reshape(-1)
            r_expand = r.unsqueeze(1).expand(-1, NEG_K).reshape(-1)
            neg_t_flat = neg_t.reshape(-1)
            
            # 计算所有负样本的得分
            # neg_scores shape: [Batch * NEG_K] -> view -> [Batch, NEG_K]
            neg_scores = model(edge_index, edge_type, h_expand, r_expand, neg_t_flat)
            neg_scores = neg_scores.view(current_batch_size, NEG_K)

            # =========================================================
            # 3. 计算 Self-adversarial Loss
            # =========================================================

            pos_loss = -F.logsigmoid(pos_scores).mean()

            neg_weights = F.softmax(neg_scores * ADVERSARIAL_TEMP, dim=1).detach()
            
            neg_loss_individual = -F.logsigmoid(-neg_scores)
            
            # 加权求和: 对 dim=1 (K个负样本) 求和，然后对 Batch 求平均
            neg_loss = (neg_weights * neg_loss_individual).sum(dim=1).mean()
            
            # 总 Loss
            loss = (pos_loss + neg_loss) / 2
            
            # =========================================================
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # 学习率调度
        scheduler.step(total_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | LR: {current_lr:.6f}")
    
    torch.save(model.state_dict(), os.path.join(DATA_DIR, 'model_weights.pth'))
    print("模型已保存。")

if __name__ == "__main__":
    train()
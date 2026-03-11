import torch
import pickle
import numpy as np
import os
from torch.utils.data import DataLoader
from model import RGCN_ComplEx


# --- 配置 ---
DATA_DIR = "./ComplEx/data"
HIDDEN_DIM = 128
BATCH_SIZE = 100  # 评估时显存占用大，Batch设小一点
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_resources():
    print("正在加载数据和模型...")
    with open(os.path.join(DATA_DIR, 'dataset.pkl'), 'rb') as f:
        data = pickle.load(f)
    
    # 准备 R-GCN 消息传递用的图结构 (使用全量训练数据)
    # 注意：在Transductive Setting下，通常用训练集构建图来推断测试集
    src = torch.tensor(data['train'][:, 0], dtype=torch.long)
    dst = torch.tensor(data['train'][:, 2], dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0).to(DEVICE)
    edge_type = torch.tensor(data['train'][:, 1], dtype=torch.long).to(DEVICE)

    # 加载模型
    model = RGCN_ComplEx(data['num_nodes'], data['num_rels'], HIDDEN_DIM).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'model_weights.pth'), map_location=DEVICE))
    model.eval()
    
    return model, edge_index, edge_type, data

def build_filter_dict(data):
    """
    构建过滤字典：记录 (h, r) 对应的所有真实 t。
    用于 Filtered MRR 计算，防止将其他正确的尾实体误判为错误预测。
    """
    all_triplets = np.concatenate([data['train'], data['valid'], data['test']], axis=0)
    filter_dict = {}
    
    for h, r, t in all_triplets:
        key = (h, r)
        if key not in filter_dict:
            filter_dict[key] = []
        filter_dict[key].append(t)
        
    # 转为集合以加速查找
    for key in filter_dict:
        filter_dict[key] = set(filter_dict[key])
        
    return filter_dict

def evaluate(model, edge_index, edge_type, test_triplets, num_nodes, filter_dict):
    print(f"开始评估 {len(test_triplets)} 条测试数据...")
    
    hits1, hits3, hits10, mrr = 0, 0, 0, 0
    total_samples = len(test_triplets)
    
    # 预先计算所有节点的 Embedding (R-GCN 只需要跑一次)
    with torch.no_grad():
        full_entity_emb = model.get_full_embeddings(edge_index, edge_type)
    
    # 转换为 DataLoader 批量处理
    test_loader = DataLoader(test_triplets, batch_size=BATCH_SIZE)
    
    for batch in test_loader:
        # batch shape: [batch_size, 3]
        h_batch = batch[:, 0].to(DEVICE)
        r_batch = batch[:, 1].to(DEVICE)
        t_batch = batch[:, 2].to(DEVICE)
        
        batch_size_curr = len(h_batch)
        
        # 1. 准备候选集：我们需要对每个样本，计算它和所有实体的分数
        # Head: [batch, 1, dim]
        heads = full_entity_emb[h_batch].unsqueeze(1)
        # Rel: [batch, 1, dim]
        rels = model.rel_emb(r_batch).unsqueeze(1)
        # Tail (All): [1, num_nodes, dim] -> [batch, num_nodes, dim]
        # 注意显存：如果 num_nodes 很大，这里需要分块计算
        tails = full_entity_emb.unsqueeze(0).expand(batch_size_curr, -1, -1)
        
        # 2. 计算 Batch 内每个样本对所有候选实体的分数
        # shape: [batch, num_nodes]
        # 扩展维度以匹配
        heads_exp = heads.expand(-1, num_nodes, -1)
        rels_exp = rels.expand(-1, num_nodes, -1)
        
        with torch.no_grad():
            all_scores = model.complex_score(heads_exp, rels_exp, tails)
            
        # 3. 过滤 (Filtered Setting)
        # 对于 batch 中的每个样本，将除了 ground truth 之外的其他已知正确答案的分数设为 -inf
        all_scores = all_scores.cpu().numpy()
        h_cpu = h_batch.cpu().numpy()
        r_cpu = r_batch.cpu().numpy()
        t_cpu = t_batch.cpu().numpy()
        
        for i in range(batch_size_curr):
            h, r, t = h_cpu[i], r_cpu[i], t_cpu[i]
            # 获取所有真实的尾实体
            true_tails = filter_dict.get((h, r), set())
            
            # 将这些真实尾实体的分数屏蔽掉 (设为极小值)
            # 注意：不能屏蔽当前的 t (target)，否则无法算出排名
            mask_indices = list(true_tails)
            # 这里的逻辑是：先把所有 true_tails 设为 -inf，再把 target 设回原始分数
            # 或者更简单：只把 (true_tails - target) 设为 -inf
            
            filter_indices = list(true_tails - {t})
            if filter_indices:
                all_scores[i, filter_indices] = -100000.0

        # 4. 计算排名
        # argsort 是从小到大，我们取反做从大到小，或者直接argsort(-scores)
        # 找到目标 t 的排名
        sorted_indices = np.argsort(-all_scores, axis=1)
        
        for i in range(batch_size_curr):
            target_t = t_cpu[i]
            # np.where 返回的是 tuple
            rank = np.where(sorted_indices[i] == target_t)[0][0] + 1
            
            mrr += 1.0 / rank
            if rank <= 1: hits1 += 1
            if rank <= 3: hits3 += 1
            if rank <= 10: hits10 += 1

    # 计算平均值
    print("\n" + "="*30)
    print("评估结果 (Filtered Metrics)")
    print("="*30)
    print(f"MRR     : {mrr / total_samples:.4f}")
    print(f"Hits@1  : {hits1 / total_samples:.4f}")
    print(f"Hits@3  : {hits3 / total_samples:.4f}")
    print(f"Hits@10 : {hits10 / total_samples:.4f}")
    print("="*30)

if __name__ == "__main__":
    model, edge_index, edge_type, data = load_resources()
    
    # 构建过滤字典
    print("构建过滤字典...")
    filter_dict = build_filter_dict(data)
    
    # 运行评估
    evaluate(model, edge_index, edge_type, data['test'], data['num_nodes'], filter_dict)
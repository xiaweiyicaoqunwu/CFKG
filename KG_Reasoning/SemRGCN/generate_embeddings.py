import os
import pickle
import numpy as np
import torch
from openai import OpenAI
from tqdm import tqdm
from collections import defaultdict

# --- 配置 ---
DATA_DIR = "./data"
DATASET_PATH = os.path.join(DATA_DIR, 'dataset.pkl')
SAVE_NODE_EMB_PATH = os.path.join(DATA_DIR, 'pretrained_node_emb.npy')
SAVE_REL_EMB_PATH = os.path.join(DATA_DIR, 'pretrained_rel_emb.npy')

# API 配置
API_KEY = ""
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIM = 256 

# 描述生成限制（防止 prompt 过长）
MAX_NEIGHBORS = 300     # 每个关系最多列出多少个一跳邻居
MAX_SUB_NEIGHBORS = 150  # 每个一跳邻居最多列出多少个二跳邻居

def load_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"未找到数据集: {DATASET_PATH}，请先运行 data_loader.py")
    with open(DATASET_PATH, 'rb') as f:
        return pickle.load(f)

def build_graph(data):
    """构建邻接表，过滤掉 inverse_ 关系"""
    triplets = np.concatenate([data['train'], data['valid'], data['test']], axis=0)
    id2name = data['id2name']
    
    # 建立 id -> relation_name 映射
    id2rel = {v: k for k, v in data['relation2id'].items()}
    
    # 构建邻接表: head -> {relation_name: [tail_ids]}
    adj = defaultdict(lambda: defaultdict(list))
    
    print("正在构建图结构以生成描述...")
    for h, r, t in tqdm(triplets):
        r_name = id2rel[r]
        # 【关键】过滤反向关系
        if r_name.startswith("inverse_"):
            continue
        adj[h][r_name].append(t)
        
    return adj, id2name, id2rel

def generate_node_description(node_id, adj, id2name):
    """
    生成节点的自然语言描述
    格式: head rel1 tail1(which rel2 sub_tail...), tail2. head rel3 ...
    """
    node_name = str(id2name.get(node_id, f"Entity_{node_id}"))
    
    # 如果该节点没有出边（叶子节点），直接返回名字
    if node_id not in adj:
        return f"{node_name}"
    
    sentences = []
    
    # 遍历该节点的所有关系 (一跳)
    for r_name, tail_list in adj[node_id].items():
        # 限制一跳邻居数量
        current_tails = tail_list[:MAX_NEIGHBORS]
        
        tail_descs = []
        for t_id in current_tails:
            t_name = str(id2name.get(t_id, f"Entity_{t_id}"))
            
            # --- 处理二跳 ---
            sub_descs = []
            if t_id in adj: # 如果这个一跳邻居还有自己的邻居
                for sub_r, sub_tails in adj[t_id].items():
                    # 限制二跳邻居数量
                    sub_tail_names = [str(id2name.get(st, f"E_{st}")) for st in sub_tails[:MAX_SUB_NEIGHBORS]]
                    if sub_tail_names:
                        sub_descs.append(f"{sub_r} {', '.join(sub_tail_names)}")
            
            # 组合一跳和二跳描述
            if sub_descs:
                # 例如: Ethanol(which activate alcohol, medicinal)
                tail_str = f"{t_name}(which {', '.join(sub_descs)})"
            else:
                tail_str = t_name
            
            tail_descs.append(tail_str)
        
        # 组合当前关系的句子
        # 例如: grapefruit hasCompound Ethanol(...), Acetone(...)
        sentence = f"{node_name} {r_name} {', '.join(tail_descs)}"
        sentences.append(sentence)
    
    full_desc = ". ".join(sentences) + "."
    return full_desc

def get_embeddings(texts, client):
    """调用 API 批量获取嵌入"""
    # 阿里云 text-embedding-v4 支持 batch，但要注意单次请求上限
    # 这里我们按小批次处理
    batch_size = 4 
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Calling Embedding API"):
        batch_texts = texts[i : i + batch_size]
        try:
            resp = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch_texts,
                dimensions=EMBEDDING_DIM
            )
            # 保证按顺序添加
            batch_emb = [item.embedding for item in resp.data]
            embeddings.extend(batch_emb)
        except Exception as e:
            print(f"\nAPI Error at batch {i}: {e}")
            # 简单的错误处理：如果失败，填入零向量或重试
            # 这里填入零向量防止程序崩溃
            embeddings.extend([[0.0] * EMBEDDING_DIM] * len(batch_texts))
            
    return np.array(embeddings, dtype=np.float32)

def main():
    # 1. 初始化
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    data = load_data()
    adj, id2name, id2rel = build_graph(data)
    
    num_nodes = data['num_nodes']
    num_rels = data['num_rels']
    
    # 2. 生成节点描述并获取嵌入
    print(f"正在为 {num_nodes} 个节点生成文本描述...")
    node_texts = []
    for i in range(num_nodes):
        desc = generate_node_description(i, adj, id2name)
        node_texts.append(desc)
    
    # 打印几个示例看看效果
    print("\n--- 节点描述示例 ---")
    print(f"Node 0: {node_texts[0][:200]}...")
    print(f"Node 1: {node_texts[1][:200]}...")
    print("--------------------\n")
    
    print("开始调用 API 生成节点嵌入...")
    node_embeddings = get_embeddings(node_texts, client)
    
    # 3. 生成关系描述并获取嵌入
    # 关系描述直接使用关系名称
    print(f"\n正在为 {num_rels} 个关系生成嵌入...")
    rel_texts = []
    # relation2id 是 name->id，我们需要按 id 0,1,2... 的顺序排列 name
    sorted_rels = sorted(data['relation2id'].items(), key=lambda x: x[1])
    rel_texts = [item[0] for item in sorted_rels] # item[0] is name
    
    # 对关系名做一点预处理，比如把 'inverse_hasCompound' 变成 'inverse hasCompound' 更好理解
    rel_texts_clean = [t.replace('_', ' ') for t in rel_texts]
    
    rel_embeddings = get_embeddings(rel_texts_clean, client)
    
    # 4. 保存
    np.save(SAVE_NODE_EMB_PATH, node_embeddings)
    np.save(SAVE_REL_EMB_PATH, rel_embeddings)
    
    print(f"\n处理完成！")
    print(f"节点嵌入已保存至: {SAVE_NODE_EMB_PATH} (Shape: {node_embeddings.shape})")
    print(f"关系嵌入已保存至: {SAVE_REL_EMB_PATH} (Shape: {rel_embeddings.shape})")

if __name__ == "__main__":
    if not API_KEY:
        print("错误: 请先设置 DASHSCOPE_API_KEY 环境变量或在代码中填写。")
    else:
        main()
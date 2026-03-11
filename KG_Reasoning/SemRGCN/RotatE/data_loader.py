import pandas as pd
import numpy as np
import pickle
from py2neo import Graph
from sklearn.model_selection import train_test_split
import os

# --- 配置 ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = ""
DATA_DIR = "" #数据集保存的文件夹

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def load_data_from_neo4j():
    print("正在连接 Neo4j...")
    graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # 获取所有三元组
    query = """
    MATCH (h)-[r]->(t)
    RETURN 
        labels(h)[0] + '_' + toString(id(h)) as head_id, 
        h.name as head_name,
        labels(h)[0] as head_type,
        type(r) as relation, 
        labels(t)[0] + '_' + toString(id(t)) as tail_id,
        t.name as tail_name,
        labels(t)[0] as tail_type
    """
    
    print("正在执行查询...")
    data = graph.run(query).to_data_frame()
    print(f"共获取 {len(data)} 条三元组。")
    return data

def process_data(df):
    # 1. 构建实体和关系映射
    entities = set(df['head_id']).union(set(df['tail_id']))
    relations = set(df['relation'])
    
    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}
    
    # 保存名称和类型映射用于展示
    id2name = {}
    id2type = {}
    
    for _, row in df.iterrows():
        h_idx = entity2id[row['head_id']]
        t_idx = entity2id[row['tail_id']]
        
        # 记录Head信息
        if h_idx not in id2name:
            id2name[h_idx] = row['head_name'] if row['head_name'] else row['head_id']
            id2type[h_idx] = row['head_type']
            
        # 记录Tail信息
        if t_idx not in id2name:
            id2name[t_idx] = row['tail_name'] if row['tail_name'] else row['tail_id']
            id2type[t_idx] = row['tail_type']

    # 2. 转换数据为 numpy 数组 [Head, Relation, Tail]
    triplets = []
    for _, row in df.iterrows():
        h = entity2id[row['head_id']]
        r = relation2id[row['relation']]
        t = entity2id[row['tail_id']]
        triplets.append([h, r, t])
        
    triplets = np.array(triplets)
    
    # 3. 划分数据集 (Train/Valid/Test)
    train_data, test_data = train_test_split(triplets, test_size=0.2, random_state=42)
    valid_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)
    
    print(f"训练集: {len(train_data)}, 验证集: {len(valid_data)}, 测试集: {len(test_data)}")
    
    # 4. 保存数据包
    data_pkg = {
        'train': train_data,
        'valid': valid_data,
        'test': test_data,
        'entity2id': entity2id,
        'relation2id': relation2id,
        'id2name': id2name,
        'id2type': id2type,
        'num_nodes': len(entity2id),
        'num_rels': len(relation2id)
    }
    
    with open(os.path.join(DATA_DIR, 'dataset.pkl'), 'wb') as f:
        pickle.dump(data_pkg, f)
    print("数据处理完成。")

if __name__ == "__main__":
    df = load_data_from_neo4j()

    print("正在添加逆关系...")
    inverse_df = df.copy()
    # 交换头尾实体
    inverse_df['head_id'] = df['tail_id']
    inverse_df['head_name'] = df['tail_name']
    inverse_df['head_type'] = df['tail_type']
    inverse_df['tail_id'] = df['head_id']
    inverse_df['tail_name'] = df['head_name']
    inverse_df['tail_type'] = df['head_type']
    # 为关系添加逆关系前缀
    inverse_df['relation'] = 'inverse_' + df['relation']
    # 合并原始数据和逆关系数据
    df = pd.concat([df, inverse_df], ignore_index=True)
    print(f"添加逆关系后，总三元组数量: {len(df)}")

    process_data(df)
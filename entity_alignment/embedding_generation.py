import torch
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from config import *

def generate_embeddings():
    # 加载节点名称
    node_names = []
    with open(NODES_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            node = json.loads(line)
            node_names.append(node["name"].strip() or "未知实体")
    
    # 初始化嵌入模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    model.max_seq_length = 128  # 限制序列长度节省内存
    
    # 分批生成嵌入
    embeddings = []
    for i in tqdm(range(0, len(node_names), BATCH_SIZE), desc="Generating embeddings"):
        batch = node_names[i:i+BATCH_SIZE]
        batch_embeddings = model.encode(
            batch,
            batch_size=32,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        embeddings.append(batch_embeddings.cpu())
    
    # 合并并保存
    final_embeddings = torch.cat(embeddings)
    torch.save(final_embeddings, EMBEDDINGS_PATH)
    print(f"Embeddings saved to {EMBEDDINGS_PATH} | Shape: {final_embeddings.shape}")

if __name__ == "__main__":
    generate_embeddings()
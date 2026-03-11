import torch
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from config import *


def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0


def generate_candidates():
    # === 载入节点与分组 ===
    nodes = []
    label_groups = defaultdict(list)
    with open(NODES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            node = json.loads(line)
            nodes.append(node)
            label_groups[node["label"]].append(node["node_id"])

    # === 加载嵌入 ===
    embeddings = torch.load(EMBEDDINGS_PATH).numpy()
    node_id_to_idx = {node["node_id"]: idx for idx, node in enumerate(nodes)}

    # === 邻居集合预处理 ===
    neighbor_sets = {}
    for node in nodes:
        neighbor_ids = {nb["id"] for nb in node["neighbors"]}
        neighbor_sets[node["node_id"]] = neighbor_ids

    # === 输出文件 ===
    with open(CANDIDATES_PATH, "w", encoding="utf-8") as f_out:

        # 遍历每个标签组
        for label, node_ids in tqdm(label_groups.items(), desc="Processing label groups"):
            if len(node_ids) < 2:
                continue

            # 当前标签下所有实体索引
            indices = [node_id_to_idx[nid] for nid in node_ids]
            group_embeddings = embeddings[indices]      # shape: (n, dim)

            n = len(node_ids)

            # ==== 对每个实体计算 Top-K ====
            for i, node_id in enumerate(node_ids):

                this_vec = group_embeddings[i:i+1]      # (1, dim)
                candidates_scores = []                  # (candidate_idx, sim)

                # 逐块计算
                for start in range(0, n, BATCH_SIZE):
                    end = min(start + BATCH_SIZE, n)
                    block = group_embeddings[start:end]

                    sims = cosine_similarity(this_vec, block)[0]  # shape: (block_size,)

                    # 存储为 (global_idx_in_group, sim)
                    for offset, score in enumerate(sims):
                        global_j = start + offset
                        # 跳过自己
                        if global_j == i:
                            continue
                        candidates_scores.append((global_j, score))

                # 全局 Top-K（按名称相似度）
                candidates_scores.sort(key=lambda x: x[1], reverse=True)
                top_scores = candidates_scores[:TOP_K]

                # 计算加权得分并保留过滤结果
                final_candidates = []
                this_neighbors = neighbor_sets[node_id]

                for j, name_sim in top_scores:
                    candidate_id = node_ids[j]
                    struct_sim = jaccard_similarity(
                        this_neighbors,
                        neighbor_sets[candidate_id]
                    )

                    combined = (
                        NAME_WEIGHT * float(name_sim) +
                        STRUCTURE_WEIGHT * struct_sim
                    )

                    if combined >= SIMILARITY_THRESHOLD:
                        final_candidates.append({
                            "candidate_id": candidate_id,
                            "name": nodes[node_id_to_idx[candidate_id]]["name"],
                            "score": float(combined)
                        })

                # 写入结果
                if final_candidates:
                    final_candidates.sort(key=lambda x: x["score"], reverse=True)

                    result = {
                        "main_node": {
                            "id": node_id,
                            "name": nodes[node_id_to_idx[node_id]]["name"],
                            "label": label,
                        },
                        "candidates": final_candidates
                    }

                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Candidates saved to {CANDIDATES_PATH}")


if __name__ == "__main__":
    generate_candidates()

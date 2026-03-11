import json
import os

def load_processed_ids(results_path):
    """加载已处理的节点ID，支持断点续传"""
    if not os.path.exists(results_path):
        return set()
    
    processed = set()
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                processed.add(data["main_id"])
            except:
                continue
    return processed
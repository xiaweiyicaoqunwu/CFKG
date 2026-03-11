import json
from py2neo import Graph
from tqdm import tqdm
from config import *

def extract_nodes_and_neighbors():
    graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # 获取总节点数
    total_nodes = graph.run("MATCH (n) RETURN count(n) AS total").evaluate()
    print(f"Total nodes to process: {total_nodes}")

    with open(NODES_PATH, 'w', encoding='utf-8') as f_out:
        # 分批处理避免内存溢出
        for skip in tqdm(range(0, total_nodes, BATCH_SIZE), desc="Processing nodes"):
            query = """
            MATCH (n)
            WITH n SKIP $skip LIMIT $limit
            OPTIONAL MATCH (n)-[r]-(neighbor)
            RETURN 
                id(n) AS node_id,
                n.name AS name,
                labels(n)[0] AS label,
                collect(DISTINCT {id: id(neighbor), name: neighbor.name}) AS neighbors
            """
            results = graph.run(query, skip=skip, limit=BATCH_SIZE)
            
            for record in results:
                node_data = {
                    "node_id": record["node_id"],
                    "name": record["name"] or "",
                    "label": record["label"],
                    "neighbors": [
                        {"id": nb["id"], "name": nb["name"]} 
                        for nb in record["neighbors"] if nb["name"]
                    ]
                }
                f_out.write(json.dumps(node_data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    extract_nodes_and_neighbors()
    print(f"Node data saved to {NODES_PATH}")
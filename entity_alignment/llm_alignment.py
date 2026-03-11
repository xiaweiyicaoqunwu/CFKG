import os
import json
import time
from openai import OpenAI
from tqdm import tqdm
from config import *
from utils import load_processed_ids

def call_llm(main_name, candidate_names):
    """调用大模型进行实体对齐判断"""
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=BASE_URL
    )
    
    system_prompt = (
        "You are an expert in citrus flavor compounds, specializing in compound nomenclature, citrus breeding, and flavor characteristics."
        "Strictly determine whether the primary entity and candidate entity represent the same research subject."
        "Return only the result in JSON format: if aligned entities exist, return {'aligned_entities': [candidate entity name 1, ...]};"
        "The returned entity name must match the input entity name."
        "If no aligned entities exist, return the string 'None'. No additional explanations are allowed."
    )
    
    user_prompt = (
        f"Main entity: '{main_name}'\n"
        f"Candidate entity list: {candidate_names}\n"
        "Please strictly return the result as required:"
    )
    
    for attempt in range(3):  # 重试机制
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # 降低随机性
                max_tokens=200,
                timeout=30
            )
            response = completion.choices[0].message.content.strip()
            return response
        except Exception as e:
            print(f"API call failed (attempt {attempt+1}/3): {str(e)}")
            time.sleep(2 ** attempt)  # 指数退避
    return None

def process_alignment():
    processed_ids = load_processed_ids(ALIGNMENT_RESULTS_PATH)
    results_count = 0
    
    with open(CANDIDATES_PATH, 'r', encoding='utf-8') as f_in, \
         open(ALIGNMENT_RESULTS_PATH, 'a', encoding='utf-8') as f_out:
        
        lines = f_in.readlines()
        total = len(lines)
        
        for line in tqdm(lines, desc="LLM Alignment"):
            data = json.loads(line)
            main_node = data["main_node"]
            
            # 跳过已处理节点
            if main_node["id"] in processed_ids:
                continue
            
            candidate_names = [c["name"] for c in data["candidates"]]
            response = call_llm(main_node["name"], candidate_names)
            
            if response == "None":
                result = {
                    "main_id": main_node["id"],
                    "main_name": main_node["name"],
                    "aligned_entities": []
                }
            else:
                try:
                    # 尝试解析JSON
                    parsed = json.loads(response)
                    aligned_names = parsed.get("aligned_entities", [])
                    
                    # 匹配候选ID
                    aligned_ids = [
                        c["candidate_id"] for c in data["candidates"]
                        if c["name"] in aligned_names
                    ]
                    
                    result = {
                        "main_id": main_node["id"],
                        "main_name": main_node["name"],
                        "aligned_entities": aligned_ids
                    }
                except:
                    # 解析失败时跳过
                    continue
            
            # 仅保存有对齐结果的条目
            if result["aligned_entities"]:
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                results_count += 1
            
            # 避免API限流
            time.sleep(0.5)
    
    print(f"Alignment completed. {results_count} valid alignments saved to {ALIGNMENT_RESULTS_PATH}")

if __name__ == "__main__":
    if not DASHSCOPE_API_KEY:
        raise ValueError("DASHSCOPE_API_KEY not set in environment variables")
    process_alignment()
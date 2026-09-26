import pandas as pd
from rapidfuzz import fuzz
import os
import gc

def shrink_candidates(candidates_path, s1_path, s2_path, s3_path, out_path):
    print("Loading datasets for shrinking...")
    cands = pd.read_csv(candidates_path, sep="\t", dtype=str)
    
    # Load just IDs and names
    s1 = pd.read_csv(s1_path, sep="\t", dtype=str, usecols=['entity_id', 'norm_name'])
    s2 = pd.read_csv(s2_path, sep="\t", dtype=str, usecols=['entity_id', 'norm_name'])
    s3 = pd.read_csv(s3_path, sep="\t", dtype=str, usecols=['entity_id', 'norm_name'])
    
    s1_dict = s1.set_index('entity_id')['norm_name'].to_dict()
    
    pool = pd.concat([s2, s3], ignore_index=True)
    pool_dict = pool.set_index('entity_id')['norm_name'].to_dict()
    del s2, s3, pool
    gc.collect()
    
    out_lines = []
    print("Shrinking candidate sets...")
    
    for _, row in cands.iterrows():
        s1_id = row['source1_entity_id']
        c_str = row['candidate_entity_ids']
        
        if pd.isna(c_str) or not c_str:
            out_lines.append(f"{s1_id}\t\n")
            continue
            
        c_list = c_str.split(',')
        s1_name = s1_dict.get(s1_id, "")
        if not isinstance(s1_name, str): s1_name = ""
            
        filtered_cands = []
        for c_id in c_list:
            c_name = pool_dict.get(c_id, "")
            if not isinstance(c_name, str): c_name = ""
                
            # Cheap secondary filter using rapidfuzz string similarity
            # token_set_ratio is good for unordered words
            sim = fuzz.token_set_ratio(s1_name, c_name)
            
            # Keep if similarity is above a certain threshold
            if sim > 40: 
                filtered_cands.append(c_id)
                
        out_lines.append(f"{s1_id}\t{','.join(filtered_cands)}\n")
        
    with open(out_path, "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        f.writelines(out_lines)
        
    print(f"Saved trimmed candidates to {out_path}")

if __name__ == "__main__":
    data_dir = os.path.join("student_resource", "normalized_dataset", "train")
    shrink_candidates(
        "candidate_pairs.tsv", 
        os.path.join(data_dir, "train_source1.tsv"),
        os.path.join(data_dir, "train_source2.tsv"),
        os.path.join(data_dir, "train_source3.tsv"),
        "candidate_pairs_shrunk.tsv"
    )

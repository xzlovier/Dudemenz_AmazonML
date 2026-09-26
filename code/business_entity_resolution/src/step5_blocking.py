import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import jellyfish
from datasketch import MinHash, MinHashLSH
import os
import gc

def generate_candidates_country(s1_df, s2_df, s3_df, country):
    print(f"--- Blocking for Country: {country} ---")
    s1 = s1_df[s1_df['country'] == country].reset_index(drop=True)
    s_pool = pd.concat([
        s2_df[s2_df['country'] == country],
        s3_df[s3_df['country'] == country]
    ], ignore_index=True)
    
    if len(s1) == 0 or len(s_pool) == 0:
        return {}

    s1_ids = s1['entity_id'].values
    pool_ids = s_pool['entity_id'].values
    
    s1_names = s1['norm_name'].fillna("").values
    pool_names = s_pool['norm_name'].fillna("").values
    
    # 1. TF-IDF + FAISS
    print("1. TF-IDF + FAISS ANN retrieval...")
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
    pool_vecs = vectorizer.fit_transform(pool_names)
    s1_vecs = vectorizer.transform(s1_names)
    
    pool_vecs_dense = pool_vecs.astype('float32').toarray()
    s1_vecs_dense = s1_vecs.astype('float32').toarray()
    
    # L2 normalize for cosine similarity
    faiss.normalize_L2(pool_vecs_dense)
    faiss.normalize_L2(s1_vecs_dense)
    
    dim = pool_vecs_dense.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(pool_vecs_dense)
    
    k = 50 # top-k
    D, I = index.search(s1_vecs_dense, k)
    
    candidates = {s1_id: set() for s1_id in s1_ids}
    
    for i, s1_id in enumerate(s1_ids):
        for j in range(k):
            if I[i, j] >= 0 and D[i, j] > 0.4: # threshold to avoid total junk
                candidates[s1_id].add(pool_ids[I[i, j]])
                
    # Free memory
    del pool_vecs, s1_vecs, pool_vecs_dense, s1_vecs_dense, index
    gc.collect()
    
    # 2. MinHash LSH
    print("2. MinHash LSH on names...")
    lsh = MinHashLSH(threshold=0.6, num_perm=128)
    
    pool_minhashes = []
    for i, name in enumerate(pool_names):
        m = MinHash(num_perm=128)
        for d in name.split():
            m.update(d.encode('utf8'))
        lsh.insert(pool_ids[i], m)
        pool_minhashes.append(m)
        
    for i, name in enumerate(s1_names):
        m = MinHash(num_perm=128)
        for d in name.split():
            m.update(d.encode('utf8'))
        result = lsh.query(m)
        for r in result:
            candidates[s1_ids[i]].add(r)
            
    del lsh, pool_minhashes
    gc.collect()

    # 3. Phonetic Blocking (Soundex/Metaphone)
    print("3. Phonetic blocking (Metaphone) on first word...")
    
    from collections import defaultdict
    pool_phonetic = defaultdict(list)
    for i, name in enumerate(pool_names):
        words = name.split()
        if words:
            code = jellyfish.metaphone(words[0])
            if code:
                pool_phonetic[code].append(pool_ids[i])
                
    for i, name in enumerate(s1_names):
        words = name.split()
        if words:
            code = jellyfish.metaphone(words[0])
            if code and code in pool_phonetic:
                # To prevent massive explosion, only add if the bucket is reasonably sized
                if len(pool_phonetic[code]) < 1000:
                    for pid in pool_phonetic[code]:
                        candidates[s1_ids[i]].add(pid)
                        
    return candidates

def generate_all_candidates():
    print("Loading normalized datasets...")
    data_dir = os.path.join("student_resource", "normalized_dataset", "train")
    # For large datasets, in reality this needs chunking or spark. 
    # Here we assume it fits in memory or we're processing a sample.
    s1 = pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t", dtype=str)
    s2 = pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t", dtype=str)
    s3 = pd.read_csv(os.path.join(data_dir, "train_source3.tsv"), sep="\t", dtype=str)
    
    all_candidates = {}
    countries = s1['country'].unique()
    for c in countries:
        if pd.isna(c): continue
        cands = generate_candidates_country(s1, s2, s3, c)
        all_candidates.update(cands)
        
    # Write output
    out_lines = []
    for s1_id, c_set in all_candidates.items():
        if len(c_set) > 0:
            out_lines.append(f"{s1_id}\t{','.join(c_set)}\n")
        else:
            out_lines.append(f"{s1_id}\t\n")
            
    with open("candidate_pairs.tsv", "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        f.writelines(out_lines)
    print("Saved candidate_pairs.tsv")

if __name__ == "__main__":
    generate_all_candidates()

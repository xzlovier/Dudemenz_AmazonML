import pandas as pd
import os
import random

data_dir = os.path.join("student_resource", "dataset")

print("Loading S1 and Ground Truth...")
gt = pd.read_csv(os.path.join(data_dir, "train", "train_ground_truth.tsv"), sep="\t", dtype=str)
s1 = pd.read_csv(os.path.join(data_dir, "train", "train_source1.tsv"), sep="\t", dtype=str)

# 1. Cardinality
gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna("")
def count_matches(x):
    return 0 if not x else len(x.split(','))
gt['match_count'] = gt['matched_entity_ids'].apply(count_matches)

print("\n--- Match Cardinality (Fraction of S1) ---")
print(gt['match_count'].apply(lambda c: "Singleton" if c==0 else ("1-match" if c==1 else "Many-match")).value_counts(normalize=True) * 100)

# 2. Country distribution
print("\n--- S1 Train Country Distribution ---")
print(s1['country'].value_counts(normalize=True) * 100)

test_s1 = pd.read_csv(os.path.join(data_dir, "test", "test_source1.tsv"), sep="\t", dtype=str)
print("\n--- S1 Test Country Distribution ---")
print(test_s1['country'].value_counts(normalize=True) * 100)

# France sample
if "France" in test_s1['country'].values:
    print("\nFrance addresses sample:")
    print(test_s1[test_s1['country'] == 'France'][['business_address']].head(5))

# 3. Fast extraction of 50 pairs
print("\nSampling 50 True Positive pairs...")
gt_with_matches = gt[gt['match_count'] > 0]
sample_gt = gt_with_matches.sample(50, random_state=42)

# Collect required S2 and S3 IDs
needed_s2 = set()
needed_s3 = set()
for _, row in sample_gt.iterrows():
    matches = row['matched_entity_ids'].split(',')
    for m in matches:
        if m.startswith('S2-'): needed_s2.add(m)
        elif m.startswith('S3-'): needed_s3.add(m)

print(f"Need {len(needed_s2)} from S2, {len(needed_s3)} from S3.")

# Read S2 and S3 in chunks and filter
s2_matches = {}
for chunk in pd.read_csv(os.path.join(data_dir, "train", "train_source2.tsv"), sep="\t", dtype=str, chunksize=100000):
    filtered = chunk[chunk['entity_id'].isin(needed_s2)]
    for _, r in filtered.iterrows():
        s2_matches[r['entity_id']] = r
    if len(s2_matches) == len(needed_s2): break

s3_matches = {}
for chunk in pd.read_csv(os.path.join(data_dir, "train", "train_source3.tsv"), sep="\t", dtype=str, chunksize=100000):
    filtered = chunk[chunk['entity_id'].isin(needed_s3)]
    for _, r in filtered.iterrows():
        s3_matches[r['entity_id']] = r
    if len(s3_matches) == len(needed_s3): break

# Write out the sample pairs
with open("eda_sample_pairs.txt", "w", encoding="utf-8") as f:
    for idx, row in sample_gt.iterrows():
        s1_id = row['source1_entity_id']
        s1_row = s1[s1['entity_id'] == s1_id].iloc[0]
        f.write(f"S1: {s1_id} | {s1_row['business_name']} | {s1_row['business_address']} | {s1_row['country']}\n")
        
        matches = row['matched_entity_ids'].split(',')
        for m in matches:
            if m.startswith('S2-') and m in s2_matches:
                match_data = s2_matches[m]
                f.write(f" -> {m} (S2) | {match_data['business_name']} | {match_data['business_address']}\n")
            elif m.startswith('S3-') and m in s3_matches:
                match_data = s3_matches[m]
                f.write(f" -> {m} (S3) | {match_data['business_name']} | {match_data['business_address']}\n")
            else:
                f.write(f" -> {m} (NOT FOUND)\n")
        f.write("-" * 80 + "\n")
print("Saved 50 sample pairs to eda_sample_pairs.txt")
